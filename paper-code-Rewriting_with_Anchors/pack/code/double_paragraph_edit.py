#!/usr/bin/env python3
"""
句子级生成式数据增强 (llama-server)
- 分句、合并短句
- 按奇偶掩码选择目标位置
- 根据位置类型：有上下文改写、仅上文改写、末尾续写
- 使用 /v1/completions 端点
- 清洗：去除 <think>...</think>，提取首句 (10‑25 字)
- 递归降温重试
"""

import json
import gzip
import re
import argparse
import random
import torch
import requests
from types import SimpleNamespace
from concurrent.futures import ThreadPoolExecutor, as_completed

# ======================= 参数解析 =======================
def parse_args():
    parser = argparse.ArgumentParser(description="句子级生成式数据增强 (llama-server)")
    parser.add_argument('-s', '--source_data', required=True,help='原真实数据')
    parser.add_argument('--output_even',default='../data/synthetic_even.jsonl',help='偶数组数据输出')
    parser.add_argument('--output_odd', default='../data/synthetic_odd.jsonl',help='奇数组数据输出')
    parser.add_argument('--model_path', default="../model/Qwen2.5-0.5B-Instruct", help='tokenizer 路径（仅分句）')
    parser.add_argument('--server_url', default="http://127.0.0.1:8080",help='服务器地址')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--min_sentence_tokens', type=int, default=20,help='最短分句长度')
    parser.add_argument('--max_new_tokens', type=int, default=512, help='生成 token 数（需足够容纳思考+正文）')
    parser.add_argument('--temperature', type=float, default=0.8)
    parser.add_argument('--top_p', type=float, default=0.85)
    parser.add_argument('--max_samples', type=int, default=5000)
    parser.add_argument('--micro_batch_size', type=int, default=16)
    parser.add_argument('--debug', action='store_true',)
    parser.add_argument('--batch_size', type=int, default=32,help='断点续传时每批处理的样本数（奇偶交替处理）')
    return parser.parse_args()

# ======================= 文件加载 =======================
def smart_open(path, mode='rt'):
    if path.endswith('.gz'):
        return gzip.open(path, mode, encoding='utf-8')
    return open(path, mode, encoding='utf-8')

def load_source(path):
    data = []
    with smart_open(path, 'rt') as f:
        for line in f:
            obj = json.loads(line.strip())
            if 'id' in obj and 'text' in obj:
                data.append(obj)
    return data

# ======================= 文本处理 =======================
def split_sentences(text):
    pattern = r'([^。！？；]+[。！？；])'
    parts = re.findall(pattern, text)
    remaining = re.sub(pattern, '', text)
    sentences = parts + ([remaining] if remaining.strip() else [])
    return [s.strip() for s in sentences if s.strip()]

def merge_short_sentences(sentences, tokenizer, min_tokens):
    """确保每个单元不小于 min_tokens，全文过短则单句"""
    if not sentences:
        return []
    tok_counts = [len(tokenizer.encode(s, add_special_tokens=False)) for s in sentences]
    total = sum(tok_counts)
    if total < min_tokens:
        return ["".join(sentences).strip()]
    merged = []
    cur_sents = []
    cur_tokens = 0
    for s, t in zip(sentences, tok_counts):
        cur_sents.append(s)
        cur_tokens += t
        if cur_tokens >= min_tokens:
            merged.append("".join(cur_sents).strip())
            cur_sents = []
            cur_tokens = 0
    if cur_sents:
        if merged:
            merged[-1] = merged[-1] + "".join(cur_sents)
        else:
            merged.append("".join(cur_sents).strip())
    return merged

# ======================= 提示词构建 =======================
def build_rewrite_with_context(prev_sentence, target_sentence, next_sentence):
    if not prev_sentence.strip():
        prev_sentence = "（故事开头）"
    return (
        f"你是一个网文作者，请根据上下文改写句子，保持风格和字数（25‑50字）：\n"
        f"上文：{prev_sentence}\n"
        f"原句：{target_sentence}\n"
        f"下文：{next_sentence}\n"
        f"只输出改写后的句子："
    )

def build_rewrite_no_context(prev_sentence, target_sentence):
    if not prev_sentence.strip():
        prev_sentence = "（故事开头）"
    return (
        f"你是一个网文作者，请根据上文改写句子，保持风格和字数（25‑50字）：\n"
        f"上文：{prev_sentence}\n"
        f"原句：{target_sentence}\n"
        f"只输出改写后的句子："
    )

def build_continuation_only(prev_sentence):
    if not prev_sentence.strip():
        prev_sentence = "（故事开头）"
    return (
        f"你是一个网文作者，请根据上文续写一句话，25‑50字，保持风格连贯：\n"
        f"上文：{prev_sentence}\n"
        f"续写："
    )

# ======================= 清洗函数 =======================
def clean_after_think(raw):
    """删除最后一个 </think> 及其之前的所有内容（若无则返回原文本）"""
    if '</think>' in raw:
        # 取最后一个 </think> 之后的部分
        return raw.rsplit('</think>', 1)[-1].strip()
    return raw.strip()

def extract_generated_sentence(output_text, min_chars=25, max_chars=50):
    """提取第一个长度合适且以句末标点结尾的句子，否则硬截断"""
    if not output_text:
        return ""
    # 规范化空白
    text = re.sub(r'\s+', ' ', output_text.replace('\n', ' ').replace('\r', ' ')).strip()
    if not text:
        return ""
    # 按句末标点切分
    sentences = re.split(r'(?<=[。！？；])', text)
    for s in sentences:
        s = s.strip()
        if not s:
            continue
        if min_chars <= len(s) <= max_chars:
            return s
    # 如果没有任何句子符合长度，则硬截断
    # 找到第一个句末标点，截断到该位置
    for s in sentences:
        s = s.strip()
        if not s:
            continue
        if len(s) < min_chars:
            continue
        # 长度超限则截断到 max_chars，尽量保留标点
        truncated = s[:max_chars]
        last_punct = max(truncated.rfind('。'), truncated.rfind('！'),
                         truncated.rfind('？'), truncated.rfind('；'))
        if last_punct != -1:
            return truncated[:last_punct+1]
        return truncated
    # 如果没有标点，硬截断全文
    if len(text) >= min_chars:
        truncated = text[:max_chars]
        return truncated
    return ""
# ======================= llama-server 交互 =======================
def llama_text_generate(prompt_text, temp, sampling_params, server_url):
    payload = {
        "prompt": prompt_text,
        "temperature": temp,
        "top_p": sampling_params.top_p,
        "max_tokens": sampling_params.max_tokens,
        "stop": sampling_params.stop,
        "stream": False,
    }
    try:
        resp = requests.post(f"{server_url}/v1/completions", json=payload, timeout=180)
        resp.raise_for_status()
        data = resp.json()
        content = data["choices"][0]["text"]
        finish_reason = data["choices"][0].get("finish_reason", "length")
    except Exception as e:
        print(f"请求失败: {e}")
        content = ""
        finish_reason = "error"

    class FakeOutput:
        def __init__(self, text, reason):
            self.outputs = [SimpleNamespace(text=text, finish_reason=reason)]
    return FakeOutput(content, finish_reason)

def batch_generate(prompts, temp, sampling_params, server_url, max_workers=8):
    results = [None] * len(prompts)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_idx = {
            executor.submit(llama_text_generate, p, temp, sampling_params, server_url): i
            for i, p in enumerate(prompts)
        }
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            results[idx] = future.result()
    return results

# ======================= 主生成逻辑 =======================
def batch_generate_variants(data, tokenizer, sampling_params, args, mask_type, debug_f, server_url):
    # 第一步：切分、合并短句
    samples = []
    for rec in data:
        sentences = split_sentences(rec['text'])
        sentences = merge_short_sentences(sentences, tokenizer, args.min_sentence_tokens)
        samples.append({
            'id': rec['id'],
            'sentences': sentences,
            'new_sentences': list(sentences),  # 拷贝，后续修改
        })

    # 第二步：收集任务
    all_prompts = []
    task_map = []  # (sample_idx, pos, original_sentence, prev_sentence, task_type)
    # task_type: 'rewrite_context', 'rewrite_no_context', 'continuation'

    for idx, sample in enumerate(samples):
        sents = sample['sentences']
        # 找出需要处理的奇偶位置
        target_positions = [i for i in range(len(sents))
                            if (i % 2 == 0 and mask_type == 'even') or (i % 2 != 0 and mask_type == 'odd')]
        if not target_positions:
            continue

        total_sents = len(sents)
        for pos in target_positions:
            # 上文

            if pos == 0:
                continue

            prev_sent = sents[pos - 1] if pos > 0 else ""

            # if pos == total_sents - 1:
            #     # 末句：续写
            #     prompt = build_continuation_only(prev_sent)
            #     task_map.append((idx, pos, None, prev_sent, 'continuation'))
            if pos >= total_sents - 2:
                # 倒数第二句：只有上文和原句改写
                original = sents[pos]
                prompt = build_rewrite_no_context(prev_sent, original)
                task_map.append((idx, pos, original, prev_sent, 'rewrite_no_context'))
            else:
                # 其余：有上下文改写
                original = sents[pos]
                next_sent = sents[pos + 1]
                prompt = build_rewrite_with_context(prev_sent, original, next_sent)
                task_map.append((idx, pos, original, prev_sent, 'rewrite_context'))

            all_prompts.append(prompt)

    if not task_map:
        return ["".join(s['new_sentences']) for s in samples]

    total_tasks = len(task_map)
    completed_tasks = 0
    print(f"共有 {total_tasks} 个续写任务 ({mask_type})")

    # 递归降温参数
    current_temp = args.temperature
    max_retries = 3
    min_temp = 3
    temp_decay = 1.2

    generation_results = [None] * total_tasks
    pending_indices = list(range(total_tasks))
    pending_prompts = all_prompts[:]

    for retry in range(max_retries + 1):
        if not pending_indices:
            break
        if debug_f:
            debug_f.write(f"\n=== 重试 {retry+1}，温度={current_temp:.3f}，待处理 {len(pending_indices)} ===\n")

        # 分微批
        batches = [pending_prompts[i:i+args.micro_batch_size] for i in range(0, len(pending_prompts), args.micro_batch_size)]
        idx_batches = [pending_indices[i:i+args.micro_batch_size] for i in range(0, len(pending_indices), args.micro_batch_size)]

        new_pending_indices = []
        new_pending_prompts = []

        for b_no, (b_prompts, b_idx) in enumerate(zip(batches, idx_batches)):
            print(f"[{mask_type}] 第{retry+1}轮 批次 {b_no+1}/{len(batches)}, 大小={len(b_prompts)}", flush=True)
            outputs = batch_generate(b_prompts, current_temp, sampling_params, server_url)

            for orig_idx, out in zip(b_idx, outputs):
                sample_idx, pos, original, prev_sent, task_type = task_map[orig_idx]
                raw = out.outputs[0].text
                reason = out.outputs[0].finish_reason

                if debug_f:
                    debug_f.write(f"\n--- ID {samples[sample_idx]['id']} pos {pos} ---\n")
                    debug_f.write(f"PROMPT: {all_prompts[orig_idx]}\n")  # 添加完整提示词
                    debug_f.write(f"PREV: {prev_sent[-50:]}\n")
                    debug_f.write(f"RAW: {raw}\n")
                    debug_f.write(f"REASON: {reason}\n")

                # 非法停止词检测
                stop_words = sampling_params.stop
                if reason == "stop" and any(w in raw for w in stop_words):
                    if debug_f: debug_f.write(">>> 非法停止词，失败\n")
                    new_pending_indices.append(orig_idx)
                    new_pending_prompts.append(pending_prompts[pending_indices.index(orig_idx)])
                    continue

                # 清洗：去思考 + 提取句子
                cleaned = ""
                raw_clean = clean_after_think(raw)
                if raw_clean:
                    cleaned = extract_generated_sentence(raw_clean)
                if debug_f:
                    debug_f.write(f"CLEANED: {cleaned}\n")

                # 判定失败
                fail = False
                if not cleaned:
                    fail = True
                elif task_type in ('rewrite_context', 'rewrite_no_context'):
                    if original and (cleaned == original or cleaned == prev_sent):
                        fail = True
                elif task_type == 'continuation':
                    if cleaned == prev_sent:
                        fail = True

                if fail:
                    if debug_f: debug_f.write(">>> 失败\n")
                    new_pending_indices.append(orig_idx)
                    new_pending_prompts.append(pending_prompts[pending_indices.index(orig_idx)])
                else:
                    generation_results[orig_idx] = cleaned
                    completed_tasks += 1
                    print(f"[{mask_type}] {completed_tasks}/{total_tasks} completed", flush=True)
                    if debug_f: debug_f.write(">>> 接受\n")

        pending_indices = new_pending_indices
        pending_prompts = new_pending_prompts
        current_temp = min(min_temp, current_temp * temp_decay)

    # 第三阶段：最终回退
    for orig_idx in pending_indices:
        sample_idx, pos, original, prev_sent, task_type = task_map[orig_idx]
        if task_type in ('rewrite_context', 'rewrite_no_context'):
            generation_results[orig_idx] = original   # 回退原文
        else:
            generation_results[orig_idx] = ""          # 续写失败，不添加新句
        completed_tasks += 1
        print(f"[{mask_type}] {completed_tasks}/{total_tasks} completed (fallback)", flush=True)
        if debug_f:
            debug_f.write(f"\n--- ID {samples[sample_idx]['id']} pos {pos} 回退 ---\n")

    # 第四阶段：应用结果到 new_sentences
    for i, (sample_idx, pos, original, _, task_type) in enumerate(task_map):
        result = generation_results[i]
        if result:
            result = result.replace('\\"', '"')
        if result is None or result == "":
            continue
        if task_type == 'continuation':
            # 在最后一句之后插入续写句
            samples[sample_idx]['new_sentences'].insert(pos + 1, result)
        else:
            # 替换原位置句子
            samples[sample_idx]['new_sentences'][pos] = result

    # 返回拼合后的全文
    return ["".join(s['new_sentences']) for s in samples]

# ======================= 主函数 =======================
import os

def main():
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    print("加载数据...")
    data = load_source(args.source_data)
    if not data:
        raise ValueError("无数据")
    if args.max_samples and args.max_samples < len(data):
        data = data[:args.max_samples]
        print(f"限制为前 {args.max_samples} 条")

    # 确保按 id 排序（如果原始数据未排序）
    data.sort(key=lambda x: x['id'])

    # ---------- 断点续传：只读最后完成的 id ----------
    progress_file = "progress.txt"
    last_done = -1  # 默认从 0 开始
    if os.path.exists(progress_file):
        with open(progress_file, "r", encoding="utf-8") as pf:
            content = pf.read().strip()
            if content.isdigit():
                last_done = int(content)
                print(f"进度文件记录最后完成 id = {last_done}")
            else:
                print("进度文件格式异常，从 0 开始处理")
    else:
        print("未找到进度文件，从头开始处理")

    # 过滤已完成的样本
    remaining_data = [rec for rec in data if rec['id'] > last_done]
    print(f"剩余待处理样本: {len(remaining_data)}/{len(data)} (从 id={last_done+1} 开始)")

    if not remaining_data:
        print("所有样本已完成，退出。")
        return

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

    sampling_params = SimpleNamespace(
        max_tokens=args.max_new_tokens,
        top_p=args.top_p,
        stop=["<｜end▁of▁sentence｜>", "<|im_end|>", "<|im_start|>", "\n\n\n", "\\"],
    )

    debug_f = open("../PythonProject2/debug.log", "w", encoding="utf-8") if args.debug else None

    # ---------- 分批处理 ----------
    batch_size = args.batch_size
    total_batches = (len(remaining_data) + batch_size - 1) // batch_size
    print(f"\n开始分批处理，共 {total_batches} 批次，每批最多 {batch_size} 条")

    for batch_idx in range(0, len(remaining_data), batch_size):
        batch = remaining_data[batch_idx:batch_idx + batch_size]
        batch_no = batch_idx // batch_size + 1
        print(f"\n>>> 批次 {batch_no}/{total_batches}，样本 id {batch[0]['id']} ~ {batch[-1]['id']}")

        # ------ 偶数生成 ------
        print("  生成偶数变体...")
        even_texts = batch_generate_variants(batch, tokenizer, sampling_params, args,
                                             'even', debug_f, args.server_url)
        # 写入偶数输出文件
        with open(args.output_even, 'a', encoding='utf-8') as f:
            for rec, txt in zip(batch, even_texts):
                f.write(json.dumps({"id": rec['id'], "text": txt, "method": "qwen35_even"},
                                   ensure_ascii=False) + '\n')

        # ------ 奇数生成 ------
        print("  生成奇数变体...")
        odd_texts = batch_generate_variants(batch, tokenizer, sampling_params, args,
                                            'odd', debug_f, args.server_url)
        with open(args.output_odd, 'a', encoding='utf-8') as f:
            for rec, txt in zip(batch, odd_texts):
                f.write(json.dumps({"id": rec['id'], "text": txt, "method": "qwen35_odd"},
                                   ensure_ascii=False) + '\n')

        # ------ 更新进度文件：写入本批最后一个 id ------
        last_id = batch[-1]['id']               # 本批最大 id
        with open(progress_file, 'w', encoding='utf-8') as pf:
            pf.write(str(last_id))
        print(f"  批次完成，进度已更新为 {last_id}")

    if debug_f:
        debug_f.close()
    print("\n全部处理完成。")
if __name__ == '__main__':
    main()