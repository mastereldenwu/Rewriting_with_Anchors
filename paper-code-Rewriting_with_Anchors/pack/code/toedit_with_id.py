#!/usr/bin/env python3
"""
基于 ToEdit 策略生成半合成数据（保留源数据 id，记录编辑信息）。
输入：带 id 字段的 .jsonl 或 .jsonl.gz 文件
输出：每行包含 id, text, method, is_edited, edit_ratio 的 .jsonl 文件
"""

import json
import gzip
import os
import argparse
import numpy as np
from tqdm import tqdm
from vllm import LLM, SamplingParams


def smart_open(filepath, mode='rt'):
    if filepath.endswith('.gz'):
        return gzip.open(filepath, mode, encoding='utf-8')
    return open(filepath, mode, encoding='utf-8')


def load_source_with_id(source_file):
    """加载带 id 的源数据，返回 (ids, texts) 列表"""
    ids, texts = [], []
    with smart_open(source_file, 'rt') as f:
        for line in f:
            obj = json.loads(line.strip())
            if 'id' not in obj or 'text' not in obj:
                continue
            ids.append(obj['id'])
            texts.append(obj['text'])
    print(f"成功加载 {len(ids)} 条源数据（含 id）")
    return ids, texts


def count_existing(output_file):
    """返回输出文件中已有有效记录的数量（用于断点续传）"""
    if not os.path.exists(output_file):
        return 0
    count = 0
    with smart_open(output_file, 'rt') as f:
        for line in f:
            if line.strip():
                count += 1
    return count


def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)


def apply_toedit_strategy(outputs, tokenizer, strategy="up_revise"):
    """
    对 vLLM 生成的 outputs 应用 ToEdit 策略，返回编辑后的文本列表及编辑信息。
    返回：list of dict: {"text": edited_text, "is_edited": bool, "edit_ratio": float}
    """
    up_threshold = 0.985
    low_threshold = 0.001

    def resampling(prob_dict, beta=1.5):
        words = list(prob_dict.keys())
        scores = np.array([prob_dict[w] for w in words])
        adjusted = softmax(scores / beta)
        return np.random.choice(words, p=adjusted)

    def convert_prob_dict(prompt_logprob):
        return {k: v.logprob for k, v in prompt_logprob.items()}

    results = []
    for output in outputs:
        final_tokens = []
        edited_count = 0
        total_count = 0

        # 跳过第一个 None
        for i, prompt_logprob in enumerate(output.prompt_logprobs):
            if prompt_logprob is None:
                # 保留原始 token
                final_tokens.append(output.prompt_token_ids[i])
                continue

            total_count += 1
            prob_dict = convert_prob_dict(prompt_logprob)
            token_id = output.prompt_token_ids[i]
            prob = np.exp(prob_dict[token_id])

            if strategy == "up_revise":
                if prob > up_threshold:
                    new_token = resampling(prob_dict)
                    final_tokens.append(new_token)
                    if new_token != token_id:
                        edited_count += 1
                else:
                    final_tokens.append(token_id)
            elif strategy == "low_drop_up_revise":
                if prob > up_threshold:
                    new_token = resampling(prob_dict)
                    final_tokens.append(new_token)
                    if new_token != token_id:
                        edited_count += 1
                elif prob < low_threshold:
                    # 丢弃该 token，不加入 final_tokens
                    edited_count += 1  # 视为编辑（删除）
                    continue
                else:
                    final_tokens.append(token_id)
            elif strategy == "filtering":
                if prob >= 0.00001:  # threshold from original code
                    final_tokens.append(token_id)
                else:
                    # 被过滤掉，视为编辑
                    edited_count += 1
                    continue

        revised_text = tokenizer.decode(final_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        is_edited = edited_count > 0
        edit_ratio = edited_count / total_count if total_count > 0 else 0.0
        results.append({
            "text": revised_text,
            "is_edited": is_edited,
            "edit_ratio": edit_ratio
        })
    return results


def main():
    parser = argparse.ArgumentParser(description="生成 ToEdit 半合成数据（带 id）")
    parser.add_argument('-s', '--source_data', default="../data/marked.jsonl", help='带 id 的源数据文件')
    parser.add_argument('-o', '--output_file', default="../data/synthetic_toedit-test.jsonl", help='输出文件路径 (.jsonl 推荐)')
    parser.add_argument('--model_path', default='../model/Qwen2.5-0.5B-Instruct')
    parser.add_argument('--max_model_len', type=int, default=2048)
    parser.add_argument('--gpu_memory_utilization', type=float, default=0.9)
    parser.add_argument('--max_num_seqs', type=int, default=128)
    parser.add_argument('--batch_size', type=int, default=128, help='每批处理的提示数')
    parser.add_argument('--strategy', default='up_revise', choices=['up_revise', 'low_drop_up_revise', 'filtering'])
    parser.add_argument('--n_candidates', type=int, default=8, help='返回的 logprob 候选数')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--num_variants', type=int, default=1, help='每条源数据生成的变体数（暂不支持>1）')
    parser.add_argument('--max_samples',type=int,default=10000,help='每次生成样本数量最大量')
    args = parser.parse_args()

    if args.num_variants != 1:
        print("警告：当前版本仅支持 num_variants=1，已自动调整。")
        args.num_variants = 1

    # 1. 加载源数据
    ids, texts = load_source_with_id(args.source_data)
    if not ids:
        raise ValueError("源数据为空")

    # 2. 加载模型
    print(f"加载模型: {args.model_path}")
    llm = LLM(
        model=args.model_path,
        tensor_parallel_size=1,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        max_logprobs=args.n_candidates,

    )
    tokenizer = llm.get_tokenizer()

    # 3. 断点续传
    start_idx = count_existing(args.output_file)
    total = len(ids)
    if start_idx >= total:
        print(f"所有 {total} 条已生成完毕，退出。")
        return
    if start_idx > 0:
        print(f"检测到已有 {start_idx} 条记录，将从第 {start_idx} 条继续生成。")

    # 4. 采样参数：max_tokens=1，仅获取 logprobs
    sampling_params = SamplingParams(
        max_tokens=1,
        prompt_logprobs=args.n_candidates,
    )

    # 5. 输出文件准备（追加模式）
    os.makedirs(os.path.dirname(args.output_file) or '.', exist_ok=True)
    mode = 'a' if start_idx > 0 else 'w'

    if args.max_samples is not None and args.max_samples < len(ids):
        ids = ids[:args.max_samples]
        texts = texts[:args.max_samples]
        print(f"限制处理前 {args.max_samples} 条样本")


    # 分批处理
    remaining_ids = ids[start_idx:]
    remaining_texts = texts[start_idx:]
    total_remaining = len(remaining_ids)

    with open(args.output_file, mode, encoding='utf-8') as fout, \
         tqdm(total=total, initial=start_idx, desc="ToEdit 生成进度", unit="条",
              bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]") as pbar:

        batch_start = 0
        while batch_start < total_remaining:
            batch_end = min(batch_start + args.batch_size, total_remaining)
            batch_ids = remaining_ids[batch_start:batch_end]
            batch_texts = remaining_texts[batch_start:batch_end]

            # vLLM 生成（仅获取 logprobs）
            outputs = llm.generate(batch_texts, sampling_params, use_tqdm=False)

            # 应用 ToEdit 策略
            edited_results = apply_toedit_strategy(outputs, tokenizer, args.strategy)

            # 保存结果
            for orig_id, res in zip(batch_ids, edited_results):
                record = {
                    "id": orig_id,
                    "text": res["text"],
                    "method": "toedit",
                    "is_edited": res["is_edited"],
                    "edit_ratio": round(res["edit_ratio"], 6)
                }
                fout.write(json.dumps(record, ensure_ascii=False) + '\n')

            fout.flush()
            pbar.update(batch_end - batch_start)
            batch_start = batch_end

    print(f"✅ ToEdit 半合成数据生成完毕！输出: {args.output_file}")


if __name__ == '__main__':
    main()