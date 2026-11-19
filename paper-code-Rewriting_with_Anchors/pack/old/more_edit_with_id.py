#!/usr/bin/env python3
"""
改进的 ToEdit 半合成数据生成脚本（新方法）
- 只允许编辑非特殊 token（自动跳过 BOS/EOS 等）
- 多窗口候选中，所有包含当前位置且长度 ≤ max_ngram 的窗口平等参与采样
- 可重复性：固定随机种子
- 稳定断点续传：基于输出文件行数与进度文件对齐
- 阻止 max_ngram > 4 的组合爆炸
- 输出带 id、is_edited、edit_ratio 的 .jsonl
"""

import os
import sys
import json
import gzip
import argparse
import time
import numpy as np
import random
from tqdm import tqdm
from vllm import LLM, SamplingParams


# --------------------------- 参数解析 ---------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="改进的半合成数据生成（新方法）")
    parser.add_argument('-s', '--source_data', default="./sampled_data/marked.jsonl", help='带 id 的源数据文件 (.jsonl 或 .jsonl.gz)')
    parser.add_argument('-o', '--output_file', default="./synthetic_data/gen2.jsonl" ,help='输出文件路径 (.jsonl 推荐)')
    parser.add_argument('--model_path', default='/home/laowu/.cache/modelscope/hub/models/Qwen/Qwen2.5-0.5B-Instruct')
    parser.add_argument('--max_model_len', type=int, default=2048)
    parser.add_argument('--gpu_memory_utilization', type=float, default=0.85)
    parser.add_argument('--max_num_seqs', type=int, default=128)
    parser.add_argument('--batch_size', type=int, default=128, help='每批处理的样本数')
    parser.add_argument('--n_candidates', type=int, default=8, help='prompt_logprobs 返回的候选数')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')

    # 编辑参数
    parser.add_argument('--up_threshold', type=float, default=0.99, help='高置信度阈值')
    parser.add_argument('--beta', type=float, default=1.5, help='单 token 替换的 softmax 温度')
    parser.add_argument('--max_ngram', type=int, default=3, help='最大 n-gram 窗口长度（<=4）')
    parser.add_argument('--ngram_threshold', type=float, default=0.85, help='窗口内所有 token 需要达到的最低概率')
    parser.add_argument('--ngram_topk', type=int, default=4, help='每个位置保留的候选 token 数')
    parser.add_argument('--ngram_temperature', type=float, default=1.8, help='多窗口候选采样的温度')
    parser.add_argument('--num_candidates', type=int, default=6, help='每个原始样本生成的候选修订版数量')
    parser.add_argument('--selection_temperature', type=float, default=1.6, help='挑选最终修订版的温度')
    parser.add_argument('--save_changes', action='store_true', default=False, help='是否保存详细修改记录')
    parser.add_argument('--print_changes', action='store_true', help='打印修改详情（调试用）')
    parser.add_argument('--force_restart', action='store_true', help='强制从头开始生成，忽略已有进度')
    parser.add_argument('--max_samples',type=int,default=None,help='每次生成样本数量最大量')
    args = parser.parse_args()

    # 防止组合爆炸
    if args.max_ngram > 4:
        print("错误：max_ngram 不能超过 4，否则候选集过大。程序终止。")
        sys.exit(1)
    return args


# --------------------------- 工具函数 ---------------------------
def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


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


def count_output_lines(output_file):
    """统计输出文件已有有效行数（用于断点续传）"""
    if not os.path.exists(output_file):
        return 0
    count = 0
    with smart_open(output_file, 'rt') as f:
        for line in f:
            if line.strip():
                count += 1
    return count


# --------------------------- 编辑核心逻辑 ---------------------------
def get_position_info(prompt_logprobs, prompt_token_ids, ngram_topk, special_token_ids):
    """解析每个位置的概率信息，过滤特殊 token"""
    positions = []
    for i, lp_dict in enumerate(prompt_logprobs):
        orig_id = prompt_token_ids[i]
        # 特殊 token 直接标记为“不可编辑”，概率设为 0，避免进入编辑流程
        if orig_id in special_token_ids or lp_dict is None:
            positions.append({
                "orig": orig_id,
                "prob": 0.0,                # 概率 < up_threshold，不会触发编辑
                "cands": [(orig_id, 0.0)]
            })
            continue

        logp = lp_dict[orig_id].logprob if orig_id in lp_dict else -np.inf
        prob = np.exp(logp)
        cands = []
        for tid, obj in sorted(lp_dict.items(), key=lambda x: x[1].logprob, reverse=True):
            if tid in special_token_ids:      # 过滤特殊 token
                continue
            cands.append((tid, obj.logprob))
        cands = cands[:ngram_topk]
        if orig_id not in [c[0] for c in cands]:
            cands.append((orig_id, logp))
        positions.append({
            "orig": orig_id,
            "prob": prob,
            "cands": cands
        })
    return positions


def hybrid_resample_once(prompt_logprobs, prompt_token_ids, tokenizer, args):
    positions = get_position_info(
        prompt_logprobs, prompt_token_ids, args.ngram_topk,
        special_token_ids=args.special_token_ids
    )
    L = len(positions)
    new_ids = []
    changes = []
    i = 0
    while i < L:
        # 特殊 token 或概率不达阈值 → 原样保留
        if positions[i]["prob"] < args.up_threshold:
            new_ids.append(positions[i]["orig"])
            i += 1
            continue

        # ---- 收集所有包含当前 i 且长度 ∈ [1, max_ngram] 的合法窗口 ----
        windows = []  # (start, length)
        for length in range(1, args.max_ngram + 1):
            for start in range(i - length + 1, i + 1):
                if start < 0 or start + length > L:
                    continue
                if not (start <= i < start + length):
                    continue
                # 窗口内所有位置概率需 ≥ ngram_threshold
                if all(positions[j]["prob"] >= args.ngram_threshold for j in range(start, start + length)):
                    windows.append((start, length))

        # 是否存在长度 ≥2 的窗口（用于决定走单 token 还是多窗口路线）
        has_window = any(length >= 2 for _, length in windows)

        if not has_window:
            # 单 token 替换（原逻辑）
            cands = positions[i]["cands"]
            logps = np.array([lp for _, lp in cands])
            probs = softmax(logps / args.beta)
            chosen = cands[np.random.choice(len(cands), p=probs)]
            sampled_id = chosen[0]
            if sampled_id != positions[i]["orig"]:
                changes.append({
                    "pos": i,
                    "orig_len": 1, "new_len": 1,
                    "orig": tokenizer.decode([positions[i]["orig"]], skip_special_tokens=True),
                    "new": tokenizer.decode([sampled_id], skip_special_tokens=True),
                })
            new_ids.append(sampled_id)
            i += 1
        else:
            # 多窗口竞争：所有合法窗口（含长度1）平等参与
            all_candidates = []  # (start, length, joint_logp, token_ids)
            for start, length in windows:
                cand_lists = [positions[j]["cands"] for j in range(start, start + length)]
                combos = [[]]
                for clist in cand_lists:
                    combos = [c + [(tid, lp)] for c in combos for (tid, lp) in clist]
                for combo in combos:
                    joint_logp = sum(lp for _, lp in combo)
                    token_ids = [tid for tid, _ in combo]
                    all_candidates.append((start, length, joint_logp, token_ids))

            logps = np.array([lp for _, _, lp, _ in all_candidates])
            probs = softmax(logps / args.ngram_temperature)
            sel_idx = np.random.choice(len(all_candidates), p=probs)
            sel_start, sel_length, _, sel_ids = all_candidates[sel_idx]

            # 撤回窗口内已生成的部分（如果有，则替换）
            if sel_start < len(new_ids):
                new_ids = new_ids[:sel_start]

            orig_tokens = [prompt_token_ids[j] for j in range(sel_start, sel_start + sel_length)]
            if sel_ids != orig_tokens:
                changes.append({
                    "pos": sel_start,
                    "orig_len": sel_length, "new_len": len(sel_ids),
                    "orig": tokenizer.decode(orig_tokens, skip_special_tokens=True),
                    "new": tokenizer.decode(sel_ids, skip_special_tokens=True),
                })
            new_ids.extend(sel_ids)
            i = sel_start + sel_length  # 跳过已处理区域
    return new_ids, changes


def generate_candidates(output, tokenizer, args):
    """为一个原始样本生成 num_candidates 个修订版"""
    candidates = []
    for _ in range(args.num_candidates):
        new_ids, changes = hybrid_resample_once(
            output.prompt_logprobs, output.prompt_token_ids, tokenizer, args
        )
        text = tokenizer.decode(new_ids, skip_special_tokens=True)
        candidates.append({"text": text, "changes": changes})
    return candidates


def edit_distance_simple(s1, s2):
    """简单的编辑距离（避免依赖 Levenshtein）"""
    if len(s1) < len(s2):
        s1, s2 = s2, s1
    if len(s2) == 0:
        return len(s1)
    prev = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        curr = [i + 1]
        for j, c2 in enumerate(s2):
            cost = 0 if c1 == c2 else 1
            curr.append(min(prev[j + 1] + 1, curr[j] + 1, prev[j] + cost))
        prev = curr
    return prev[-1]


def select_best_candidate(original_text, candidates, args):
    """根据编辑距离与温度选择最合适的候选"""
    if len(candidates) == 1:
        return candidates[0]
    distances = [edit_distance_simple(original_text, c["text"]) for c in candidates]
    norm_dist = [d / max(1, len(original_text)) for d in distances]
    max_d = max(norm_dist) if norm_dist else 1.0
    scores = [max_d - d for d in norm_dist]   # 距离越小分数越高
    if args.selection_temperature > 0:
        probs = softmax(np.array(scores) / args.selection_temperature)
        chosen_idx = np.random.choice(len(candidates), p=probs)
    else:
        chosen_idx = np.argmax(scores)
    return candidates[chosen_idx]


# --------------------------- 主流程 ---------------------------
def main():
    args = parse_args()

    # 设置随机种子
    random.seed(args.seed)
    np.random.seed(args.seed)

    # 1. 加载源数据
    ids, texts = load_source_with_id(args.source_data)
    if not ids:
        raise ValueError("源数据为空")

    # 2. 准备模型
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
    args.special_token_ids = set(tokenizer.all_special_ids)

    sampling_params = SamplingParams(
        max_tokens=1,
        prompt_logprobs=args.n_candidates,
    )

    # 3. 断点续传处理
    output_exists = os.path.exists(args.output_file)
    existing_lines = count_output_lines(args.output_file)

    if args.force_restart and output_exists:
        os.remove(args.output_file)
        existing_lines = 0
        print("已删除旧输出文件，强制重新开始。")

    start_idx = existing_lines
    total = len(ids)
    if start_idx >= total:
        print(f"所有 {total} 条已生成完毕，退出。")
        return
    if start_idx > 0:
        print(f"断点续传：已有 {start_idx} 条，将从第 {start_idx} 条继续。")



    if args.max_samples is not None and args.max_samples < len(ids):
        ids = ids[:args.max_samples]
        texts = texts[:args.max_samples]
        print(f"限制处理前 {args.max_samples} 条样本")


    # 4. 输出目录与文件准备
    os.makedirs(os.path.dirname(args.output_file) or '.', exist_ok=True)
    mode = 'a' if start_idx > 0 else 'w'

    remaining_ids = ids[start_idx:]
    remaining_texts = texts[start_idx:]
    total_remaining = len(remaining_ids)

    with open(args.output_file, mode, encoding='utf-8') as fout, \
         tqdm(total=total, initial=start_idx, desc="改进方法生成进度", unit="条",
              bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]") as pbar:

        batch_start = 0
        while batch_start < total_remaining:
            batch_end = min(batch_start + args.batch_size, total_remaining)
            batch_ids = remaining_ids[batch_start:batch_end]
            batch_texts = remaining_texts[batch_start:batch_end]

            # 获取模型 logprobs
            outputs = llm.generate(batch_texts, sampling_params, use_tqdm=False)

            # 对每个样本进行编辑
            for orig_id, output, orig_text in zip(batch_ids, outputs, batch_texts):
                # 生成多个候选
                candidates = generate_candidates(output, tokenizer, args)
                best = select_best_candidate(orig_text, candidates, args)

                # 计算编辑信息
                original_ids = output.prompt_token_ids
                new_ids = tokenizer.encode(best["text"], add_special_tokens=False)
                # 简单的编辑比例：基于 token 长度差异（近似）
                total_tokens = len(original_ids)
                # 计算实际修改的 token 数量（这里用编辑距离近似）
                changes = best.get("changes", [])
                edited_count = sum(chg["orig_len"] for chg in changes)  # 被替换/删除的原始 token 数
                edit_ratio = edited_count / total_tokens if total_tokens > 0 else 0.0
                is_edited = edited_count > 0

                record = {
                    "id": orig_id,
                    "text": best["text"],
                    "method": "improved_toedit",
                    "is_edited": is_edited,
                    "edit_ratio": round(edit_ratio, 6)
                }
                fout.write(json.dumps(record, ensure_ascii=False) + '\n')

            fout.flush()
            pbar.update(batch_end - batch_start)
            batch_start = batch_end

    print(f"✅ 新方法半合成数据生成完成！输出: {args.output_file}")


if __name__ == '__main__':
    main()