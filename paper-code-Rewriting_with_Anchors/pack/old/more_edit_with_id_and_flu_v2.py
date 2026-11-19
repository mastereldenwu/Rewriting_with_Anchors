#!/usr/bin/env python3
"""
改进的半合成数据生成脚本（三阈值流畅度修正 + 查看替换token）
===========================================================
新增：每条输出记录中，若发生编辑，则包含：
  - orig_token : 原始 token 文本
  - new_token  : 最终采用的 token 文本
"""

import json, gzip, os, sys, argparse, random, numpy as np
from tqdm import tqdm
from vllm import LLM, SamplingParams


# ======================== 参数解析（同前） ========================
def parse_args():
    parser = argparse.ArgumentParser(description="改进的半合成数据生成（三阈值流畅度修正）")

    parser.add_argument('-s', '--source_data', default="./sampled_data/marked.jsonl")
    parser.add_argument('-o', '--output_file', default="./synthetic_data/flue429_1.jsonl")
    parser.add_argument('--model_path',
                        default='/home/laowu/.cache/modelscope/hub/models/Qwen/Qwen2.5-0.5B-Instruct')
    parser.add_argument('--max_model_len', type=int, default=2048)
    parser.add_argument('--gpu_memory_utilization', type=float, default=0.9)
    parser.add_argument('--max_num_seqs', type=int, default=128)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--n_candidates', type=int, default=8)
    parser.add_argument('--seed', type=int, default=42)

    # 编辑参数
    parser.add_argument('--up_threshold', type=float, default=0.99)
    parser.add_argument('--beta', type=float, default=1.5)
    parser.add_argument('--topk_single', type=int, default=8)
    parser.add_argument('--max_ngram', type=int, default=3)
    parser.add_argument('--ngram_threshold', type=float, default=0.85)
    parser.add_argument('--ngram_topk', type=int, default=3)
    parser.add_argument('--ngram_temperature', type=float, default=1.8)

    # 三阈值

    parser.add_argument('--fluency_window_size', type=int, default=3,
                        help='局部流畅度评估窗口半径')
    parser.add_argument('--T_drop', type=float, default=0.2,
                        help='触发修正的流畅度下降阈值 (S_orig - S_toedit)')
    parser.add_argument('--T_final', type=float, default=0.5,
                        help='修正后允许的最大下降 (S_orig - S_final)')
    parser.add_argument('--T_keep', type=float, default=0.8,
                        help='保留原编辑的最大下降阈值 (当修正失败时)')
    parser.add_argument('--fluency_beta', type=float, default=0.1,
                        help='邻居微调采样温度 (0=选最优)')
    # 控制
    parser.add_argument('--max_samples', type=int, default=500)
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--no_record_fluency', action='store_true')

    args = parser.parse_args()
    if args.max_ngram > 4:
        print("错误：max_ngram 不能超过 4")
        sys.exit(1)
    return args


# ======================== 工具函数 ========================
def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def smart_open(filepath, mode='rt'):
    if filepath.endswith('.gz'):
        return gzip.open(filepath, mode, encoding='utf-8')
    return open(filepath, mode, encoding='utf-8')

def load_source_with_id(source_file):
    ids, texts = [], []
    with smart_open(source_file, 'rt') as f:
        for line in f:
            obj = json.loads(line.strip())
            if 'id' in obj and 'text' in obj:
                ids.append(obj['id'])
                texts.append(obj['text'])
    return ids, texts

def count_existing(output_file):
    if not os.path.exists(output_file):
        return 0
    with smart_open(output_file, 'rt') as f:
        return sum(1 for _ in f)

def local_fluency_score(llm, tokenizer, token_ids, pos, window):
    start = max(0, pos - window)
    end = min(len(token_ids), pos + window + 1)
    context = tokenizer.decode(token_ids[start:end])
    sp = SamplingParams(max_tokens=1, prompt_logprobs=1)
    try:
        output = llm.generate([context], sp, use_tqdm=False)[0]
        logprobs = [list(lp.values())[0].logprob for lp in output.prompt_logprobs[1:] if lp is not None]
        return np.mean(logprobs) if logprobs else -np.inf
    except:
        return -np.inf

def is_repeating(window_tokens):
    if len(window_tokens) < 4:
        return False
    for n in range(2, 4):
        for i in range(len(window_tokens) - 2*n + 1):
            if window_tokens[i:i+n] == window_tokens[i+n:i+2*n]:
                return True
    return False


# ======================== 原 ToEdit 逻辑 ========================
def get_position_info(prompt_logprobs, prompt_token_ids, topk_single, ngram_topk, special_token_ids):
    positions = []
    for i, lp_dict in enumerate(prompt_logprobs):
        orig_id = prompt_token_ids[i]
        if orig_id in special_token_ids or lp_dict is None:
            positions.append({"orig": orig_id, "prob": 0.0,
                              "cands_single": [(orig_id, 0.0)],
                              "cands_ngram": [(orig_id, 0.0)]})
            continue
        logp = lp_dict[orig_id].logprob if orig_id in lp_dict else -np.inf
        prob = np.exp(logp)
        all_cands = [(tid, obj.logprob) for tid, obj in sorted(
            lp_dict.items(), key=lambda x: x[1].logprob, reverse=True)
            if tid not in special_token_ids]
        cands_single = all_cands[:topk_single]
        if orig_id not in [c[0] for c in cands_single]:
            cands_single.append((orig_id, logp))
        cands_ngram = all_cands[:ngram_topk]
        if orig_id not in [c[0] for c in cands_ngram]:
            cands_ngram.append((orig_id, logp))
        positions.append({"orig": orig_id, "prob": prob,
                          "cands_single": cands_single, "cands_ngram": cands_ngram})
    return positions

def toedit_single_changes(prompt_logprobs, prompt_token_ids, tokenizer, args):
    positions = get_position_info(prompt_logprobs, prompt_token_ids,
                                  args.topk_single, args.ngram_topk, args.special_token_ids)
    changes = []
    for i, pi in enumerate(positions):
        if pi["prob"] < args.up_threshold:
            continue
        cands = pi["cands_single"]
        logps = np.array([lp for _, lp in cands])
        probs = softmax(logps / args.beta)
        chosen = cands[np.random.choice(len(cands), p=probs)]
        if chosen[0] != pi["orig"]:
            changes.append({"pos": i, "orig_len": 1, "new_len": 1,
                            "orig": [pi["orig"]], "new": [chosen[0]]})
    return changes, positions


# ======================== 三阈值决策 ========================
def generate_one_improved(output, llm, tokenizer, args):
    prompt_logprobs = output.prompt_logprobs
    prompt_token_ids = output.prompt_token_ids

    changes, positions = toedit_single_changes(prompt_logprobs, prompt_token_ids, tokenizer, args)
    if len(changes) == 0:
        return prompt_token_ids, False, [], False, "", "", {}

    ch = changes[0]
    pos = ch["pos"]
    orig_ids = ch["orig"]
    toedit_ids = ch["new"]

    S_orig = local_fluency_score(llm, tokenizer, prompt_token_ids, pos, args.fluency_window_size)
    toedit_seq = list(prompt_token_ids)
    toedit_seq[pos] = toedit_ids[0]
    S_toedit = local_fluency_score(llm, tokenizer, toedit_seq, pos, args.fluency_window_size)

    info = {"decision": None, "S_orig": float(S_orig), "S_toedit": float(S_toedit),
            "S_final": None, "fix_triggered": False, "kept_original": False}
    # 用于记录原始和最终 token 文本
    orig_token_str = tokenizer.decode(orig_ids, skip_special_tokens=True)
    toedit_token_str = tokenizer.decode(toedit_ids, skip_special_tokens=True)

    drop = S_orig - S_toedit
    if drop <= args.T_drop:
        # 直接接受 ToEdit
        info["decision"] = "accept_toedit"
        info["S_final"] = float(S_toedit)
        final_seq = toedit_seq
        edited = True
        final_changes = [ch]
        fluence_flag = False
        return final_seq, edited, final_changes, fluence_flag, \
               orig_token_str, toedit_token_str, info

    # 触发修正
    info["fix_triggered"] = True
    best_seq = list(toedit_seq)
    best_score = S_toedit
    neighbor_offsets = [p for p in [pos-1, pos+1] if 0 <= p < len(prompt_token_ids)]

    for nb_pos in neighbor_offsets:
        cands = positions[nb_pos]["cands_single"]
        for cand_id, _ in cands:
            if cand_id == prompt_token_ids[nb_pos]:
                continue
            temp_seq = list(best_seq)
            temp_seq[nb_pos] = cand_id
            window = temp_seq[max(0, pos-3):min(len(temp_seq), pos+3)]
            if is_repeating(window):
                continue
            score = local_fluency_score(llm, tokenizer, temp_seq, pos, args.fluency_window_size)
            if score > best_score:
                best_score = score
                best_seq = temp_seq
                break

    S_final = best_score
    info["S_final"] = float(S_final)

    # 判断是否接受修正
    if S_final > S_toedit and (S_orig - S_final) <= args.T_final:
        info["decision"] = "accept_fix"
        final_seq = best_seq
        final_changes = []
        if final_seq[pos] != prompt_token_ids[pos]:
            final_changes.append({"pos": pos, "orig_len": 1, "new_len": 1,
                                  "orig": [prompt_token_ids[pos]], "new": [final_seq[pos]]})
        for nb_pos in neighbor_offsets:
            if final_seq[nb_pos] != prompt_token_ids[nb_pos]:
                final_changes.append({"pos": nb_pos, "orig_len": 1, "new_len": 1,
                                      "orig": [prompt_token_ids[nb_pos]], "new": [final_seq[nb_pos]]})
        edited = len(final_changes) > 0
        fluence_flag = (final_seq[pos] != prompt_token_ids[pos] and final_seq[pos] != toedit_ids[0]) or len(final_changes) > 1
        final_token_str = tokenizer.decode([final_seq[pos]], skip_special_tokens=True)
        return final_seq, edited, final_changes, fluence_flag, \
               orig_token_str, final_token_str, info

    # 修正失败，考虑保留原编辑
    if S_orig - S_toedit <= args.T_keep:
        info["decision"] = "keep_original_toedit"
        info["S_final"] = float(S_toedit)
        final_seq = toedit_seq
        edited = True
        final_changes = [ch]
        fluence_flag = False
        return final_seq, edited, final_changes, fluence_flag, \
               orig_token_str, toedit_token_str, info
    else:
        info["decision"] = "reject_all"
        info["kept_original"] = True
        final_seq = prompt_token_ids
        edited = False
        fluence_flag = False
        return final_seq, False, [], False, \
               orig_token_str, orig_token_str, info  # 最终 token 同原始


# ======================== 主流程（修改了 record 构建） ========================
def main():
    args = parse_args()
    random.seed(args.seed); np.random.seed(args.seed)

    ids, texts = load_source_with_id(args.source_data)
    if not ids: raise ValueError("无数据")
    if args.max_samples and args.max_samples < len(ids):
        ids, texts = ids[:args.max_samples], texts[:args.max_samples]
        print(f"限制处理前 {args.max_samples} 条样本")

    llm = LLM(model=args.model_path, tensor_parallel_size=1,
              gpu_memory_utilization=args.gpu_memory_utilization,
              max_model_len=args.max_model_len, max_num_seqs=args.max_num_seqs,
              max_logprobs=args.n_candidates)
    tokenizer = llm.get_tokenizer()
    args.special_token_ids = set(tokenizer.all_special_ids)
    sampling_params = SamplingParams(max_tokens=1, prompt_logprobs=args.n_candidates)

    start_idx = count_existing(args.output_file)
    total = len(ids)
    if start_idx >= total: print("已完成"); return
    if start_idx > 0: print(f"从第 {start_idx} 条继续。")
    os.makedirs(os.path.dirname(args.output_file) or '.', exist_ok=True)
    mode = 'a' if start_idx > 0 else 'w'

    remaining_ids, remaining_texts = ids[start_idx:], texts[start_idx:]
    total_edit = total_fluence = 0

    with open(args.output_file, mode, encoding='utf-8') as fout, \
         tqdm(total=total, initial=start_idx, desc="三阈值修正") as pbar:
        for b_start in range(0, len(remaining_ids), args.batch_size):
            b_end = min(b_start+args.batch_size, len(remaining_ids))
            b_ids = remaining_ids[b_start:b_end]; b_texts = remaining_texts[b_start:b_end]
            outputs = llm.generate(b_texts, sampling_params, use_tqdm=False)

            for oid, out in zip(b_ids, outputs):
                # 注意：generate_one_improved 现在返回 7 个值
                final_seq, edited, changes, fluence, orig_tok, new_tok, info = generate_one_improved(out, llm, tokenizer, args)

                if edited: total_edit += 1
                if fluence: total_fluence += 1

                final_text = tokenizer.decode(final_seq, skip_special_tokens=True)
                total_tok = len(out.prompt_token_ids)
                edit_ratio = (sum(c.get("orig_len", 1) for c in changes) / total_tok) if (edited and total_tok>0) else 0.0

                record = {
                    "id": oid, "text": final_text,
                    "method": "improved_three_threshold",
                    "is_edited": edited,
                    "edit_ratio": round(edit_ratio, 6),
                    "fluence_edit": fluence,
                    # 新增：替换前后的token文本
                    "orig_token": orig_tok,
                    "new_token": new_tok
                }
                if not args.no_record_fluency:
                    record["fluency_info"] = info
                    # 可选：计算差值
                    if info.get("S_orig") is not None and info.get("S_toedit") is not None:
                        record["fluency_info"]["drop"] = round(info["S_orig"] - info["S_toedit"], 4)
                    if info.get("S_final") is not None and info.get("S_orig") is not None:
                        record["fluency_info"]["final_drop"] = round(info["S_orig"] - info["S_final"], 4)

                # 控制台打印（可选）
                if args.verbose and edited:
                    print(f"\n[样本 {oid}] 决策={info['decision']}")
                    print(f"  {orig_tok} → {new_tok}  "
                          f"(S_orig={info['S_orig']:.2f}, S_toedit={info['S_toedit']:.2f}, S_final={info['S_final']})")

                fout.write(json.dumps(record, ensure_ascii=False) + '\n')

            fout.flush()
            pbar.update(b_end - b_start)

    processed = total - start_idx
    print(f"\n完成。共 {processed} 条，编辑 {total_edit}，fluence_edit {total_fluence}")
    print(f"输出: {args.output_file}")


if __name__ == '__main__':
    main()