#!/usr/bin/env python3
"""
生成纯合成数据（前50字提示，保留源数据id）
- 利用剩余显存加速推理：gpu_memory_utilization=0.95，max_num_seqs=256
- 分批提交 vLLM 请求，降低内存峰值
- 支持断点续传：检查输出文件已有记录，跳过已生成部分
- tqdm 显示整体进度与预估剩余时间
- 输出为 .jsonl（可后续手动压缩）
"""

import json
import gzip
import os
import argparse
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


def prepare_prompts_with_id(ids, texts, prefix_length, num_variants):
    """生成提示列表和对应的原始 id 列表"""
    prompts = []
    prompt_ids = []
    for original_id, text in zip(ids, texts):
        prompt = text[:prefix_length] if len(text) > prefix_length else text
        for _ in range(num_variants):
            prompts.append(prompt)
            prompt_ids.append(original_id)
    print(f"准备 {len(prompts)} 个提示 (num_variants={num_variants}, prefix_len={prefix_length})")
    return prompts, prompt_ids


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


def main():
    parser = argparse.ArgumentParser(description="生成纯合成数据（加速版，利用剩余显存）")
    # 基础路径参数
    parser.add_argument('-s', '--source_data', default="./sampled_data/marked.jsonl", help='带 id 的源数据文件')
    parser.add_argument('-o', '--output_file', default="./sampled_data/sy_marked.jsonl", help='输出文件路径 (.jsonl，推荐)')
    parser.add_argument('--model_path', default='/home/laowu/.cache/modelscope/hub/models/Qwen/Qwen2.5-0.5B-Instruct',
                        help='模型路径')
    # 生成控制
    parser.add_argument('--prefix_length', type=int, default=20, help='提示前缀字符数')
    parser.add_argument('--num_variants', type=int, default=1, help='每条源数据的合成变体数')
    parser.add_argument('--max_tokens', type=int, default=512, help='生成最大 token 数')
    parser.add_argument('--temperature', type=float, default=1.0, help='采样温度')
    parser.add_argument('--top_p', type=float, default=0.95, help='top-p 采样')
    # vLLM 显存与加速参数
    parser.add_argument('--gpu_memory_utilization', type=float, default=0.90,
                        help='GPU 显存利用率 (已调高至 0.90)')
    parser.add_argument('--max_num_seqs', type=int, default=128,
                        help='最大并发序列数 (已调高至 256)')
    parser.add_argument('--max_model_len', type=int, default=512, help='模型最大序列长度')
    parser.add_argument('--batch_size', type=int, default=256, help='每批提交的提示数 (已调大至 256)')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    # 输出格式
    parser.add_argument('--compress', action='store_true', help='强制 .gz 输出 (实际输出为 .jsonl，需手动压缩)')
    parser.add_argument('--no-compress', dest='compress', action='store_false', help='输出 .jsonl (默认)')
    parser.set_defaults(compress=False)  # 默认输出非压缩 .jsonl
    args = parser.parse_args()

    # 前后兼容：若用户指定 --compress，输出文件名建议去除 .gz
    if args.compress:
        print("注意：断点续传功能需输出非压缩 .jsonl，脚本将忽略压缩选项，输出为 .jsonl。")
        if args.output_file.endswith('.gz'):
            args.output_file = args.output_file[:-3]  # 移除 .gz

    # 1. 加载源数据
    ids, texts = load_source_with_id(args.source_data)
    if not ids:
        raise ValueError("源数据为空")

    # 2. 准备提示与 id
    prompts, prompt_ids = prepare_prompts_with_id(ids, texts, args.prefix_length, args.num_variants)

    # 3. 加载模型（使用优化后的显存参数）
    print(f"加载模型: {args.model_path}")
    print(f"显存利用率: {args.gpu_memory_utilization}, 最大并发序列: {args.max_num_seqs}")
    llm = LLM(
        model=args.model_path,
        tensor_parallel_size=1,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        max_logprobs=0,
    )

    # 4. 断点续传
    start_idx = count_existing(args.output_file)
    total_prompts = len(prompts)
    if start_idx >= total_prompts:
        print(f"所有 {total_prompts} 条已生成完毕，退出。")
        return
    if start_idx > 0:
        print(f"检测到已有 {start_idx} 条记录，将从第 {start_idx} 条继续生成。")

    # 5. 采样参数
    sampling_params = SamplingParams(
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        n=1,
        skip_special_tokens=True,
        spaces_between_special_tokens=True,
    )

    # 6. 输出文件准备（非压缩追加模式）
    os.makedirs(os.path.dirname(args.output_file) or '.', exist_ok=True)
    mode = 'a' if start_idx > 0 else 'w'

    remaining_prompts = prompts[start_idx:]
    remaining_ids = prompt_ids[start_idx:]
    total_remaining = len(remaining_prompts)

    # 7. 分批生成，带进度条
    with open(args.output_file, mode, encoding='utf-8') as fout, \
         tqdm(total=total_prompts, initial=start_idx, desc="生成进度", unit="条",
              bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]") as pbar:

        batch_start = 0
        while batch_start < total_remaining:
            batch_end = min(batch_start + args.batch_size, total_remaining)
            batch_prompts = remaining_prompts[batch_start:batch_end]
            batch_ids = remaining_ids[batch_start:batch_end]

            # 生成
            outputs = llm.generate(batch_prompts, sampling_params, use_tqdm=False)

            # 保存有效条目
            for prompt_id, output in zip(batch_ids, outputs):
                text = output.outputs[0].text.strip()
                if len(text) < 10:
                    continue
                record = {
                    "id": prompt_id,
                    "text": text,
                    "method": "pure_generation",
                    "generation_params": {
                        "max_tokens": args.max_tokens,
                        "temperature": args.temperature,
                        "top_p": args.top_p
                    }
                }
                fout.write(json.dumps(record, ensure_ascii=False) + '\n')

            fout.flush()
            pbar.update(batch_end - batch_start)
            batch_start = batch_end

    print(f"✅ 生成完成！输出文件: {args.output_file}")
    print("提示：如需压缩，可运行: gzip", args.output_file)


if __name__ == '__main__':
    main()
