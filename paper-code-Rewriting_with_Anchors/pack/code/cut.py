#!/usr/bin/env python3
"""从 .jsonl.gz 中蓄水池抽样，划分 train/test 并保存。"""

import argparse
import gzip
import json
import os
import random

# 定义命令行参数
parser = argparse.ArgumentParser(description='从 .jsonl.gz 采样并划分训练/测试集')
parser.add_argument('--input_file', default="../data/all_novels_merged.jsonl.gz", help='输入的 .jsonl.gz 文件路径')
parser.add_argument('--output_dir', default='../data', help='输出目录 (默认 ../data)')
parser.add_argument('--num_samples', type=int, default=95000, help='抽取的总样本数 (默认 4000)')
parser.add_argument('--text_field', default='text', help='JSON 中文本字段名 (默认 text)')
parser.add_argument('--train_ratio', type=float, default=0.8, help='训练集比例 (默认 0.8)')
parser.add_argument('--seed', type=int, default=42, help='随机种子 (默认 42)')
parser.add_argument('--compress', action='store_true', help='输出时使用 gzip 压缩')


def reservoir_sample(iterator, k, seed):
    random.seed(seed)
    reservoir = []
    for i, item in enumerate(iterator):
        if i < k:
            reservoir.append(item)
        else:
            j = random.randint(0, i)
            if j < k:
                reservoir[j] = item
    return reservoir


def read_jsonl_gz(file_path, text_field):
    skipped = 0
    with gzip.open(file_path, 'rt', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                skipped += 1
                continue
            if text_field not in obj:
                skipped += 1
                continue
            yield obj
    print(f'跳过 {skipped} 行（缺失字段或解析错误）')


def save_jsonl(data, path, compress=False):
    opener = gzip.open(path, 'wt', encoding='utf-8') if compress else open(path, 'w', encoding='utf-8')
    with opener as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')


def main():
    args = parser.parse_args()

    if not os.path.exists(args.input_file):
        raise FileNotFoundError(f'输入文件不存在: {args.input_file}')
    os.makedirs(args.output_dir, exist_ok=True)

    print(f'正在从 {args.input_file} 采样 {args.num_samples} 条...')
    samples = reservoir_sample(
        read_jsonl_gz(args.input_file, args.text_field),
        args.num_samples,
        args.seed
    )
    random.Random(args.seed).shuffle(samples)

    split = int(len(samples) * args.train_ratio)
    train, test = samples[:split], samples[split:]

    ext = '.jsonl.gz' if args.compress else '.jsonl'
    train_path = os.path.join(args.output_dir, f'train{ext}')
    test_path = os.path.join(args.output_dir, f'test{ext}')

    save_jsonl(train, train_path, args.compress)
    save_jsonl(test, test_path, args.compress)

    print(f'完成！训练集 {len(train)} 条 -> {train_path}')
    print(f'测试集 {len(test)} 条 -> {test_path}')


if __name__ == '__main__':
    main()