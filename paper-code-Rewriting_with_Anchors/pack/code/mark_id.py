#!/usr/bin/env python3
"""
给 .jsonl 或 .jsonl.gz 文件的每一行添加行号 id（从0开始）。
支持输入/输出压缩格式的自动判断。
用法示例：
  python add_id.py -i data.jsonl.gz -o data_with_id.jsonl.gz
  python add_id.py -i data.jsonl -o data_with_id.jsonl --no-compress
"""

import argparse
import gzip
import json
import os


def open_file(filepath, mode='rt'):
    """根据扩展名自动打开 .jsonl.gz 或 .jsonl 文件"""
    if filepath.endswith('.gz'):
        return gzip.open(filepath, mode, encoding='utf-8')
    else:
        return open(filepath, mode, encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description='为 JSONL 数据添加行号 id')
    parser.add_argument('-i', '--input', default="./sampled_data/train.jsonl", help='输入文件 (.jsonl 或 .jsonl.gz)')
    parser.add_argument('-o', '--output', default="./sampled_data/marked.jsonl", help='输出文件路径')
    parser.add_argument('--compress', action='store_true', default=None,
                        help='强制输出为 .gz 压缩格式（默认根据输出文件名后缀判断）')
    parser.add_argument('--no-compress', dest='compress', action='store_false',
                        help='强制输出为普通 .jsonl 格式')
    parser.set_defaults(compress=None)  # None 表示自动
    args = parser.parse_args()

    if not os.path.exists(args.input):
        raise FileNotFoundError(f"输入文件不存在: {args.input}")

    # 自动判断输出是否压缩（若未手动指定）
    if args.compress is None:
        args.compress = args.output.endswith('.gz')

    # 统计
    total = 0
    skipped = 0

    with open_file(args.input, 'rt') as fin, \
         open_file(args.output, 'wt' if not args.compress else 'wt') as fout:

        # 如果输出需要压缩，但 open_file 根据后缀不是 .gz 可能返回普通 open
        # 所以这里手动处理压缩输出
        if args.compress and not args.output.endswith('.gz'):
            # 输出文件不包含 .gz，但指定了压缩，则使用 gzip.open
            fout.close()
            fout = gzip.open(args.output, 'wt', encoding='utf-8')
        elif not args.compress and args.output.endswith('.gz'):
            # 输出文件后缀为 .gz，但指定不压缩，则用普通 open
            fout.close()
            fout = open(args.output, 'wt', encoding='utf-8')

        with fout:
            for idx, line in enumerate(fin):
                total += 1
                line = line.strip()
                if not line:
                    skipped += 1
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    skipped += 1
                    print(f"警告: 第 {idx} 行 JSON 解析失败，跳过")
                    continue
                obj['id'] = idx
                fout.write(json.dumps(obj, ensure_ascii=False) + '\n')

    print(f"处理完成: 总行数 {total}, 成功 {total - skipped}, 跳过 {skipped}")
    print(f"输出文件: {args.output}")


if __name__ == '__main__':
    main()