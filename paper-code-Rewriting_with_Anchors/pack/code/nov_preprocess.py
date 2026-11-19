#!/usr/bin/env python3
"""
批量小说预处理脚本 (合并输出版)
可选择将所有小说合并输出为单个 .jsonl.gz 文件，便于ToEdit统一处理
"""

import json
import gzip
import re
import os
import argparse
from tqdm import tqdm
from pathlib import Path


def clean_text(text):
    """清洗单段文本：移除广告屏蔽词、特殊字符，规范化空白"""
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'[ \t\r\f\v]+', ' ', text)
    text = text.strip()

    ads_keywords = []"""通用广告屏蔽词"""
    for keyword in ads_keywords:
        text = text.replace(keyword, '')

    text = ' '.join(text.split())
    return text


def split_into_chunks(text, max_chunk_length=256):
    """将长文本分割成适合处理的小块"""
    sentence_endings = r'[。！？!?\.\?\!]'
    sentences = re.split(f'({sentence_endings})', text)

    chunks = []
    current_chunk = ""

    for i in range(0, len(sentences) - 1, 2):
        sentence = sentences[i] + (sentences[i + 1] if i + 1 < len(sentences) else '')
        if len(current_chunk) + len(sentence) <= max_chunk_length:
            current_chunk += sentence
        else:
            if current_chunk:
                chunks.append(current_chunk)
            current_chunk = sentence

    if current_chunk:
        chunks.append(current_chunk)

    return chunks


def process_single_novel(input_path, max_chunk_length=256, min_chunk_length=100):
    """处理单个小说文件，返回文本块列表（不直接写入文件）"""
    encodings_to_try = ['utf-8', 'gbk', 'gb2312', 'gb18030', 'utf-8-sig']
    content = None

    for enc in encodings_to_try:
        try:
            with open(input_path, 'r', encoding=enc) as f:
                content = f.read()
            break
        except (UnicodeDecodeError, FileNotFoundError):
            continue

    if content is None:
        print(f"  警告：无法读取文件 {input_path}，跳过")
        return []

    content = clean_text(content)

    # 章节分割
    chapter_pattern = r'(第[零一二三四五六七八九十百千万\d]+章[^\n]*)'
    chapters = re.split(chapter_pattern, content)

    if len(chapters) == 1:
        chapters = [content]
    else:
        chapters = [chapters[i] + chapters[i + 1] for i in range(0, len(chapters) - 1, 2)]

    all_chunks = []
    for chapter in chapters:
        chunks = split_into_chunks(chapter, max_chunk_length)
        chunks = [ch for ch in chunks if len(ch) >= min_chunk_length]
        all_chunks.extend(chunks)

    return all_chunks


def main():
    parser = argparse.ArgumentParser(description="批量预处理网络小说文件夹（支持合并输出）")
    parser.add_argument('--input_dir', '-i', default='./data/raw',
                        help='输入文件夹路径，包含所有小说文本文件')
    parser.add_argument('--output_dir', '-o', default='./data/processed',
                        help='输出文件夹路径，默认 ./processed_novels')
    parser.add_argument('--merge', '-m',default=True, action='store_true',
                        help='启用合并模式：将所有小说合并为单个输出文件')
    parser.add_argument('--single_output', default='all_novels_merged.jsonl.gz',
                        help='合并模式下的输出文件名，默认 all_novels_merged.jsonl.gz')
    parser.add_argument('--max_len', type=int, default=256,
                        help='每个文本块的最大字符长度，默认256')
    parser.add_argument('--min_len', type=int, default=100,
                        help='每个文本块的最小字符长度，默认100')
    parser.add_argument('--ext', default='.txt',
                        help='要处理的小说文件扩展名，默认 .txt')

    args = parser.parse_args()

    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 获取所有小说文件
    input_path = Path(args.input_dir)
    novel_files = list(input_path.glob(f'*{args.ext}'))

    if not novel_files:
        print(f"错误：在 {args.input_dir} 中未找到任何 {args.ext} 文件")
        return

    print(f"找到 {len(novel_files)} 个小说文件")
    if args.merge:
        print("📦 启用合并模式：所有内容将输出到单个文件")
        print("-" * 50)

    # 处理统计
    stats = {
        'total_files': len(novel_files),
        'processed_files': 0,
        'failed_files': 0,
        'total_chunks': 0
    }

    if args.merge:
        # 合并模式：处理所有文件，写入单个输出
        merged_output_path = output_dir / args.single_output
        print(f"合并输出文件：{merged_output_path}")

        with gzip.open(merged_output_path, 'wt', encoding='utf-8') as out_f:
            for novel_file in tqdm(novel_files, desc="处理并合并"):
                try:
                    chunks = process_single_novel(
                        str(novel_file),
                        args.max_len,
                        args.min_len
                    )

                    if chunks:
                        stats['processed_files'] += 1
                        stats['total_chunks'] += len(chunks)

                        # 将当前小说的所有块写入合并文件
                        for chunk in chunks:
                            json_line = json.dumps({"text": chunk}, ensure_ascii=False)
                            out_f.write(json_line + '\n')

                        print(f"  ✓ {novel_file.name} → {len(chunks)} 个文本块（已合并）")
                    else:
                        stats['failed_files'] += 1
                        print(f"  ✗ {novel_file.name} → 处理失败或为空")

                except Exception as e:
                    stats['failed_files'] += 1
                    print(f"  ✗ {novel_file.name} → 错误: {str(e)}")

        print("-" * 50)
        print(f"✅ 合并处理完成！")
        print(f"  总文件数: {stats['total_files']}")
        print(f"  成功处理: {stats['processed_files']}")
        print(f"  处理失败: {stats['failed_files']}")
        print(f"  总文本块数: {stats['total_chunks']}")
        print(f"  合并文件: {merged_output_path.absolute()}")
        print(f"  文件大小: {os.path.getsize(merged_output_path) / 1024 / 1024:.2f} MB")
        print(f"\n📝 使用说明：")
        print(f"  在ToEdit配置中直接使用: --test_file {merged_output_path.absolute()}")

    else:
        # 原模式：每个文件单独输出
        print("启用独立模式：每个小说输出单独的文件")
        print("-" * 50)

        for novel_file in tqdm(novel_files, desc="处理进度"):
            try:
                output_file = output_dir / f"{novel_file.stem}_processed.jsonl.gz"

                chunks = process_single_novel(
                    str(novel_file),
                    args.max_len,
                    args.min_len
                )

                if chunks:
                    # 写入单独的文件
                    with gzip.open(output_file, 'wt', encoding='utf-8') as f:
                        for chunk in chunks:
                            json_line = json.dumps({"text": chunk}, ensure_ascii=False)
                            f.write(json_line + '\n')

                    stats['processed_files'] += 1
                    stats['total_chunks'] += len(chunks)
                    print(f"  ✓ {novel_file.name} → {len(chunks)} 个文本块")
                else:
                    stats['failed_files'] += 1
                    print(f"  ✗ {novel_file.name} → 处理失败或为空")

            except Exception as e:
                stats['failed_files'] += 1
                print(f"  ✗ {novel_file.name} → 错误: {str(e)}")

        # 输出独立模式的摘要
        print("-" * 50)
        print(f"✅ 批量处理完成！")
        print(f"  总文件数: {stats['total_files']}")
        print(f"  成功处理: {stats['processed_files']}")
        print(f"  处理失败: {stats['failed_files']}")
        print(f"  总文本块数: {stats['total_chunks']}")
        print(f"  输出目录: {output_dir.absolute()}")
        print(f"\n📝 使用说明：")
        print(f"  在ToEdit配置中，设置: --test_file {output_dir.absolute()}")


if __name__ == '__main__':
    main()
