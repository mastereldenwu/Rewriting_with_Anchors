#!/usr/bin/env python3
"""
全量 unique N-gram 提取（分组独立参照池 + 频次过滤）
每个方法的参照池 = 其他三个池子的 N-gram 并集
只保留 count >= 3 的 N-gram
"""

import csv
import re
import jieba
from collections import Counter


# ============ 文件读取 ============
def load_texts_regex(path):
    texts = []
    with open(path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            match = re.search(r'"text":\s*"(.+?)"\s*[,}]', line)
            if match:
                text = match.group(1)
                text = text.replace('\\"', '"').replace('\\n', '\n').replace('\\t', '\t')
                texts.append(text)
            else:
                match2 = re.search(r'"text":\s*"(.+)"\s*[,}]', line)
                if match2:
                    texts.append(match2.group(1))
    return texts


def get_all_ngrams(texts, n=2):
    counter = Counter()
    for text in texts:
        words = [w for w in jieba.cut(text) if len(w) >= 2 and not re.match(r'^[0-9\W]+$', w)]
        for i in range(len(words) - n + 1):
            counter[tuple(words[i:i + n])] += 1
    return counter


# ============ 主流程 ============
print("加载数据...")
real_texts = load_texts_regex('../data/marked.jsonl')
toedit_texts = load_texts_regex('../PythonProject2/synthetic_data/gen/synthetic_toedit-test.jsonl')
pure_texts = load_texts_regex('../PythonProject2/sampled_data/sy_marked.jsonl')
new_even = load_texts_regex('synthetic_even_cleaned.jsonl')
new_odd = load_texts_regex('synthetic_odd_cleaned.jsonl')
new_texts = new_even + new_odd

pools = {
    'Real': real_texts,
    'Pure': pure_texts,
    'ToEdit': toedit_texts,
    'New': new_texts
}

print("构建各池子的完整 N-gram 计数...")
ngram_counters = {name: get_all_ngrams(texts) for name, texts in pools.items()}
ngram_sets = {name: set(counter.keys()) for name, counter in ngram_counters.items()}

# ============ 为每个方法提取 unique，且 count >= 3 ============
MIN_COUNT = 3
all_rows = []

for method_name in pools:
    # 参照池 = 其他三个池子的 N-gram 并集
    ref_set = set()
    for other_name in pools:
        if other_name != method_name:
            ref_set.update(ngram_sets[other_name])

    counter = ngram_counters[method_name]
    method_unique = []
    for ngram, count in counter.items():
        if count >= MIN_COUNT and ngram not in ref_set:
            phrase = ''.join(ngram)
            method_unique.append({
                'phrase': phrase,
                'count': count,
                'source': method_name
            })

    method_unique.sort(key=lambda x: x['count'], reverse=True)
    print(f"{method_name}: {len(method_unique)} 个 unique N-gram (count >= {MIN_COUNT})")

    # 导出每个方法的独立 CSV
    fname = f'unique_ngrams_{method_name}_min{MIN_COUNT}.csv'
    with open(fname, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['phrase', 'count', 'label', 'note'])
        for item in method_unique:
            writer.writerow([item['phrase'], item['count'], '', ''])
    print(f"  已导出: {fname}")

    # 加上 source 标签，用于总表
    for item in method_unique:
        item['label'] = ''
        item['note'] = ''
    all_rows.extend(method_unique)

# 导出统一大表
all_rows.sort(key=lambda x: x['count'], reverse=True)
with open('../PythonProject2/all_unique_ngrams_unified_min3.csv', 'w', encoding='utf-8', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['source', 'phrase', 'count', 'label', 'note'])
    for item in all_rows:
        writer.writerow([item['source'], item['phrase'], item['count'], '', ''])

print(f"\n统一大表已导出: all_unique_ngrams_unified_min3.csv ({len(all_rows)} 条)")
print("完成！现在您可以对四个组分别进行人工标注了。")