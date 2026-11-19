#!/usr/bin/env python3
"""
完整数据级评估：原有指标 + α-Precision / β-Recall / Authenticity
"""

import json, re, gzip
import numpy as np
from collections import Counter
from sklearn.neighbors import NearestNeighbors
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sentence_transformers import SentenceTransformer
import jieba

# ========== 1. 数据加载（正则提取 text 字段） ==========
def load_texts_regex(path):
    texts = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            m = re.search(r'"text":\s*"(.+?)"\s*[,}]', line)
            if m:
                texts.append(m.group(1).replace('\\"', '"'))
            else:
                m2 = re.search(r'"text":\s*"(.+)"\s*[,}]', line)
                if m2:
                    texts.append(m2.group(1))
    return texts

# ========== 2. 特征提取 ==========
embedder = SentenceTransformer('/home/laowu/all-MiniLM-L6-v2')  # 你的本地模型

# 加载数据
real_texts = load_texts_regex('./sampled_data/marked.jsonl')
toedit_texts = load_texts_regex('./synthetic_data/gen/synthetic_toedit-test.jsonl')
pure_texts = load_texts_regex('./sampled_data/sy_marked.jsonl')
new_even = load_texts_regex('synthetic_even_cleaned.jsonl')
new_odd = load_texts_regex('synthetic_odd_cleaned.jsonl')
new_texts = new_even + new_odd

pools = {
    'Real': real_texts,
    'Pure': pure_texts,
    'ToEdit': toedit_texts,
    'New': new_texts,
}

# 对每个池子采样2000条（节省时间），并编码特征
sample_size = 2000
features = {}
for name, texts in pools.items():
    if len(texts) > sample_size:
        sampled = np.random.choice(texts, sample_size, replace=False).tolist()
    else:
        sampled = texts
    features[name] = embedder.encode(sampled, show_progress_bar=True)
    print(f"{name} features shape: {features[name].shape}")

# ========== 3. 原有指标：语义覆盖度（以真实数据为基准） ==========
def avg_nn_dist(ref_vecs, cand_vecs):
    nn = NearestNeighbors(n_neighbors=1, metric='cosine').fit(cand_vecs)
    dists, _ = nn.kneighbors(ref_vecs)
    return float(dists.mean())

def coverage_at(ref_vecs, cand_vecs, threshold=0.85):
    nn = NearestNeighbors(n_neighbors=1, metric='cosine').fit(cand_vecs)
    dists, _ = nn.kneighbors(ref_vecs)
    sims = 1 - dists.flatten()
    return float(np.mean(sims >= threshold))

real_vecs = features['Real']
results = {}
for name in ['Pure', 'ToEdit', 'New']:
    cand_vecs = features[name]
    results[f'DATA_{name}_coverage'] = {
        'avg_nn_dist': avg_nn_dist(real_vecs, cand_vecs),
        'coverage@0.85': coverage_at(real_vecs, cand_vecs),
    }

# ========== 4. 原有指标：分布外检测（以真实数据训练孤立森林） ==========
ood = IsolationForest(contamination=0.05, random_state=42, n_jobs=-1)
ood.fit(real_vecs)
for name in ['Pure', 'ToEdit', 'New']:
    preds = ood.predict(features[name])
    anomaly_rate = float(np.mean(preds == -1))
    results.setdefault(f'DATA_{name}_anomaly', {})['anomaly_rate'] = anomaly_rate

# ========== 5. 稀有 N-gram（已有代码逻辑，略） ==========
# 这里假设你之前已经计算好了 unique_ngram_count，我们手工填入（示例）
# 实际可以用你之前修正过的逻辑重新计算

# ========== 6. 新增指标：α-Precision, β-Recall, Authenticity ==========
def compute_ap_br_auth(real_feats, synth_feats, seed=42):
    # 混合并打标签
    X = np.vstack([real_feats, synth_feats])
    y = np.hstack([np.ones(len(real_feats)), np.zeros(len(synth_feats))])
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=seed, stratify=y
    )
    clf = LogisticRegression(random_state=seed, max_iter=1000)
    clf.fit(X_train, y_train)
    proba = clf.predict_proba(X_test)[:, 1]  # 属于“真实”的概率

    real_mask = (y_test == 1)
    synth_mask = (y_test == 0)

    alpha = np.mean(proba[synth_mask])          # 合成样本被视为“真”的比例
    beta  = np.mean(proba[real_mask])           # 真实样本被视为“真”的比例
    auth  = np.mean(proba[synth_mask] > 0.5)    # 合成样本中高置信度被判真的比例
    return alpha, beta, auth

real_feats = features['Real']
for name in ['Pure', 'ToEdit', 'New']:
    synth_feats = features[name]
    alpha, beta, auth = compute_ap_br_auth(real_feats, synth_feats)
    results[f'DATA_{name}_apbr'] = {
        'alpha_precision': alpha,
        'beta_recall': beta,
        'authenticity': auth,
    }

# ========== 7. 保存结果 ==========
with open('full_evaluation_results.json', 'w', encoding='utf-8') as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print("所有指标已保存至 full_evaluation_results.json")