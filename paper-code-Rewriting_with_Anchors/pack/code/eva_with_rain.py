#!/usr/bin/env python3
"""
完整实验脚本（修正增强版）：
- 训练（QLoRA，修正label mask）
- 评估（PPL逐样本、ECE、MAUVE、语义覆盖度、稀有模式、分布外检测）
- 突出“数据翻倍”对比（等量真实数据消耗下，New组总数据量翻倍）
用法示例：
    python train_and_eva_enhanced.py \
        --real_file ... \
        --pure_file ... \
        --toedit_file ... \
        --new_file_even ... \
        --new_file_odd ... \
        --test_file ... \
        --output_dir ./exp_results_enhanced \
        --epochs 2
"""

import os, json, gzip, argparse, random, time
import torch
import torch.nn.functional as F
import numpy as np
import re, jieba
import random
from tqdm import tqdm
from collections import Counter
from sklearn.neighbors import NearestNeighbors
from sklearn.ensemble import IsolationForest
from sentence_transformers import SentenceTransformer
from mauve import compute_mauve

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel
from datasets import Dataset

# ======================== 工具函数 ========================
def smart_open(filepath, mode='rt'):
    if filepath.endswith('.gz'):
        return gzip.open(filepath, mode, encoding='utf-8')
    return open(filepath, mode, encoding='utf-8')

def load_jsonl_with_id(filepath):
    data = []
    with smart_open(filepath, 'rt') as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if not line: continue
            try:
                obj = json.loads(line)
            except: continue
            if 'text' not in obj: continue
            if 'id' not in obj:
                obj['id'] = idx
            data.append(obj)
    return data

def load_test_texts(filepath, subset=None, seed=42):
    texts = []
    with smart_open(filepath, 'rt') as f:
        for line in f:
            try:
                obj = json.loads(line.strip())
                if 'text' in obj: texts.append(obj['text'])
            except: continue
    if subset and len(texts) > subset:
        rng = random.Random(seed)
        texts = rng.sample(texts, subset)
    return texts

# ======================== 1. 修正后的 DataCollator ========================
class CausalLMCollator(DataCollatorForLanguageModeling):
    def torch_call(self, examples):
        batch = super().torch_call(examples)
        if 'labels' in batch and 'attention_mask' in batch:
            labels = batch['labels']
            labels[batch['attention_mask'] == 0] = -100
            batch['labels'] = labels
        return batch

# ======================== 2. 训练集构建 ========================
def build_split(real_pool, syn_pool, n_syn, n_real, seed):
    rng = random.Random(seed)
    syn_sample = rng.sample(syn_pool, n_syn)
    forbidden_ids = {item['id'] for item in syn_sample}
    allowed_real = [item for item in real_pool if item['id'] not in forbidden_ids]
    if n_real > len(allowed_real):
        raise ValueError(f"真实样本不足: 需要 {n_real}, 可用 {len(allowed_real)}")
    real_sample = rng.sample(allowed_real, n_real)
    train = real_sample + syn_sample
    rng.shuffle(train)
    return train

def texts_to_dataset(texts, tokenizer, max_length=512):
    dataset = Dataset.from_dict({"text": texts})
    def tokenize(examples):
        return tokenizer(examples["text"], truncation=True, max_length=max_length)
    return dataset.map(tokenize, batched=True, remove_columns=["text"])

# ======================== 3. 修正后的 PPL（逐样本） ========================
def eval_ppl_per_sample(model, tokenizer, texts, batch_size=16, max_length=512):
    model.eval()
    all_ppls = []
    for i in tqdm(range(0, len(texts), batch_size), desc="PPL逐样本"):
        batch_texts = texts[i:i+batch_size]
        enc = tokenizer(batch_texts, return_tensors="pt", truncation=True,
                        max_length=max_length, padding=True).to(model.device)
        input_ids = enc["input_ids"]
        attention_mask = enc["attention_mask"]
        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
        for j in range(len(batch_texts)):
            s_logits = logits[j, :-1, :]
            s_labels = input_ids[j, 1:]
            s_mask = attention_mask[j, 1:]
            if s_mask.sum() == 0: continue
            loss = F.cross_entropy(s_logits, s_labels, reduction='none')
            all_ppls.append(torch.exp(loss[s_mask.bool()].mean()).item())
        torch.cuda.empty_cache()
    return all_ppls

# ======================== 4. ECE 校准误差 ========================
def compute_ece(logits, labels, n_bins=15):
    probs = torch.softmax(logits, dim=-1)
    confidences, preds = torch.max(probs, dim=-1)
    accs = preds.eq(labels).cpu().numpy()
    confs = confidences.cpu().numpy()
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        in_bin = (confs > bins[i]) & (confs <= bins[i+1])
        if np.sum(in_bin) > 0:
            ece += (np.sum(in_bin) / len(confs)) * np.abs(np.mean(accs[in_bin]) - np.mean(confs[in_bin]))
    return ece

def eval_ece_batched(model, tokenizer, texts, batch_size=16, max_length=512, n_bins=15):
    model.eval()
    all_eces = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        enc = tokenizer(batch_texts, return_tensors="pt", truncation=True,
                        max_length=max_length, padding=True).to(model.device)
        input_ids, attention_mask = enc["input_ids"], enc["attention_mask"]
        with torch.no_grad():
            logits = model(input_ids, attention_mask=attention_mask).logits
        for j in range(len(batch_texts)):
            s_logits = logits[j, :-1, :]
            s_labels = input_ids[j, 1:]
            s_mask = attention_mask[j, 1:]
            if s_mask.sum() == 0: continue
            all_eces.append(compute_ece(s_logits[s_mask.bool()], s_labels[s_mask.bool()], n_bins))
        torch.cuda.empty_cache()
    return all_eces

# ======================== 5. 多样性指标 & MAUVE ========================
def distinct_n(tokens, n):
    ngrams = [tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)]
    return len(set(ngrams))/len(ngrams) if ngrams else 0.0

def self_bleu(tokens, n=4):
    if len(tokens) < n: return 0.0
    ngrams = [tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)]
    if len(ngrams) <= 1: return 0.0
    counter = Counter(ngrams)
    return sum(1 for v in counter.values() if v > 1)/len(counter)

def eval_diversity_and_mauve(model_dir, tokenizer, prompts, human_ref_texts,
                              max_new_tokens, temperature, top_p, batch_size,
                              bnb_config, model_name):
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name, quantization_config=bnb_config, device_map="auto", trust_remote_code=True)
    model = PeftModel.from_pretrained(base_model, model_dir).eval()
    device = next(model.parameters()).device
    tokenizer.padding_side = 'left'
    generated_texts = []
    all_tokens = []
    for i in tqdm(range(0, len(prompts), batch_size), desc="生成文本"):
        batch = prompts[i:i+batch_size]
        inputs = tokenizer(batch, return_tensors="pt", padding=True,
                           truncation=True, max_length=256).to(device)
        input_len = inputs["input_ids"].shape[1]
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=max_new_tokens,
                                     temperature=temperature, top_p=top_p,
                                     do_sample=True, pad_token_id=tokenizer.eos_token_id)
        for seq in outputs:
            gen_tokens = seq[input_len:]
            all_tokens.append(gen_tokens.cpu().tolist())
            generated_texts.append(tokenizer.decode(gen_tokens, skip_special_tokens=True))
        torch.cuda.empty_cache()

    del model, base_model
    torch.cuda.empty_cache()

    vocab_size = tokenizer.vocab_size
    rare_set = set(range(int(vocab_size*0.8), vocab_size))
    rare_ratios = [np.mean([1 if t in rare_set else 0 for t in tok]) if tok else 0.0 for tok in all_tokens]

    div_metrics = {
        "distinct-1": np.mean([distinct_n(t,1) for t in all_tokens]),
        "distinct-2": np.mean([distinct_n(t,2) for t in all_tokens]),
        "self_bleu_4": np.mean([self_bleu(t,4) for t in all_tokens]),
        "rare_token_ratio": np.mean(rare_ratios),
        "avg_generated_length": np.mean([len(t) for t in all_tokens]),
    }

    # MAUVE
    print("计算 MAUVE ...")
    mauve_score = 0.0
    if len(generated_texts) > 10 and len(human_ref_texts) > 10:
        try:
            out = compute_mauve(p_text=human_ref_texts, q_text=generated_texts,
                                device_id=0, max_text_length=256,
                                mauve_scaling_factor=5, verbose=False)
            mauve_score = out.mauve
        except:
            mauve_score = -1.0
    div_metrics["mauve"] = mauve_score
    tokenizer.padding_side = 'right'
    return div_metrics

# ======================== 6. 数据级评估（语义覆盖、稀有模式、分布外检测） ========================
class DataLevelEvaluator:
    def __init__(self, embedder_model='all-MiniLM-L6-v2'):
        self.embedder = SentenceTransformer('/home/laowu/all-MiniLM-L6-v2')
        self.ood = IsolationForest(contamination=0.05, random_state=42, n_jobs=-1)

    def semantic_coverage(self, real_texts, gen_texts, threshold=0.85):
        real_v = self.embedder.encode(real_texts, show_progress_bar=True)
        gen_v = self.embedder.encode(gen_texts, show_progress_bar=True)
        nn = NearestNeighbors(n_neighbors=1, metric='cosine').fit(gen_v)
        dists, _ = nn.kneighbors(real_v)
        cover = np.mean((1 - dists.flatten()) >= threshold)
        return {"avg_nn_dist": float(dists.mean()), f"coverage@{threshold}": float(cover)}

    def unique_ngrams(self, target_texts, reference_texts_list, n=2, top_k=1000, max_count=5):

        def get_ngrams(texts):
            counter = Counter()
            for text in texts:
                words = [w for w in jieba.cut(text) if len(w)>=2 and not re.match(r'^[0-9\W]+$', w)]
                for i in range(len(words)-n+1):
                    counter[tuple(words[i:i+n])] += 1
            return counter.most_common(top_k)
        target_ng = get_ngrams(target_texts)
        ref_set = set()
        for ref_texts in reference_texts_list:
            ref_set.update([ng for ng, _ in get_ngrams(ref_texts)])
        uniq = [(ng, c) for ng, c in target_ng if ng not in ref_set and c <= max_count]
        return {"unique_ngram_count": len(uniq), "sample": ["".join(ng) for ng, _ in uniq[:20]]}

    def ood_detect(self, real_texts, gen_texts):
        real_v = self.embedder.encode(real_texts, show_progress_bar=True)
        self.ood.fit(real_v)
        gen_v = self.embedder.encode(gen_texts, show_progress_bar=True)
        preds = self.ood.predict(gen_v)
        return {"anomaly_rate": float(np.mean(preds == -1))}




class DualBenchmarkDataEvaluator:
    """
    双重基准数据级评估器：
    - 基准一（污染版）：用生成增强数据的那批真实数据做参照
    - 基准二（无污染版）：用独立测试集做参照
    """
    def __init__(self, embedder_path='/home/laowu/all-MiniLM-L6-v2'):
        self.embedder = SentenceTransformer(embedder_path)
        self.ood_primary = IsolationForest(contamination=0.05, random_state=42, n_jobs=-1)
        self.ood_secondary = IsolationForest(contamination=0.05, random_state=42, n_jobs=-1)

    def encode(self, texts):
        return self.embedder.encode(texts, show_progress_bar=True)

    # ========== 语义覆盖度 ==========
    def semantic_coverage(self, ref_texts, gen_texts, threshold=0.85):
        """以 ref_texts 为参照，评估 gen_texts 对它的语义覆盖度"""
        if not gen_texts:
            return {"avg_nn_dist": float('inf'), f"coverage@{threshold}": 0.0}
        ref_v = self.encode(ref_texts)
        gen_v = self.encode(gen_texts)
        nn = NearestNeighbors(n_neighbors=1, metric='cosine').fit(gen_v)
        dists, _ = nn.kneighbors(ref_v)
        avg_dist = float(dists.mean())
        cover = float(np.mean((1 - dists.flatten()) >= threshold))
        return {"avg_nn_dist": avg_dist, f"coverage@{threshold}": cover}

    # ========== 分布外检测 ==========
    def ood_detect(self, train_texts, test_texts):
        """用 train_texts 训练孤立森林，检测 test_texts 的异常率"""
        train_v = self.encode(train_texts)
        # 每次新 fit 一个森林，避免基准混淆
        ood = IsolationForest(contamination=0.05, random_state=42, n_jobs=-1)
        ood.fit(train_v)
        test_v = self.encode(test_texts)
        preds = ood.predict(test_v)
        anomaly_rate = float(np.mean(preds == -1))
        return anomaly_rate

    # ========== 稀有 N-gram 挖掘 ==========
    def unique_ngrams(self, target_texts, reference_texts_list, n=2, top_k=1000, max_count=5):
        """找出 target 中在 reference 里没有的稀有 N-gram"""
        def get_ngrams(texts):
            counter = Counter()
            for text in texts:
                words = [w for w in jieba.cut(text) if len(w) >= 2 and not re.match(r'^[0-9\W]+$', w)]
                for i in range(len(words) - n + 1):
                    counter[tuple(words[i:i+n])] += 1
            return counter.most_common(top_k)

        target_ng = get_ngrams(target_texts)
        ref_set = set()
        for ref_texts in reference_texts_list:
            ref_set.update([ng for ng, _ in get_ngrams(ref_texts)])
        uniq = [(ng, c) for ng, c in target_ng if ng not in ref_set and c <= max_count]
        return {
            "unique_ngram_count": len(uniq),
            "sample": ["".join(ng) for ng, _ in uniq[:20]]
        }

    # ========== 一站式双重基准评估 ==========
    def evaluate_dual_benchmarks(
        self,
        primary_real_texts,      # 生成增强数据的那批真实数据
        secondary_test_texts,    # 独立测试集
        gen_texts_dict,          # {"ToEdit": [...], "New": [...], "Pure": [...]}
        coverage_threshold=0.85
    ):
        """
        gen_texts_dict: 各组增强数据的文本列表
        返回：两个基准下的完整指标字典
        """
        results = {}

        # 基准一：用 primary_real_texts 做参照
        ref1 = primary_real_texts
        # 训练孤立森林（基准一污染版）
        print("=== 基准一：原始真实数据（污染版）===")
        ood1 = IsolationForest(contamination=0.05, random_state=42, n_jobs=-1)
        ood1.fit(self.encode(ref1))

        for name, gen_texts in gen_texts_dict.items():
            print(f"  评估 {name} ...")
            cov1 = self.semantic_coverage(ref1, gen_texts, coverage_threshold)
            anomaly1 = float(np.mean(ood1.predict(self.encode(gen_texts)) == -1))
            # 稀有 N-gram：参照集合仅为真实数据
            ngram1 = self.unique_ngrams(gen_texts, [ref1])
            results[f"DATA_{name}_primary"] = {
                **cov1,
                "anomaly_rate": anomaly1,
                **ngram1
            }

        # 基准二：用 secondary_test_texts 做参照
        print("=== 基准二：独立测试集（无污染版）===")
        ref2 = secondary_test_texts
        ood2 = IsolationForest(contamination=0.05, random_state=42, n_jobs=-1)
        ood2.fit(self.encode(ref2))

        for name, gen_texts in gen_texts_dict.items():
            print(f"  评估 {name} ...")
            cov2 = self.semantic_coverage(ref2, gen_texts, coverage_threshold)
            anomaly2 = float(np.mean(ood2.predict(self.encode(gen_texts)) == -1))
            # 稀有 N-gram 也可以基于测试集，但更合理的是仍然使用真实数据+测试集作为参照集合
            ngram2 = self.unique_ngrams(gen_texts, [ref1, ref2])
            results[f"DATA_{name}_secondary"] = {
                **cov2,
                "anomaly_rate": anomaly2,
                **ngram2
            }

        return results




# ======================== 主函数 ========================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--real_file', default="../data/sampled_data/marked.jsonl")
    parser.add_argument('--pure_file', default="../data/sampled_data/sy_marked.jsonl")
    parser.add_argument('--toedit_file', default="..data/synthetic_toedit-test.jsonl")
    parser.add_argument('--new_file_even', default='../data/synthetic_even_cleaned.jsonl')
    parser.add_argument('--new_file_odd', default='../data/synthetic_odd_cleaned.jsonl')
    parser.add_argument('--test_file', default="../data/test.jsonl")
    parser.add_argument('--model_name', default='../model/Qwen2.5-0.5B-Instruct')
    parser.add_argument('--output_dir', default='../data/exp_results_enhanced')
    parser.add_argument('--total_samples', type=int, default=5000)
    parser.add_argument('--ratios', nargs='+', type=float, default=[0.3, 0.7, 0.9])
    parser.add_argument('--epochs', type=int, default=2)
    parser.add_argument('--train_batch_size', type=int, default=8)
    parser.add_argument('--eval_batch_size', type=int, default=16)
    parser.add_argument('--max_length', type=int, default=512)
    parser.add_argument('--seed_base', type=int, default=1000)
    parser.add_argument('--skip_train', action='store_true')
    parser.add_argument('--test_subset', type=int, default=2000)
    parser.add_argument('--eval_diversity', action='store_true', default=True)
    parser.add_argument('--diversity_samples', type=int, default=1000)
    parser.add_argument('--diversity_batch_size', type=int, default=64)
    parser.add_argument('--max_new_tokens', type=int, default=200)
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--top_p', type=float, default=0.95)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'right'

    bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
    lora_config = LoraConfig(r=16, lora_alpha=32, target_modules=["q_proj", "v_proj"],
                             lora_dropout=0.05, bias="none", task_type="CAUSAL_LM")

    # 加载数据
    real_pool = load_jsonl_with_id(args.real_file)
    pure_pool = load_jsonl_with_id(args.pure_file)
    toedit_pool = load_jsonl_with_id(args.toedit_file)
    new_pool_even = load_jsonl_with_id(args.new_file_even)
    new_pool_odd = load_jsonl_with_id(args.new_file_odd)
    test_texts = load_test_texts(args.test_file, args.test_subset, args.seed_base)
    human_ref_texts = [item['text'] for item in random.Random(42).sample(real_pool, min(len(real_pool), 2000))]

    methods = {'Pure': pure_pool, 'ToEdit': toedit_pool}
    groups = ['R100']
    for m in methods:
        for r in args.ratios:
            groups.append(f'{m}_{int(r*100)}')
    for r in args.ratios:
        groups.append(f'New_{int(r * 100)}')

    # 训练
    for group in groups:
        model_dir = os.path.join(args.output_dir, f'model_{group}')
        if os.path.exists(model_dir) or args.skip_train: continue
        if group == 'R100':
            train_set = random.Random(args.seed_base).sample(real_pool, args.total_samples)
        elif group.startswith('New'):
            method, ratio_str = group.split('_')
            ratio = int(ratio_str) / 100
            n_real = int(args.total_samples * (1 - ratio))
            n_syn_base = int(args.total_samples * ratio)  # 偶数、奇数各取的数量

            # 因为数据升序且id连续，直接切片即可
            syn_even_sample = new_pool_even[:n_syn_base]
            syn_odd_sample = new_pool_odd[:n_syn_base]
            # 真实数据取 id 在 [n_syn_base, n_syn_base + n_real) 区间内的
            real_sample = real_pool[n_syn_base: n_syn_base + n_real]

            train_set = real_sample + syn_even_sample + syn_odd_sample
            rng = random.Random(args.seed_base)  # 保持打乱的可复现性
            rng.shuffle(train_set)
        else:
            method, ratio_str = group.split('_')
            ratio = int(ratio_str) / 100
            n_real = int(args.total_samples * (1 - ratio))
            n_syn = int(args.total_samples * ratio) * (2 if group.startswith('New') else 1)
            train_set = build_split(real_pool, methods[method], n_syn, n_real, args.seed_base + hash(group) % 1000)
        trainer = Trainer(
            model=get_peft_model(prepare_model_for_kbit_training(
                AutoModelForCausalLM.from_pretrained(args.model_name, quantization_config=bnb_config, device_map="auto", trust_remote_code=True)), lora_config),
            args=TrainingArguments(output_dir=model_dir, per_device_train_batch_size=args.train_batch_size,
                                   num_train_epochs=args.epochs, learning_rate=2e-4, fp16=True,
                                   logging_steps=10, save_strategy="no", report_to="none"),
            train_dataset=texts_to_dataset([item['text'] for item in train_set], tokenizer, args.max_length),
            data_collator=CausalLMCollator(tokenizer, mlm=False)
        )
        trainer.train()
        trainer.model.save_pretrained(model_dir)
        tokenizer.save_pretrained(model_dir)

    # 评估
    full_results = {}
    data_eval = DataLevelEvaluator()
    for group in groups:
        model_dir = os.path.join(args.output_dir, f'model_{group}')
        if not os.path.exists(model_dir): continue
        print(f"\n评估: {group}")
        base_model = AutoModelForCausalLM.from_pretrained(
            args.model_name, quantization_config=bnb_config, device_map="auto", trust_remote_code=True)
        model = PeftModel.from_pretrained(base_model, model_dir)
        res = {"ppl": np.mean(eval_ppl_per_sample(model, tokenizer, test_texts, args.eval_batch_size, args.max_length)),
               "ece": np.mean(eval_ece_batched(model, tokenizer, test_texts, args.eval_batch_size, args.max_length))}
        if args.eval_diversity:
            prompts = [tokenizer.decode(tokenizer.encode(t, truncation=True, max_length=30), skip_special_tokens=True)
                       for t in load_test_texts(args.test_file, args.diversity_samples, args.seed_base)]
            div = eval_diversity_and_mauve(model_dir, tokenizer, prompts, human_ref_texts,
                                           args.max_new_tokens, args.temperature, args.top_p,
                                           args.diversity_batch_size, bnb_config, args.model_name)
            res.update(div)
        full_results[group] = res
        del model, base_model
        torch.cuda.empty_cache()
        time.sleep(1)

    # 数据级评估
    # ---------- 数据级评估（单基准，保留） ----------
    # 构建 New 方法合成数据池
    new_pool_combined = new_pool_even + new_pool_odd
    methods['New'] = new_pool_combined
    real_baseline_texts = [item['text'] for item in random.Random(42).sample(real_pool, 2000)]
    for method_name, pool in methods.items():
        gen_texts = [item['text'] for item in random.Random(42).sample(pool, min(len(pool), 2000))]
        full_results[f"DATA_{method_name}"] = {
            **data_eval.semantic_coverage(real_baseline_texts, gen_texts),
            **data_eval.unique_ngrams(gen_texts, [real_baseline_texts]),
            **data_eval.ood_detect(real_baseline_texts, gen_texts)
        }

    # ---------- 双重基准评估（污染版 + 无污染版） ----------
    # 1. 提取原始真实数据（生成增强数据的那批）
    random.seed(42)
    primary_real_ids = set(range(5000))
    primary_real_texts = [item['text'] for item in real_pool if item['id'] in primary_real_ids][:2000]

    # 2. 独立测试集文本
    secondary_test_texts = test_texts[:2000]

    # 3. 构建增强数据字典
    gen_dict = {
        "Real": primary_real_texts,
        "Pure": [item['text'] for item in pure_pool[:2000]],
        "ToEdit": [item['text'] for item in toedit_pool[:2000]],
        "New": [item['text'] for item in new_pool_combined[:2000]]
    }

    # 4. 执行双重基准评估
    evaluator_dual = DualBenchmarkDataEvaluator(embedder_path='/home/laowu/all-MiniLM-L6-v2')
    dual_results = evaluator_dual.evaluate_dual_benchmarks(
        primary_real_texts=primary_real_texts,
        secondary_test_texts=secondary_test_texts,
        gen_texts_dict=gen_dict,
        coverage_threshold=0.85
    )

    # 5. 合并结果
    full_results.update(dual_results)

    # ---------- 保存最终结果 ----------
    with open(os.path.join(args.output_dir, 'full_results.json'), 'w', encoding='utf-8') as f:
        json.dump(full_results, f, indent=2, ensure_ascii=False)

    print("完成。")

if __name__ == '__main__':
    main()