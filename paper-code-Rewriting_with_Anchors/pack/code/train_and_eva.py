
#!/usr/bin/env python3
"""
完整实验脚本：训练 + PPL/多样性评估（可单独开启多样性）。
用法示例（快速测试）：
    python run_exp.py \
        --real_file train_with_id.jsonl.gz \
        --pure_file synthetic_pure_with_id.jsonl \
        --toedit_file synthetic_toedit_with_id.jsonl \
        --new_file synthetic_improved_with_id.jsonl \
        --test_file test_with_id.jsonl.gz \
        --output_dir ./exp_results \
        --total_samples 4000 \
        --ratios 0.3 0.7 \
        --epochs 1 \
        --train_batch_size 8 \
        --eval_batch_size 16 \
        --test_subset 2000 \
        --eval_diversity \
        --diversity_samples 500 \
        --diversity_batch_size 16 \
        --max_new_tokens 80 \
        --temperature 0.8
"""

import os, json, gzip, argparse, random
import numpy as np
import torch
from tqdm import tqdm
from scipy import stats
from collections import Counter
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

# ======================== 文件加载 ========================
def smart_open(path, mode='rt'):
    if path.endswith('.gz'):
        return gzip.open(path, mode, encoding='utf-8')
    return open(path, mode, encoding='utf-8')

def load_jsonl_with_id(path):
    data = []
    with smart_open(path, 'rt') as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if not line: continue
            try:
                obj = json.loads(line)
            except:
                continue
            if 'text' not in obj: continue
            if 'id' not in obj:
                obj['id'] = idx
            data.append(obj)
    return data

def load_test_texts(path, subset=None, seed=42):
    texts = []
    with smart_open(path, 'rt') as f:
        for line in f:
            try:
                obj = json.loads(line.strip())
                if 'text' in obj:
                    texts.append(obj['text'])
            except:
                continue
    if subset is not None and subset < len(texts):
        rng = random.Random(seed)
        texts = rng.sample(texts, subset)
        print(f"从测试集中随机抽取了 {subset} 条")
    return texts

# ==================== 训练集构建 ====================
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

# ==================== PPL 评估（批量，安全） ====================
def eval_ppl_batched(model, tokenizer, texts, batch_size=16, max_length=512):
    model.eval()
    all_ppls = []
    for i in tqdm(range(0, len(texts), batch_size), desc="评估 PPL"):
        batch_texts = texts[i:i+batch_size]
        enc = tokenizer(batch_texts, return_tensors="pt", truncation=True,
                        max_length=max_length, padding=True)
        input_ids = enc["input_ids"].to(model.device)
        attention_mask = enc["attention_mask"].to(model.device)
        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask, labels=input_ids)
            loss = outputs.loss
        batch_ppl = torch.exp(loss).item()
        all_ppls.extend([batch_ppl] * len(batch_texts))
        torch.cuda.empty_cache()
    return all_ppls

# 在文件开头不需要导入 vllm，而是使用 transformers 的生成功能
# 删除原有的 eval_diversity 函数，改为以下实现：

# ==================== 多样性指标计算 ====================
def distinct_n(tokens, n):
    ngrams = [tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)]
    if len(ngrams) == 0:
        return 0.0
    return len(set(ngrams)) / len(ngrams)

def self_bleu(tokens, n=4):
    """简化的 Self-BLEU：生成 tokens 中 n-gram 的重复率 (越低多样性越好，但崩塌时重复率升高)"""
    if len(tokens) < n:
        return 0.0
    ngrams = [tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)]
    if len(ngrams) <= 1:
        return 0.0
    counter = Counter(ngrams)
    # 重复的 n-gram 占比
    repeat = sum(1 for v in counter.values() if v > 1) / len(counter) if counter else 0.0
    return repeat

def eval_diversity_with_transformers(model_dir, tokenizer, prompts,
                                    max_new_tokens, temperature, top_p,
                                    batch_size, base_model_name, bnb_config):
    """使用 PeftModel 直接生成，不合并，显存更安全，速度可接受。"""
    # 加载基础模型 + adapter
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(base_model, model_dir)
    model.eval()
    device = next(model.parameters()).device

    all_tokens = []
    for i in tqdm(range(0, len(prompts), batch_size), desc="生成文本"):
        batch = prompts[i:i+batch_size]
        inputs = tokenizer(batch, return_tensors="pt", padding=True,
                           truncation=True, max_length=512, padding_side='left').to(device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )
        # 去除输入部分，只保留新生成的 token
        input_len = inputs["input_ids"].shape[1]
        for out_seq in outputs:
            gen_tokens = out_seq[input_len:]
            all_tokens.append(gen_tokens.cpu().tolist())

        # 每批次清理显存（防止碎片累积）
        torch.cuda.empty_cache()

    del model, base_model
    torch.cuda.empty_cache()
    import gc
    gc.collect()

    # 计算指标（保持不变）
    distinct1 = np.mean([distinct_n(t, 1) for t in all_tokens])
    distinct2 = np.mean([distinct_n(t, 2) for t in all_tokens])
    vocab_size = tokenizer.vocab_size
    rare_threshold = int(vocab_size * 0.8)
    rare_set = set(range(rare_threshold, vocab_size))
    rare_ratios = []
    for t in all_tokens:
        if len(t) == 0:
            rare_ratios.append(0.0)
        else:
            rare_ratios.append(sum(1 for tid in t if tid in rare_set) / len(t))
    avg_rare = np.mean(rare_ratios)
    self_bleu_vals = [self_bleu(t, 4) for t in all_tokens]
    avg_self_bleu = np.mean(self_bleu_vals)
    avg_len = np.mean([len(t) for t in all_tokens])
    return {
        "distinct-1": distinct1,
        "distinct-2": distinct2,
        "rare_token_ratio": avg_rare,
        "self_bleu_4": avg_self_bleu,
        "avg_generated_length": avg_len,
    }
# ======================== 主流程 ========================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--real_file', default="./sampled_data/marked.jsonl", help='原始真实数据文件 (带 id)')
    parser.add_argument('--pure_file', default="./sampled_data/sy_marked.jsonl", help='纯合成数据文件 (带 id)')
    parser.add_argument('--toedit_file', default="./synthetic_data/gen/synthetic_toedit-test.jsonl",help='ToEdit 半合成数据文件 (带 id)')
    parser.add_argument('--new_file', default="./synthetic_data/flue429999.jsonl",help='改进的半合成数据文件 (带 id)')
    parser.add_argument('--test_file', default="/home/laowu/PycharmProjects/PythonProject2/sampled_data/test.jsonl",help='测试集文件 (带 id, 文本)')
    parser.add_argument('--model_name', default='/home/laowu/.cache/modelscope/hub/models/Qwen/Qwen2.5-0.5B-Instruct',help='基座模型')
    parser.add_argument('--output_dir', default='./exp_results', help='结果保存目录')
    parser.add_argument('--total_samples', type=int, default=5000)
    parser.add_argument('--ratios', nargs='+', type=float, default=[0.3, 0.7,0.9])
    parser.add_argument('--epochs', type=int, default=2)
    parser.add_argument('--train_batch_size', type=int, default=8)
    parser.add_argument('--eval_batch_size', type=int, default=16)
    parser.add_argument('--max_length', type=int, default=512)
    parser.add_argument('--seed_base', type=int, default=1000)
    parser.add_argument('--skip_train', action='store_true')
    parser.add_argument('--test_subset', type=int, default=50, help='测试集随机抽取条数')
    # 多样性评估参数
    parser.add_argument('--eval_diversity', action='store_true',default=True, help='是否评估多样性指标')
    parser.add_argument('--diversity_samples', type=int, default=2000, help='用于生成文本的提示数量')
    parser.add_argument('--diversity_batch_size', type=int, default=512)
    parser.add_argument('--max_new_tokens', type=int, default=50, help='生成最大 token 数')
    parser.add_argument('--temperature', type=float, default=0.3, help='生成温度')
    parser.add_argument('--top_p', type=float, default=0.95, help='生成 top-p')
    parser.add_argument('--gpu_mem_diversity', type=float, default=0.85, help='多样性评估时显存利用率')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # ---------- 数据加载 ----------
    print("加载数据池...")
    real_pool = load_jsonl_with_id(args.real_file)
    pure_pool = load_jsonl_with_id(args.pure_file)
    toedit_pool = load_jsonl_with_id(args.toedit_file)
    new_pool = load_jsonl_with_id(args.new_file)
    print(f"[DEBUG] new_pool 大小: {len(new_pool)}")
    print(f"[DEBUG] new_file 路径: {args.new_file}")
    test_texts = load_test_texts(args.test_file, subset=args.test_subset, seed=args.seed_base)
    print(f"测试集文本数: {len(test_texts)}")
    assert len(test_texts) > 0, "测试集为空！"

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'  # decoder 模型必须左填充
    # 训练配置
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    lora_config = LoraConfig(
        r=16, lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05, bias="none",
        task_type="CAUSAL_LM",
    )

    methods = {'Pure': pure_pool, 'ToEdit': toedit_pool, 'New': new_pool}
    groups = ['R100']
    for m in methods:
        for r in args.ratios:
            groups.append(f'{m}_{int(r*100)}')

    # ---------- 训练 ----------
    for group in groups:
        model_dir = os.path.join(args.output_dir, f'model_{group}')
        if os.path.exists(model_dir):
            print(f"模型 {model_dir} 已存在，跳过训练。")
            continue
        if args.skip_train:
            print(f"跳过训练 {group}（--skip_train）。")
            continue
        # 构建训练集...
        if group == 'R100':
            rng = random.Random(args.seed_base)
            train_set = rng.sample(real_pool, args.total_samples)
        else:
            method, ratio_str = group.split('_')
            ratio = int(ratio_str) / 100
            n_syn = int(args.total_samples * ratio)
            n_real = args.total_samples - n_syn
            seed = args.seed_base + abs(hash(group)) % 1000
            syn_pool = methods[method]
            print(f"[DEBUG] group={group}, method={method}, len(syn_pool)={len(syn_pool)}, n_syn={n_syn}")
            if len(syn_pool) < n_syn:
                raise ValueError(f"合成池不足！组：{group}，需要 {n_syn}，实际 {len(syn_pool)}")
            train_set = build_split(real_pool, syn_pool, n_syn, n_real, seed)



            train_set = build_split(real_pool, syn_pool, n_syn, n_real, seed)
        train_texts = [item['text'] for item in train_set]
        print(f"训练组 {group}: 样本数 {len(train_texts)}")

        base_model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )
        base_model = prepare_model_for_kbit_training(base_model)
        model = get_peft_model(base_model, lora_config)
        train_dataset = texts_to_dataset(train_texts, tokenizer, args.max_length)
        trainer = Trainer(
            model=model,
            args=TrainingArguments(
                output_dir=model_dir,
                per_device_train_batch_size=args.train_batch_size,
                gradient_accumulation_steps=1,
                num_train_epochs=args.epochs,
                learning_rate=2e-4,
                fp16=True,
                logging_steps=10,
                save_strategy="no",
                report_to="none",
            ),
            train_dataset=train_dataset,
            data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
        )
        trainer.train()
        model.save_pretrained(model_dir)
        tokenizer.save_pretrained(model_dir)
        del model, trainer, base_model
        torch.cuda.empty_cache()

    # ---------- PPL 评估 ----------
    results_ppl = {}
    for group in groups:
        model_dir = os.path.join(args.output_dir, f'model_{group}')
        if not os.path.exists(model_dir):
            continue
        print(f"\n加载模型 {group} ...")
        base_model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )
        model = PeftModel.from_pretrained(base_model, model_dir)
        ppl_list = eval_ppl_batched(model, tokenizer, test_texts,
                                    batch_size=args.eval_batch_size,
                                    max_length=args.max_length)
        results_ppl[group] = ppl_list
        print(f"  {group} 平均 PPL: {np.mean(ppl_list):.4f}")
        del model, base_model
        torch.cuda.empty_cache()

    # 统计检验
    print("\n=== 统计检验 (vs R100) ===")
    control_ppls = results_ppl.get('R100')
    if control_ppls:
        for group in groups:
            if group == 'R100' or group not in results_ppl:
                continue
            treat = results_ppl[group]
            c = [x for x in control_ppls if not np.isnan(x)]
            t = [x for x in treat if not np.isnan(x)]
            if len(c) < 2 or len(t) < 2:
                print(f"{group}: 样本不足，跳过")
                continue
            t_stat, p = stats.ttest_ind(c, t, equal_var=False)
            mean_diff = np.mean(t) - np.mean(c)
            pooled_std = np.sqrt((np.var(c) + np.var(t)) / 2)
            d = mean_diff / pooled_std if pooled_std != 0 else 0.0
            print(f"{group}: t={t_stat:.3f}, p={p:.5f}, Cohen's d={d:.3f}")

    # ---------- 多样性评估 ----------
    if args.eval_diversity:
        print("\n=== 多样性评估 ===")
        # 准备提示：取测试集前 diversity_samples 条，截断前 30 token 作为提示
        prompts_full = load_test_texts(args.test_file, subset=args.diversity_samples, seed=args.seed_base)
        prompts = []
        for t in prompts_full:
            tok = tokenizer.encode(t, truncation=True, max_length=20)
            # 保留前面 token，留最后一个 token 作为起始？生成时会自动接续
            prompts.append(tokenizer.decode(tok, skip_special_tokens=True))

        diversity_results = {}
        for group in groups:
            model_dir = os.path.join(args.output_dir, f'model_{group}')
            if not os.path.exists(model_dir):
                continue
            print(f"  评估 {group} 多样性...")
            # 加载模型（复用之前训练时的加载方式）
            base_model = AutoModelForCausalLM.from_pretrained(
                args.model_name,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True,
            )
            model = PeftModel.from_pretrained(base_model, model_dir)
            device = next(model.parameters()).device
            metrics = eval_diversity_with_transformers(
                model_dir, tokenizer, prompts,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                batch_size=args.diversity_batch_size,
                base_model_name=args.model_name,
                bnb_config=bnb_config
            )
            diversity_results[group] = metrics
            print(f"    {group}: {metrics}")
            del model, base_model
            torch.cuda.empty_cache()

        with open(os.path.join(args.output_dir, 'diversity_metrics.json'), 'w') as f:
            json.dump(diversity_results, f, indent=2)

    # 保存 PPL 结果
    summary_ppl = {g: [float(x) for x in ppls] for g, ppls in results_ppl.items()}
    with open(os.path.join(args.output_dir, 'ppl_summary.json'), 'w') as f:
        json.dump(summary_ppl, f, indent=2)
    print(f"\n所有结果已保存至 {args.output_dir}")

if __name__ == '__main__':
    main()