import os
import re
import json
import torch
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification
import argparse

# ======== 参数定义 ========
parser = argparse.ArgumentParser(description="Analyze rewards for different alpha outputs")
parser.add_argument('--data_dir', type=str, default="experiments/data/Qwen2.5-3B-Instruct", help="Directory with cd output jsons")
parser.add_argument('--reward_model_path', type=str, default="RLHFlow/ArmoRM-Llama3-8B-v0.1", help="Reward model path")
parser.add_argument('--target_model_path', type=str, default="saves/Qwen2.5-3B-Instruct/dpo", help="Target model path")
parser.add_argument('--ref_model_path', type=str, default="Qwen/Qwen2.5-3B-Instruct", help="Reference model path")
parser.add_argument('--original_json', type=str, default="data/ultrafeedback_qwen2.5_3B.json", help="Original DPO json")
parser.add_argument('--device', type=str, default="cuda", help="Device to use")
parser.add_argument('--batch_size', type=int, default=4, help="Batch size for reward model scoring")
args = parser.parse_args()

# ======== 加载模型 ========
reward_model = AutoModelForSequenceClassification.from_pretrained(
    args.reward_model_path, device_map=args.device, trust_remote_code=True, torch_dtype=torch.bfloat16
)
reward_tokenizer = AutoTokenizer.from_pretrained(args.reward_model_path, use_fast=True)

target_model = AutoModelForCausalLM.from_pretrained(
    args.target_model_path, device_map=args.device, trust_remote_code=True
)
target_tokenizer = AutoTokenizer.from_pretrained(args.target_model_path, trust_remote_code=True)

ref_model = AutoModelForCausalLM.from_pretrained(
    args.ref_model_path, device_map=args.device, trust_remote_code=True
)
ref_tokenizer = AutoTokenizer.from_pretrained(args.ref_model_path, trust_remote_code=True)

# ======== 工具函数 ========
@torch.no_grad()
def compute_log_prob(prompt, response, model, tokenizer):
    input_text = prompt + response
    inputs = tokenizer(input_text, return_tensors="pt").to(args.device)
    input_ids = inputs["input_ids"]
    prompt_len = len(tokenizer(prompt, return_tensors="pt")["input_ids"][0])

    labels = input_ids.clone()
    labels[:, :prompt_len] = -100
    response_len = input_ids.shape[1] - prompt_len

    outputs = model(input_ids=input_ids, labels=labels)
    log_likelihood = -outputs.loss.item() * response_len
    return log_likelihood, response_len

# ======== 读取原始数据 ========
with open(args.original_json, 'r', encoding='utf-8') as f:
    original_data = json.load(f)
print(f"✅ Loaded {len(original_data)} samples from {args.original_json}")

results = []
all_alpha_rewards = {}   # 原始 explicit 分数，用于胜率矩阵
long_rows = []           # {sample_id, alpha, reward_type, reward} (explicit=范围归一化, implicit=长度归一化)

# ======== 遍历文件 ========
for fname in sorted(os.listdir(args.data_dir)):
    match = re.match(r'ultrafeedback_cd_alpha([\d.]+)_n\d+\.json', fname)
    if not match:
        continue

    alpha = float(match.group(1))
    path = os.path.join(args.data_dir, fname)
    print(f"\n🚀 Processing {fname} (alpha={alpha}) ...")

    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    prompts = [d["prompt"] for d in data]
    responses = [d["chosen"] for d in data]

    # === Explicit reward (原始分数) ===
    alpha_scores_raw = []
    for i in tqdm(range(0, len(prompts), args.batch_size), desc=f"Alpha {alpha} - Explicit"):
        batch_prompts = prompts[i:i+args.batch_size]
        batch_responses = responses[i:i+args.batch_size]
        messages = [[{"role": "user", "content": p}, {"role": "assistant", "content": r}]
                    for p, r in zip(batch_prompts, batch_responses)]
        inputs = reward_tokenizer.apply_chat_template(
            messages, return_tensors="pt", padding=True, truncation=True
        ).to(args.device)
        with torch.no_grad():
            output = reward_model(inputs)
            scores = output.score.detach().float().cpu().tolist()
        alpha_scores_raw.extend(scores)

    all_alpha_rewards[alpha] = alpha_scores_raw

    # === Implicit reward (长度归一化) ===
    implicit_norm = []
    for p, r in tqdm(list(zip(prompts, responses)), desc=f"Alpha {alpha} - Implicit"):
        try:
            lp_target, len_target = compute_log_prob(p, r, target_model, target_tokenizer)
            lp_ref, len_ref = compute_log_prob(p, r, ref_model, ref_tokenizer)
            diff = lp_target - lp_ref
            implicit_norm.append(diff / max(1, len_target))
        except Exception as e:
            print(f"[Error Implicit] {e}")
            implicit_norm.append(None)

    # === 写入长表（raw explicit 等待归一化；implicit 已经是长度归一化值） ===
    n = max(len(alpha_scores_raw), len(implicit_norm))
    for j in range(n):
        if j < len(alpha_scores_raw):
            long_rows.append({
                "sample_id": j, "alpha": alpha,
                "reward_type": "explicit_raw", "reward": alpha_scores_raw[j]
            })
        if j < len(implicit_norm) and implicit_norm[j] is not None:
            long_rows.append({
                "sample_id": j, "alpha": alpha,
                "reward_type": "implicit", "reward": implicit_norm[j]
            })

# ======== 构建 DataFrame 并对 explicit 做样本内范围归一化 ========
df_long = pd.DataFrame(long_rows)

# 先拆 explicit_raw 出来
df_exp = df_long[df_long.reward_type=="explicit_raw"].copy()
df_impl = df_long[df_long.reward_type=="implicit"].copy()

def norm_per_sample(g):
    lo, hi = g.reward.min(), g.reward.max()
    g = g.copy()
    g["reward"] = (g.reward - lo) / (hi - lo + 1e-9)
    g["reward_type"] = "explicit"  # 重命名为 explicit
    return g

df_exp_norm = df_exp.groupby("sample_id", group_keys=False).apply(norm_per_sample)
df_long_final = pd.concat([df_exp_norm, df_impl], ignore_index=True)

# ======== 写 summary (基于归一化后的 explicit / implicit) ========
results = []
for alpha, g in df_long_final.groupby("alpha"):
    g_exp = g[g.reward_type=="explicit"].reward
    g_impl = g[g.reward_type=="implicit"].reward
    results.append({
        "alpha": alpha,
        "explicit_reward_mean": g_exp.mean(),
        "explicit_reward_std": g_exp.std(),
        "implicit_reward_mean": g_impl.mean(),
        "implicit_reward_std": g_impl.std(),
    })
df_summary = pd.DataFrame(results).sort_values("alpha").reset_index(drop=True)

summary_path = os.path.join(args.data_dir, "reward_summary.csv")
df_summary.to_csv(summary_path, index=False)
print(f"\n✅ Saved {summary_path}")
print(df_summary)

# ======== 胜率矩阵 (仍然用原始 explicit 分数) ========
alpha_list = sorted(all_alpha_rewards.keys())
winrate_matrix = pd.DataFrame(index=alpha_list, columns=alpha_list, dtype=float)
for a1 in alpha_list:
    for a2 in alpha_list:
        if a1 == a2:
            winrate_matrix.loc[a1,a2]=0.5
            continue
        s1,s2 = all_alpha_rewards[a1], all_alpha_rewards[a2]
        total = min(len(s1), len(s2))
        if total==0: 
            winrate_matrix.loc[a1,a2]=float("nan"); continue
        wins = sum(x>y for x,y in zip(s1[:total], s2[:total]))
        winrate_matrix.loc[a1,a2] = wins/total

winrate_path = os.path.join(args.data_dir, "explicit_winrate_matrix.csv")
winrate_matrix.to_csv(winrate_path)
print(f"\n✅ Saved {winrate_path}")

# ======== 保存长表 (explicit=归一化, implicit=长度归一化) ========
df_long_final.sort_values(["reward_type","sample_id","alpha"], inplace=True)

long_path = os.path.join(args.data_dir, "per_sample_rewards_long.csv")
df_long_final.to_csv(long_path, index=False)
print(f"\n✅ Saved {long_path}")
print(df_long_final.head(10))