## 1. Installation

```bash
pip install -e ./transformers-main
cd LLaMA-Factory
pip install -e ".[torch,metrics]"
```

## 2. Quick Start

### Step 1: Preference Data Construction

Align preference data by running the pipeline:

```bash
cd on_policy_data_gen
sh run_pipline.sh   # You can adjust model, path, sampling parameters, etc.

python convert_data_to_dpo.py \
  --input_path datasets/Llama3.2-3B-Instruct/all_outputs_bin.json \
  --output_path ../data/ultrafeedback_Llama3.2_3B.json
```
Note: Both the model and output path can be modified as needed.

### Step 2: DPO Alignment

Run DPO alignment with the processed dataset.
Make sure to configure the model, dataset path, and hyperparameters according to your setup.
```bash
llamafactory-cli train examples/train_full/llama3.2_3B_full_dpo_ds3.yaml
```
### Step 3: Contrastive Decoding

```bash
cd experiments
sh launch_parallel_cd.sh
sh merge_parallel_cd.sh
```
Note: Contrastive decoding is not compatible with vLLM acceleration.
On large datasets, the process can be very slow. To address this, parallel execution is used.

### Step 4: ConG

```bash
cd ../
cd on_policy_data_gen
sh run_llama3_8B_w2s_3B_8B.sh
```
