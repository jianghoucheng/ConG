# Contrastive Weak-to-Strong Generalization

Official implementation of **Contrastive Weak-to-Strong Generalization (ConG)**, accepted by **ICML 2026**.

**Repository:** [https://github.com/jianghoucheng/ConG](https://github.com/jianghoucheng/ConG)

---

## Introduction

Large language models can benefit from alignment signals provided by weaker models. However, directly transferring responses from a weak aligned model may also transfer its limitations, biases, and imperfect preferences.

**Contrastive Weak-to-Strong Generalization (ConG)** addresses this issue by leveraging the behavioral difference between a weak model **before** and **after** alignment. Instead of simply imitating the post-alignment weak model, ConG uses **contrastive decoding** to amplify the alignment-relevant improvement direction revealed by the weak model.

The resulting contrastive responses are then used to train a stronger model, enabling more effective weak-to-strong generalization.

---

## Method Overview

ConG contains four main stages:

1. **Preference Data Construction**  
   Generate or process preference data for weak-model alignment.

2. **Weak-Model DPO Alignment**  
   Align the weak model using Direct Preference Optimization.

3. **Contrastive Decoding**  
   Decode responses by contrasting the token distributions of the pre-alignment and post-alignment weak models.

4. **ConG Training**  
   Use contrastive responses to train a stronger model.

The overall pipeline is:

```text
Preference Data
      |
      v
Weak-Model DPO Alignment
      |
      v
Pre-Alignment Weak Model + Post-Alignment Weak Model
      |
      v
Contrastive Decoding
      |
      v
Contrastive Training Data
      |
      v
Strong Model Training
```

---


## 1. Installation

Install the modified Transformers package:

```bash
pip install -e ./transformers-main
```

Then install LLaMA-Factory with the required dependencies:

```bash
cd LLaMA-Factory
pip install -e ".[torch,metrics]"
```

Please make sure that your CUDA, PyTorch, and GPU environment are properly configured before running training or contrastive decoding.

---

## 2. Quick Start

### Step 1: Preference Data Construction

Generate preference data by running the on-policy data generation pipeline:

```bash
cd on_policy_data_gen
sh run_pipline.sh
```

You can modify the model path, output path, sampling parameters, and other generation settings in `run_pipline.sh`.

Then convert the generated data into DPO format:

```bash
python convert_data_to_dpo.py \
  --input_path datasets/Llama3.2-3B-Instruct/all_outputs_bin.json \
  --output_path ../data/ultrafeedback_Llama3.2_3B.json
```

Both the input path and output path can be modified according to your own setup.

---

### Step 2: DPO Alignment

Run DPO alignment with the processed preference dataset:

```bash
llamafactory-cli train examples/train_full/llama3.2_3B_full_dpo_ds3.yaml
```

Before training, please check the YAML configuration file and make sure the following fields are correctly set:

- model path;
- dataset path;
- output directory;
- learning rate;
- batch size;
- number of training epochs;
- DeepSpeed configuration.

This step produces the **post-alignment weak model**, which will be used together with the original weak model in the contrastive decoding stage.

---

### Step 3: Contrastive Decoding

Run contrastive decoding under the `experiments` directory:

```bash
cd experiments
sh launch_parallel_cd.sh
sh merge_parallel_cd.sh
```

The script `launch_parallel_cd.sh` launches parallel contrastive decoding jobs.

The script `merge_parallel_cd.sh` merges the outputs from different parallel shards.

The merge script is located at:

```bash
LLaMA-Factory/experiments/merge_parallel_cd.sh
```

Please check the contrastive decoding scripts and configure:

- pre-alignment weak model path;
- post-alignment weak model path;
- input data path;
- output directory;
- number of parallel shards;
- decoding temperature;
- top-p;
- maximum generation length;
- contrastive decoding coefficient.

Note that contrastive decoding is currently **not compatible with vLLM acceleration**.

For large datasets, contrastive decoding can be slow. Therefore, this repository uses parallel execution to improve efficiency.

---

### Step 4: ConG Training

After obtaining the contrastive decoding results, run weak-to-strong training:

```bash
cd ../
cd on_policy_data_gen
sh run_llama3_8B_w2s_3B_8B.sh
```

This script trains a stronger model using the contrastive responses generated from the weak model pair.

You may modify the script to configure:

- weak model size;
- strong model size;
- training data path;
- output directory;
- SFT or DPO training settings;
- distributed training settings.

---

## Details of Contrastive Decoding

Contrastive decoding compares the token-level distributions of two weak models:

- the **pre-alignment weak model**;
- the **post-alignment weak model**.

The post-alignment model captures the behavior after preference optimization, while the pre-alignment model provides a reference for the original weak-model distribution.

By contrasting these two distributions, ConG highlights responses that are more strongly associated with alignment improvement.

This allows the strong model to learn from the weak model's alignment direction rather than merely imitating the weak model's raw outputs.

---


---

## Citation

If you find this repository useful, please consider citing our paper:

```bibtex
@inproceedings{jiang2026contrastive,
  title     = {Contrastive Weak-to-Strong Generalization},
  author    = {Jiang, Houcheng and Fang, Junfeng and Wu, Jiaxin and Zhang, Tianyu and Gao, Chen and Wang, Xiang and He, Xiangnan and Deng, Yang},
  booktitle = {Proceedings of the 43rd International Conference on Machine Learning},
  year      = {2026}
}
```

---

## Acknowledgements

This repository builds upon the following open-source projects:

- [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory)
- [Transformers](https://github.com/huggingface/transformers)

We sincerely thank the authors and contributors of these projects for their valuable open-source efforts.

---

## Contact

If you have any questions about the code or experiments, please feel free to open an issue in this repository.
