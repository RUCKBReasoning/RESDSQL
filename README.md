<p align="left">
    <br>
    <img src="resdsql.png" width="700"/>
    <br>
<p>

# RESDSQL: Decoupling Schema Linking and Skeleton Parsing for Text-to-SQL
This is the official implementation of the paper "RESDSQL: Decoupling Schema Linking and Skeleton Parsing for Text-to-SQL" (AAAI 2023).

If this repository could help you, please cite the following paper:
```
@inproceedings{li2022resdsql,
  author = {Haoyang Li and Jing Zhang and Cuiping Li and Hong Chen},
  title = "RESDSQL: Decoupling Schema Linking and Skeleton Parsing for Text-to-SQL",
  booktitle = "AAAI",
  year = "2023"
}
```

## Overview
We introduce a new Text-to-SQL parser, **RESDSQL** (**R**anking-enhanced **E**ncoding plus a **S**keleton-aware **D**ecoding framework for Text-to-**SQL**), which attempts to decoulpe the schema linking and the skeleton parsing to reduce the difficuty of Text-to-SQL. More details can be found in our [paper](https://arxiv.org/abs/2302.05965). All experiments are conducted on a single NVIDIA A100 80G GPU.

## Evaluation Results
We evaluate RESDSQL on four datasets: Spider, Spider-DK, Spider-Syn, and Spider-Realistic. Let's look at the following numbers:

**On Spider:**

| Model | Dev EM | Dev EX | Test EM | Test EX | Google Drive/OneDrive | Baidu Netdisk |
|-------|--------|--------|---------|---------|-----------------------|----------------|
| RESDSQL-3B+NatSQL | **80.5%** | **84.1%** | **72.0%** | **79.9%** | [OneDrive link](https://1drv.ms/u/s!Ak05bBUBFYiktcdziiE79xaeKtO6qg?e=e9424n) | [Link](https://pan.baidu.com/s/1ReHso0QgX5aT-hGySUnlrQ) (pwd: 4r98) |
| RESDSQL-3B | 78.0% | 81.8% | - | - | [Google Drive link](https://drive.google.com/file/d/1M-zVeB6TKrvcIzaH8vHBIKeWqPn95i11/view?usp=sharing) | [Link](https://pan.baidu.com/s/1mZxakfes4wRSEwnRW43i5A) (pwd: sc62) |
| RESDSQL-Large+NatSQL | 76.7% | 81.9% | - | - | [Google Drive link](https://drive.google.com/file/d/1ZwFsH24_qKC3xwYdedPi6T_8argguWHe/view?usp=sharing) | [Link](https://pan.baidu.com/s/18H8lgnv9gfXmUo_oO_CdOA) (pwd: 7iyq) |
| RESDSQL-Large | 75.8% | 80.1% | - | - | [Google Drive link](https://drive.google.com/file/d/1-xwtKwfJZSrmJrU-_Xdkx1kPuZao7r7e/view?usp=sharing) | [Link](https://pan.baidu.com/s/1Mwg0OZZ48APEq9jPvQQNtw) (pwd: q58k) |
| RESDSQL-Base+NatSQL | 74.1% | 80.2% | - | - | [Google Drive link](https://drive.google.com/file/d/1QyfSfHHrxfIM5X9gKUYNr_0ZRVvb1suV/view?usp=share_link) | [Link](https://pan.baidu.com/s/1XegaZFvXuZ_jf3P-9YPQCQ) (pwd: pyxf) |
| RESDSQL-Base | 71.7% | 77.9% | - | - | [Google Drive link](https://drive.google.com/file/d/1lqZ81f_fSZtg6BRcRw1-Ol-RJCcKRsmH/view?usp=sharing) | [Link](https://pan.baidu.com/s/1-6H7zStq0WCJHTjDuVspoQ) (pwd: wuek) |

**On Spider-DK, Spider-Syn, and Spider-Realistic:**

| Model | DK EM | DK EX | Syn EM | Syn EX | Realistic EM | Realistic EX |
|-------|-------|-------|--------|--------|--------------|--------------|
| RESDSQL-3B+NatSQL| 53.3% | 66.0% | 69.1% | 76.9% | 77.4% | 81.9% |

## Prerequisites
Create a virtual anaconda environment:
```sh
conda create -n your_env_name python=3.8.5
```
Active it and install the cuda version Pytorch:
```sh
conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=11.3 -c pytorch
```
Install other required modules and tools:
```sh
pip install -r requirements.txt
pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.2.0/en_core_web_sm-2.2.0.tar.gz
python nltk_downloader.py
```
Create several folders:
```sh
mkdir eval_results
mkdir models
mkdir tensorboard_log
mkdir third_party
```
Clone evaluation scripts:
```sh
cd third_party
git clone https://github.com/ElementAI/spider.git
git clone https://github.com/ElementAI/test-suite-sql-eval.git
mv ./test-suite-sql-eval ./test_suite
cd ..
```
Download datasets (including Spider, Spider-DK, Spider-Syn, and Spider-Realistic) from [data](https://drive.google.com/file/d/1b86WpUrLo1IL98Tnogi3U6MTWK39yhMO/view?usp=share_link) and [database](https://drive.google.com/file/d/1s4ItreFlTa8rUdzwVRmUR2Q9AHnxbNjo/view?usp=share_link) and unzip them:
```sh
unzip data.zip
unzip database.zip
```

## Inference
Our results can be easily reproduced through our released checkpionts.
### Step1: Prepare Checkpoints
First, you should download T5 checkpoints from the table shown above. Then, you **must** download our cross-encoder checkpoints because our method is two-stage. For SQL and NatSQL, we train two cross-encoder because the classification labels of them are slightly different. Here are links: 
| Cross-encoder Checkpoints | Google Drive | Baidu Netdisk |
|----------|-----------|--------------|
| Cross-encoder (for SQL) | [Link](https://drive.google.com/file/d/1zHAhECq1uGPR9Rt1EDsTai1LbRx0jYIo/view?usp=share_link") | [Link](https://pan.baidu.com/s/1trSi8OBOcPo5NkZb_o-T4g) (pwd: dr62) |
| Cross-encoder (for NatSQL) | [Link](https://drive.google.com/file/d/1UWNj1ZADfKa1G5I4gBYCJeEQO6piMg4G/view?usp=share_link") | [Link](https://pan.baidu.com/s/15eGPMSx7K8oLV4hkjnCzaw) (pwd: 18w8) |

After downloading and unpacking all checkpoints, the `models` folder should be:
```
├── models
│   ├── text2natsql_schema_item_classifier
│   ├── text2natsql-t5-3b
│   │   ├── checkpoint-61642
│   │   └── checkpoint-78302
│   ├── text2natsql-t5-base
│   │   └── checkpoint-14352
│   ├── text2natsql-t5-large
│   │   └── checkpoint-21216
│   ├── text2sql_schema_item_classifier
│   ├── text2sql-t5-3b
│   │   └── checkpoint-103292
│   ├── text2sql-t5-base
│   │   └── checkpoint-39312
│   └── text2sql-t5-large
│       └── checkpoint-30576
```

### Step2: Run Inference
The inference scripts are located in `scripts/inference`. Concretely, `infer_text2natsql.sh` represents RESDSQL-\*+NatSQL, `infer_text2sql.sh` represents RESDSQL-\*. For example, you can run the inference of RESDSQL-3B+NatSQL on Spider's dev set via:
```sh
sh scripts/inference/infer_text2natsql.sh 3b spider
```
The first argument can be selected from [base, large, 3b] and the second argument can be selected from [spider, spider-realistic, spider-syn, spider-dk].

The predicted SQL queries are recorded in `predicted_sql.txt`.

## Training
If you want to train RESDSQL, we also provide training scripts in `scripts/text2natsql` and `scripts/text2sql`.

**Training and evaluating RESDSQL-\*+NatSQL on Spider**
```sh
# Step1: preprocess dataset
sh scripts/train/text2natsql/preprocess.sh
# Step2: train cross-encoder
sh scripts/train/text2natsql/train_text2natsql_schema_item_classifier.sh
# Step3: prepare text-to-natsql training and development set for T5
sh scripts/train/text2natsql/generate_text2natsql_dataset.sh
# Step4: fine-tune T5-3B (RESDSQL-3B+NatSQL)
sh scripts/train/text2natsql/train_text2natsql_t5_3b.sh
# Step4: (or) fine-tune T5-Large (RESDSQL-Large+NatSQL)
sh scripts/train/text2natsql/train_text2natsql_t5_large.sh
# Step4: (or) fine-tune T5-Base (RESDSQL-Base+NatSQL)
sh scripts/train/text2natsql/train_text2natsql_t5_base.sh
```

**Training and evaluating RESDSQL-\* on Spider**
```sh
# Step1: preprocess dataset
sh scripts/train/text2sql/preprocess.sh
# Step2: train cross-encoder
sh scripts/train/text2sql/train_text2sql_schema_item_classifier.sh
# Step3: prepare text-to-sql training and development set for T5
sh scripts/train/text2sql/generate_text2sql_dataset.sh
# Step4: fine-tune T5-3B (RESDSQL-3B)
sh scripts/train/text2sql/train_text2sql_t5_3b.sh
# Step4: (or) fine-tune T5-Large (RESDSQL-Large)
sh scripts/train/text2sql/train_text2sql_t5_large.sh
# Step4: (or) fine-tune T5-Base (RESDSQL-Base)
sh scripts/train/text2sql/train_text2sql_t5_base.sh
```

During the training process, the cross-encoder always saves the best checkpoint based on the total AUC, but T5 saves all intermediate checkpoints. We evaluate T5's checkpoints at the end of the training process to select the best checkpoint. The evaluation results are saved in `eval_results`.

As described in our paper, we can also use the scripts in `scripts/evaluate_robustness` to evaluate checkpoints of RESDSQL-3B+NatSQL on Spider-DK, Spider-Syn, and Spider-Realistic.

For example, we can evaluate RESDSQL-3B+NatSQL on Spider-DK via:
```sh
# Step1: preprocess dataset
sh scripts/evaluate_robustness/preprocess_spider_dk.sh
# Step2: Run evaluation on Spider-DK
sh scripts/evaluate_robustness/evaluate_on_spider_dk.sh
```

## Acknowledgements
We would thanks to Hongjin Su and Tao Yu for their help in evaluating our method on Spider's test set. We would also thanks to PICARD ([paper](https://arxiv.org/abs/2109.05093), [code](https://github.com/ServiceNow/picard)), NatSQL ([paper](https://arxiv.org/abs/2109.05153), [code](https://github.com/ygan/NatSQL)), Spider ([paper](https://arxiv.org/abs/1809.08887), [dataset](https://yale-lily.github.io/spider)), Spider-DK ([paper](https://arxiv.org/abs/2109.05157), [dataset](https://github.com/ygan/Spider-DK)), Spider-Syn ([paper](https://arxiv.org/abs/2106.01065), [dataset](https://github.com/ygan/Spider-Syn)), and Spider-Realistic ([paper](https://arxiv.org/abs/2010.12773), [dataset](https://doi.org/10.5281/zenodo.5205322)) for their interesting work and open-sourced code and dataset.
