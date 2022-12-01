# RESDSQL: Decoupling the Skeleton Parsing and Schema Linking for Text-to-SQL

The PyTorch implementation of "Decoupling the Skeleton Parsing and Schema Linking for Text-to-SQL". The code released in this repository is the version we submitted to Spider's leaderboard.

## Quick start
All our experiments are conducted on a single RTX A6000 (48G) GPU, and you can reproduce our results as follows:
### Step1: Set up environment
- Ubuntu 18.04
- Python 3.8.5

Clone this repository and set up the environment as follows:
```bash
cd RESDSQL
mkdir data
mkdir third_party
pip install -r requirements.txt
python nltk_downloader.py
cd third_party
git clone https://github.com/ElementAI/spider.git
git clone https://github.com/ElementAI/test-suite-sql-eval.git
mv ./test-suite-sql-eval ./test_suite
cd ..
```

### Step2: Download Spider dataset
Then, you should **manually** download spider.zip ([from here](https://yale-lily.github.io/spider)) from Spider's official website and copy some necessary files:
```bash
unzip spider.zip
cp ./spider/tables.json ./data
cp ./spider/train_spider.json ./data
cp ./spider/dev.json ./data
cp -r ./spider/database ./

rm spider.zip
rm -rf ./spider
```

### Step3: Download checkpoints
Our checkpoints are available in the following Google Drive links:
| Model name      | EM | EXEC |
| ----------- | ----------- | ----- |
| [RESDSQL + T5-1.1-lm100k-base](https://drive.google.com/file/d/15p6osU76CcZ-KB6p3R1OSE2DEBsmkMyh/view?usp=sharing) | 71.4 | 75.1 |
| [RESDSQL + T5-1.1-lm100k-large](https://drive.google.com/file/d/16uhRTpWPOYPIXTru5CVGkDI7lvoDbFGx/view?usp=sharing) | 76.6 | 80.3 |
| [RESDSQL + T5-1.1-lm100k-xl](https://drive.google.com/file/d/1ucvEkC_ATZvjY3eSmhYvd5nwiKgdIrop/view?usp=share_link) | 78.1 | 82.3 |
| [Ranker model](https://drive.google.com/file/d/1j54h1fS5pZjl3aF5Ja3qe0J7CS7xdSLk/view?usp=sharing) | - | - |

Note that the ranker model saves parameters of the ``cross-encoder`` described in our paper. After downloading all checkpoints, you should unzip them in the ``./RESDSQL`` path. Now, the structure of the root folder should be:
```
.
├── data
│   ├── dev.json
│   ├── tables.json
│   └── train_spider.json
├── database
│   ├── academic
│   ├── activity_1
│   ├── aircraft
│   └── ...
├── nltk_downloader.py
├── preprocessing.py
├── ranker_model
│   ├── added_tokens.json
│   ├── config.json
│   ├── dense_ranker.pt
│   └── ...
├── ranker.py
├── README.md
├── requirements.txt
├── run_base_model_on_codalab.sh
├── run_large_model_on_codalab.sh
├── run_xl_model_on_codalab.sh
├── text2sql_base
│   └── best_EM_model
├── text2sql_large
│   └── best_EM_model
├── text2sql.py
├── text2sql_xl
│   └── best_EM_model
├── third_party
│   ├── spider
│   └── test_suite
└── utils
    ├── bridge_content_encoder.py
    ├── load_dataset.py
    ├── models.py
    └── ...
```
### Step4: Run inference
To reproduce our results, you can simply run the following three scripts:
```bash
sh run_base_model_on_codalab.sh # for RESDSQL + T5-1.1-lm100k-base
sh run_large_model_on_codalab.sh # for RESDSQL + T5-1.1-lm100k-large
sh run_xl_model_on_codalab.sh # for RESDSQL + T5-1.1-lm100k-xl
```

## Future Work
We are preparing a cleaner and more modular version of code. And we also have some new findings based on current results. We will release it as soon as possible!

## Acknowledgements
We would thanks to Hongjin Su and Tao Yu for their help in evaluating our models on Spider's test set. We would also thanks to [PICARD: Parsing Incrementally for Constrained Auto-Regressive Decoding from Language Models](https://arxiv.org/abs/2109.05093) ([code](https://github.com/ServiceNow/picard))for their interesting work and open-sourced code.