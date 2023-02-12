# NatSQL
This repository contains code for the EMNLP 2021 findings paper ["Natural SQL: Making SQL Easier to Infer from Natural Language Specifications"](https://arxiv.org/abs/2109.05153).

If you use NatSQL in your work, please cite it as follows:
``` bibtex
@inproceedings{gan-etal-2021-natural-sql,
    title = "Natural {SQL}: Making {SQL} Easier to Infer from Natural Language Specifications",
    author = "Gan, Yujian  and
      Chen, Xinyun  and
      Xie, Jinxia  and
      Purver, Matthew  and
      Woodward, John R.  and
      Drake, John  and
      Zhang, Qiaofu",
    booktitle = "Findings of the Association for Computational Linguistics: EMNLP 2021",
    month = nov,
    year = "2021",
    address = "Punta Cana, Dominican Republic",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.findings-emnlp.174",
    doi = "10.18653/v1/2021.findings-emnlp.174",
    pages = "2030--2042",
}
```

## Environment Setup

Install Python dependency via `pip install -r requirements.txt`.





## Usage

### Step 1: Download the Spider dataset

Download the datasets: [Spider](https://yale-lily.github.io/spider). Make sure to download the `06/07/2020` version or newer.
Unpack the datasets somewhere outside this project and put `train_spider.json`, `dev.json`,  `tables.json` and `database` folder under `./data/` directory.

Run `check_and_preprocess.sh` to check and preprocess the dataset. It will generate (1) the `train_spider.json` and `dev.json` with NatSQL<sub>G</sub> ; (2) preprocessed `tables.json` and `tables_for_natsql.json` ; under  `./NatSQLv1_6/` directory. 



### Step 2: Convert NatSQL to SQL

Run `natsql2sql.sh [train/dev] [natsql/natsqlg]` to convert the NatSQL to SQL.
You should get the SQL queries in `results.sql`.  The evaluation results are different from the paper since we have updated the NatSQL which improves the performance of NatSQL<sub>G</sub> but decreases the performance of NatSQL.

##### Evaluation results of converting gold NatSQL with values into SQL:
|    | Train <br /> Exact Match |  Train<br /> Execution Match  | Dev<br /> Exact Match  | Dev<br /> Execution Match  |
| ----------- | ------------------------------------- | -------------------------------------- |-------------------------------------- |-------------------------------------- |
| NatSQL<sub>G</sub>    | 96.6%                        | 95.7%                      |  97.3%                        | 96.8%                      | 
| NatSQL | 92.9%                          | 93.8%                      |  92.7%                        | 93.4%                      | 


### Step 3: Convert NatSQL without Values to Executable SQL

To generate executable SQL, you need to find out the possible values in the question in advance to facilitate copying them to SQL.
Here, the preprocess code for finding out values is very complicated, which can be implemented in another simpler way. However, due to the severe coupling between this process and our other works, we cannot provide a relatively straightforward implementation.
For example, this preprocess code brings values to the SQL and slightly improves the exact match accuracy.


Run `natsql2sql_without_values.sh [train/dev] [natsql/natsqlg]` to convert the NatSQL without values to executable SQL. You should get the SQL queries in `results.sql`. 


##### Evaluation results of converting gold NatSQL without values into executable SQL:
|    | Train <br /> Exact Match |  Train<br /> Execution Match  | Dev<br /> Exact Match  | Dev<br /> Execution Match  |
| ----------- | ------------------------------------- | -------------------------------------- |-------------------------------------- |-------------------------------------- |
| NatSQL<sub>G</sub>    | 96.5%                        | 94.8%                      |  97.7%                        | 96.6%                      | 
| NatSQL | 92.9%                          | 92.9%                      |  93.8%                        | 92.8%                      | 




## NatSQL V1.6.1
The NatSQL version introduced in our NatSQL paper is V1.6.
The V1.6.1 version was used in [Spider-SS](https://github.com/ygan/SpiderSS-SpiderCG).
It extends set operators for NatSQL and corrects some annotation errors from the original Spider dataset. Therefore, exact match and execution match accuracy in `./NatSQLv1_6_1` are significantly lower than that in `./NatSQLv1_6`.
This version is not for chasing the Spider leaderboard but is proposed to give a closer NatSQL query to the natural language question.


## About SQL2NatSQL
We have not completed the SQL2NatSQL conversion code at present. We welcome contributions to NatSQL.


## Acknowledgement
The `./data/20k-original.pkl` and `./data/20k.pkl` are extract from [google-10000-english](https://github.com/first20hours/google-10000-english) that is under the LDC license.

The `./data/conceptnet.pkl` is extract from [conceptnet5](https://github.com/commonsense/conceptnet5) that is under the Creative Commons Attribution Share-Alike 4.0 license.


## License
The code and NatSQL except the data in `./data` folder are under the [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/legalcode) license.