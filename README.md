# Unsupervised Single Document Abstractive Summarization using Semantic Units

This is the source code of our AACL'2022 paper Unsupervised Single Document Abstractive Summarization using Semantic Units (paper link will be added soon).

## Environment

Our code requires the settings below:

|||
|-|-|
|Operation system|`Ubuntu 18.04+`|
|`Python` version|`3.6.9+`|
|CUDA version|`cuda11.2`|
|Packages|`sum_dist/requirements.txt`|


## Installation
1. Download this repo
```bash
git clone git@github.com:IKMLab/UASSU.git
# or
git clone https://github.com/IKMLab/UASSU.git
```
2. Install packages
```bash
cd UASSU
pip install -r requirements.txt
pip install git+https://github.com/huggingface/datasets
# If using CUDA11:
pip install torch==1.7.1+cu110 -f https://download.pytorch.org/whl/torch_stable.html
```
And we need to install pyrouge for evaluation.
```bash
# We packed the steps into a script.
bash pyrouge_setup.sh
```
Successful installation of pyrouge will display the output like:
```
------------------------------------------------
Ran 10 tests in 4.482s

OK
```
Reference of potential problems when installing pyrouge:      
- [Cannot open exception db file for reading](https://github.com/bheinzerling/pyrouge/issues/8)
- [Cannot set data directory because the path /absolute/path/to/ROUGE-1.5.5/directory/data does not exist](https://github.com/bheinzerling/pyrouge/issues/25)
- [Can't locate XML/Parser.pm in @INC](https://github.com/bheinzerling/pyrouge/issues/27)

We also have to setup spaCy.
```bash
pip install -U pip setuptools wheel
pip install -U spacy

# Install models for corresponding languages
# en (for CNN/DM, XSum, Wiki_en, ArXiv)
python -m spacy download en_core_web_sm
# de (for MLSUM_de)
python -m spacy download de_core_news_sm
# es (for MLSUM_es)
python -m spacy download it_core_news_sm
# ru (for MLSUM_ru)
python -m spacy download ru_core_news_sm
```

## Data pre-processing
Pre-processed data (.pkl) are available at [this link](https://drive.google.com/drive/folders/1K0E_TZBHMF7MyN8M_ffl_AnAzypu4HsD), and place the downloaded .pkl file at `sum_dist/data/preprocess/`

Or you can process data with the following scripts:

### CNN/DM
```sh
python -m sum_dist.preprocess.preprocess \
-dataset cnndm \
-read_config sum_dist/exp_configs/config_preliminary_cnndm.json \
-tokenizer bert \
-save_dir ./sum_dist/data/preprocess \
-num_worker 10
```

### XSum
```sh
python -m sum_dist.preprocess.preprocess \
-dataset xsum \
-read_config sum_dist/exp_configs/config_preliminary_xsum.json \
-tokenizer bert \
-save_dir ./sum_dist/data/preprocess \
-num_worker 10
```

### MLSUM_de
```sh
python -m sum_dist.preprocess.preprocess \
-dataset mlsum_de \
-read_config sum_dist/exp_configs/config_preliminary_mlsum_de.json \
-tokenizer bert \
-save_dir ./sum_dist/data/preprocess \
-num_worker 10
```

### MLSUM_es
```sh
python -m sum_dist.preprocess.preprocess \
-dataset mlsum_es \
-read_config sum_dist/exp_configs/config_preliminary_mlsum_es.json \
-tokenizer bert \
-save_dir ./sum_dist/data/preprocess \
-num_worker 10
```

### MLSUM_ru
```sh
python -m sum_dist.preprocess.preprocess \
-dataset mlsum_ru \
-read_config sum_dist/exp_configs/config_preliminary_mlsum_ru.json \
-tokenizer bert \
-save_dir ./sum_dist/data/preprocess \
-num_worker 10
```

### Wiki_en
```sh
python -m sum_dist.preprocess.preprocess \
-dataset wiki_en \
-read_config sum_dist/exp_configs/config_preliminary_wiki_en.json \
-tokenizer bert \
-save_dir ./sum_dist/data/preprocess \
-num_worker 10
```

### ArXiv
```sh
python -m sum_dist.preprocess.preprocess \
-dataset arxiv \
-read_config ./sum_dist/exp_configs/config_preliminary_arxiv.json \
-tokenizer bert \
-save_dir ./sum_dist/data/preprocess \
-num_worker 5
```


## Model training
[Download link for trained checkpoints](https://drive.google.com/drive/folders/1X9lgk9toRVZx4LdHO9UsC8oNpXw3YUV7?usp=sharing)

```bash
bash scripts/train_cnndm_w5.sh
```


## Inference

```
bash scripts/infer_cnndm_w5.sh
```

## Evaluation

```
bash scripts/evaluate_cnndm_w5.sh
```

## Datasets & Required Summary Length

For setting `truncate_len` during evaluation.

|Dataset|Summary Length|
|-|-|
|CNN/DM|50|
|XSum|50|
|MLSUM_de|30|
|MLSUM_es|20|
|MLSUM_ru|15|

