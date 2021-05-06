# SarcDetectionRus
> Sarcasm and irony detection (Russian language)

The project is dedicated to collecting, analyzing,evaluating data for the 
task of sarcasm and irony detection in Russian. The data sources are citaty.info
and bbf.ru/quotes.

## Packages / Installing
* python 3.8
* pandas, nltk, pymystem3, seaborn - for analyzing and evaluating
* sklearn, tensorflow, keras - for creating and training models 

Install dependencies:
``` bash
pip3 install -r requirements.txt
```
## Data
Quotes:
https://drive.google.com/drive/folders/1D920QOEqyXekE9wDHWydyBpuvpqyCksh

Sarcasm on Reddit:
https://www.kaggle.com/danofer/sarcasm

Place data in data/Quotes and data/Sarcasm_on_Reddit directories.

## Getting started
The package structure:
```
SarcDetectionRus
  |- data
  | |- Embeddings
  | | |- ...
  | |- Models
  | | |- ...
  | |- Quotes
  | | |- ...
  | |- Sarcasm_on_Reddit
  | | |- ...
  |- notebooks
  | |- ...
  |- results
  | |- quotes
  | | |- ...
  | |- reddit
  | | |- ...
  |- sarcsdet
  | |- configs
  | | |- ...
  | |- embeddings
  | | |- ...
  | |- models
  | | |- ...
  | |- utils
  | | |- ...
  | |- __init__.py
  | |- train.py
  |- README.md
  |- requirements.txt
  |- setup.py
  |- train_all_reddit.sh
  |- train_all.sh
  |- train_bilstm_reddit.sh
  |- train_bilstm.sh
```

The directories:
* data - data, embeddings and saved models directory
* notebooks - ipynb directory (with experiments)
* results - directory with configs of models results
* sarcsdet - package directory (with library code)

### Experiments
All experiments (.ipynb) are located in the notebooks directory.

 Step |  | Quotes exteriments | Sarcasm on Reddit experiments | Twitter experiments 
--- | --- | --- | --- | --- 
0 | Translation | --- | sarcasm_on_reddit_translate_data.ipynb | --- 
1 | Preprocessing and analysis | quotes_analysis_and_preprocessing.ipynb | sarcasm_on_reddit_analysis_and_preprocessing.ipynb | twitter_prepare.ipynb 
2 | Extracting linguistic features | quotes_extract_ling_feats.ipynb | sarcasm_on_reddit_extract_ling_feats.ipynb | --- 
3 | Sklearn models | quotes_sklearn_models.ipynb | sarcasm_on_reddit_sklearn_models.ipynb | --- 
4 | BiLSTM | quotes_bilstm.ipynb | sarcasm_on_reddit_bilstm.ipynb | --- 
5 | RuBERT | quotes_rubert.ipynb | sarcasm_on_reddit_rubert.ipynb | --- 
6 |Evaluation | quotes_eval.ipynb | sarcasm_on_reddit_eval.ipynb | twitter_predict.ipynb 
7 | RuBERT with extra features | quotes_rubert_extra_feats.ipynb | sarcasm_on_reddit_rubert_extra_feats.ipynb | --- 

### Results

#### Quotes

Method | F1 | Precision | Recall | PR_AUC | ROC_AUC
--- | --- | --- | --- | --- | --- 
Random | 0.120 | 0.068 | 0.500 | 0.070 | 0.501
Best Sklearn model without extra feats | 0.223 | 0.143 | 0.513 | 0.150 | 0.718
Best Sklearn model with extra feats | 0.235 | 0.149 | 0.547 | 0.159 | 0.731
BiLSTM (Word2Vec) | 0.234 | 0.148 | 0.562 | 0.152 | 0.735
BiLSTM (Glove) | 0.238 | 0.148 | 0.606 | 0.164 | 0.744
RuBERT | 0.258 | 0.159 | 0.672 | 0.191 | 0.779
RuBERT with extra feats | 0.261 | 0.161 | 0.690 | 0.198 | 0.783

#### Sarcasm on Reddit

Method | F1 | Precision | Recall | PR_AUC | ROC_AUC
--- | --- | --- | --- | --- | --- 
Random | 0.510 | 0.519 | 0.500 | 0.519 | 0.499
Best Sklearn model without extra feats | 0.696 | 0.729 | 0.665 | 0.786 | 0.767
Best Sklearn model with extra feats | 0.702 | 0.735 | 0.672 | 0.794 | 0.774
BiLSTM (Word2Vec) | 0.724 | 0.749 | 0.686 | 0.801 | 0.785
BiLSTM (Glove) | 0.733 | 0.751 | 0.691 | 0.821 | 0.788 
RuBERT | 0.742 | 0.779 | 0.710 | 0.843 | 0.825
RuBERT with extra feats | 0.783 | 0.824 | 0.750 | 0.881 | 0.869

## Acknowledgments
We express our gratitude to the administration of citaty.info and bbf.ru for providing the data.

## License
This project is licensed under the terms of the MIT license.