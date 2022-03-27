# MSGI
Official code and data of the Findings of ACL 2022 paper "Sememe Prediction for BabelNet Synsets using Multilingual and Multimodal Information"

## Introduction
In linguistics, a sememe is defined as the minimum semantic unit of languages. Sememe knowledge bases (KBs), which are built by manually annotating words with sememes, have been successfully applied to various NLP tasks. However, existing sememe KBs only cover a few languages, which hinders the wide utilization of sememes. To address this issue, the task of sememe prediction for BabelNet synsets (SPBS) is presented, aiming to build a multilingual sememe KB based on BabelNet, a multilingual encyclopedia dictionary. By automatically predicting sememes for a BabelNet synset, the words in many languages in the synset would obtain sememe annotations simultaneously. However, previous SPBS methods have not taken full advantage of the abundant information in BabelNet. In this paper, we utilize the multilingual synonyms, multilingual glosses and images in BabelNet for SPBS. We design a multimodal information fusion model to encode and combine this information for sememe prediction. Experimental results show the substantial outperformance of our model over previous methods (about 10 MAP and F1 scores).

## Data
Please unzip the `data.zip` in the root directory of the repository. Under the `data` folder is the source data of the MSGI model, including:
1. `babel_data_full.txt`: The training data of the model, containing the BabelNet id, synonyms and glosses in three languages and image urls of the BabelNet synsets. Here is an example:
    ```
    bn:00076380n
    2	telefilm	television_film
    2	电视电影	電視電影
    1	téléfilm
    A movie that is made to be shown on television
    電視電影是指在電視上播放的電影，通常由電視臺或電影公司製作后再賣給電視臺。
    Le téléfilm est un genre ou format de type fiction au sein de la production audiovisuelle, destiné à une diffusion télévisée.
    5	A movie that is made to be shown on television	A movie that is made to be shown on television	A television film is a feature-length motion picture that is produced and originally distributed by or to a television network, in contrast to theatrical films made explicitly for initial showing in movie theaters.	Feature film that is a television program produced for and originally distributed by a television network	A film made for television.
    1	電視電影是指在電視上播放的電影，通常由電視臺或電影公司製作后再賣給電視臺。
    2	Le téléfilm est un genre ou format de type fiction au sein de la production audiovisuelle, destiné à une diffusion télévisée.	Film ou production particulièrement destiné à une diffusion télévisée
    https://upload.wikimedia.org/wikipedia/commons/5/58/Multimedia_icon.png
    3	https://upload.wikimedia.org/wikipedia/commons/5/58/Multimedia_icon.png	https://upload.wikimedia.org/wikipedia/commons/5/59/Info_blue.svg	https://upload.wikimedia.org/wikipedia/commons/7/79/Rename_icon.svg
    ```
2. `synset_sememes.txt`: The Babel Sememe dataset, contains sememe annotations of about 15k BabelNet synsets. Here is an example:
    ```
    bn:00076380n	image|图像 shows|表演物 disseminate|传播 institution|机构 ProperName|专 information|信息
    ```
Under the `dataset` folder is the data split of the data. The data is split into training set, validation set and test set in an 8:1:1 ratio.

## Usage

1. `utils/data_utils.py` : Provides APIs and classes to download images, read source data, preprocess the data, tokenize and get dataloaders.
2. `utils/train_utils.py` : Provides APIs to assist training.
3. `model.py` and `train.py` : Provides the model and training process.

## Citation
