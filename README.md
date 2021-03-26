# Unsupervised-Self-Training

The code contains the pipeline for Unsupervised Self Training for code switched datasets and models.

<!-- Run the main.py file to run the pipeline. The pipeline very closely resembles the pipeline described in the paper. The code is split into the following steps: -->

## Overview

The pipeline is as described in the paper. It can be run using `run_experiments.ipynb`.

1. Pretrains with `pretrain.py` (if needed; only needed for non-english pretraining).
2. Finetunes with `main.py`.
    - Makes zero shot predictions with given model.
    - Selects top N sentences using selection criteria.
    - Runs specified # of fine tuning iterations.
        - Fine tunes model based on selected sentences (FINE TUNE BLOCK)
        - Makes predictions from fine tuned model (PREDICTION BLOCK)
        - Selects and splits data (SELECTION BLOCK)

**Note:** This code is currently configured for **ratio**-based selection, not vanilla. If you wish to switch to vanilla, you must uncomment and comment several lines of code (indicated in comments) in `run_experiments.ipynb` and `main.py`. For descriptions of ratio and vanilla selection, refer to the paper.

## Instructions for Running

First, install required libraries by running this console command:

```
pip install -r requirements.txt
```

Next, you can either run `run_experiments.ipynb` (recommended) or run commands manually in console using the example calls below.

If you choose to run `run_experiments.ipynb`, you can specify a list of experiments to run. The experiment params are stored in tuples with schema:

```
(ratio, num_epochs, pretrain_lang, finetune_lang)
```

## Example calls to scripts
Here are example calls to `pretrain.py` and `main.py`.

**NOTE**: the formatting for the arg `experiment_name` is important and will break the script if not properly set. Simple code for the formatting can be found in `run_experiments.ipynb`.

**Pretraining**:
```
python3 pretrain.py --experiment_name tamil_pretrain_mal_eval_0_03_ratio_3_epochs --num_epochs 3 --pretrain_data ../Dataset/sentiment_tamil_train.pkl --predev_data ../Dataset/sentiment_tamil_dev.pkl --pretest_data ../Dataset/sentiment_tamil_test.pkl
```

**Finetuning**:
```
python3 main.py --experiment_name tamil_pretrain_mal_eval_0_03_ratio_3_epochs --train_data ../Dataset/sentiment_mal_train.pkl --dev_data ../Dataset/sentiment_mal_dev.pkl --test_data ../Dataset/sentiment_mal_test.pkl --dataset_size 3915 --ratio 0.03 --pro_pos 0.516475 --pro_neg 0.140230 --pro_neu 0.343295
```

`dataset_size` is the total number of observations in the finetuning language train dataset. `pro_pos`, `pro_neg`, and `pro_neu` are the proportions of the label counts per sentiment class (i.e. 30% positive, 45% negative, 25% neutral). 

We've precalculated `dataset_size` and the proportions for you in `run_experiments.ipynb`.

`ratio` determines the batch size per finetune iter.

## Datasets
You should have at least one codeswitched language dataset. If you wish to pretrain and finetune on different languages, you must have at least two datasets (i.e. Hinglish pretraing and Spanglish finetune). 

Each dataset file is stored as a `.pkl`, containing a list of tuples with schema:

```
(sample_id, sample, label)
```