import argparse
import os

import numpy as np
from tqdm import tqdm


def is_last_iter(i):
    with open(f"{args.experiment_name}/iteration{i}/logs/datasplit_logs", "r") as f:
        lines = f.readlines()
    return lines[-1].split()[-1] == "True"


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Runs single experiment (similar to Recurive.py)')
    parser.add_argument('--experiment_name', type=str, required=True, help='Enter name of experiment (used to name outputs)')
    parser.add_argument('--train_data', type=str, required=True, help='Enter path to train dataset')
    parser.add_argument('--dev_data', type=str, required=True, help='Enter path to dev dataset')
    parser.add_argument('--test_data', type=str, required=True, help='Enter path to train dataset')
    parser.add_argument('--num_iters', type=int, required=False, default=15, help='Enter number of fine tuning iters to run')
    parser.add_argument('--dataset_size', type=int, required=True, help='Enter the size of the finetune training set')
    parser.add_argument('--ratio', type=float, required=True, help='Enter percentage of data to be split for a class (like 0.02 for 2%)')
    
    # Args for ratio selection strategy
    parser.add_argument('--pro_pos', type=float, required=True, help='Enter the proportion of this class label to entire dataset (as percent) (like 0.02 for 2%)')
    parser.add_argument('--pro_neg', type=float, required=True, help='Enter the proportion of this class label to entire dataset (as percent) (like 0.02 for 2%)')
    parser.add_argument('--pro_neu', type=float, required=True, help='Enter the proportion of this class label to entire dataset (as percent) (like 0.02 for 2%)')
    
    # Args for vanilla selection strategy (to use vanilla, uncomment these lines and comment out the above lines)
    # parser.add_argument('--pos_ratio', type=float, required=True, help='Enter percentage of data to be split for a class (like 0.02 for 2%)')
    # parser.add_argument('--neg_ratio', type=float, required=True, help='Enter percentage of data to be split for a class (like 0.02 for 2%)')
    # parser.add_argument('--neu_ratio', type=float, required=True, help='Enter percentage of data to be split for a class (like 0.02 for 2%)')
    args = parser.parse_args()

    #---------
    # File prep
    #---------

    print(f"Running experiment: {args.experiment_name} for {args.num_iters} iterations.")

    # Create iteration0 directory and logs directory
    os.makedirs(f"{args.experiment_name}/iteration0/logs", exist_ok=True)

    # Copy over data file and rename it
    os.system(f'cp {args.train_data} {args.experiment_name}/iteration0/processed_data_0.pkl')

    #---------------------
    # Selection ratio prep
    #---------------------
    
    # (selection ratio, by dataset dist) Calculate # samples for each label
    batch_size = int(args.dataset_size * args.ratio * 3)
    print(f"batch_size:{batch_size}")

    pos_ratio = int(batch_size * args.pro_pos)
    neg_ratio = int(batch_size * args.pro_neg)
    neu_ratio = int(batch_size * args.pro_neu)

    # (vanilla selection) Calculate # samples for each label. To use, uncomment these lines and comment out the above
    # pos_ratio = int(args.dataset_size * args.pos_ratio)
    # neg_ratio = int(args.dataset_size * args.neg_ratio)
    # neu_ratio = int(args.dataset_size * args.neu_ratio)

    #---------------------
    # Run zeroshot and split
    #---------------------
    print(f"Dataset size: {args.dataset_size}")
    print(f"# pos: {pos_ratio}, # neg: {neg_ratio}, # neu: {neu_ratio}")

    n_epochs = 1 # one epoch of fine tuning

    print(f"Beginning zero-shot prediction and splitting dataset")

    #Make zero shot predictions
    os.system(f'python3 prediction.py --experiment_name {args.experiment_name} --iter 0 --type zeroshot > {args.experiment_name}/iteration0/logs/prediction_zeroshot_log')

    #Split data
    os.system(f'python3 datasplit.py --experiment_name {args.experiment_name} --iter 0 --positive {pos_ratio} --negative {neg_ratio} --neutral {neu_ratio} > {args.experiment_name}/iteration0/logs/datasplit_logs')

    print(f"Beginning fine-tuning")

    #---------------------
    # Finetuning
    #---------------------
    for i in tqdm(range(1, args.num_iters+1), total=args.num_iters):
        #FINE TUNE MODEL
        os.system(f'python3 finetune.py --experiment_name {args.experiment_name} --iter {i} --n_epochs {n_epochs} > {args.experiment_name}/iteration{i}/logs/fine_tune_logs')

        #loaded predictions
        os.system(f'python3 prediction.py --experiment_name {args.experiment_name} --iter {i} --type load > {args.experiment_name}/iteration{i}/logs/prediction_load_log')

        #SPLIT DATA
        os.system(f'python3 datasplit.py --experiment_name {args.experiment_name} --iter {i} --positive {pos_ratio} --negative {neg_ratio} --neutral {neu_ratio} > {args.experiment_name}/iteration{i}/logs/datasplit_logs')

        if is_last_iter(i):
            print(f"Last iteration due to running out of data: breaking after iter {i}")
            break

    #---------------------
    # Eval
    #---------------------
    print(f"Finished training and beginning eval.")

    os.system(f'python3 evaluation_model_perspective.py --experiment_name {args.experiment_name} --test_data {args.test_data} --num_iters {i}')

    os.system(f'python3 evaluation_algorithm_perspective.py --experiment_name {args.experiment_name} --num_iters {i}')

    #---------------------
    # Cleanup
    #---------------------
    # Delete all model folders
    for i in range(1, i+1):
        os.system(f"rm -r {args.experiment_name}/iteration{i}/saved_model")

    # delete pretraining model
    os.system(f"rm -r {args.experiment_name}/pretrain")
