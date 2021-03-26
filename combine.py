import argparse
import os
import pickle
import random

import numpy as np


def save_data(filename, data):
    #Storing data with labels
    a_file = open(filename, "wb")
    pickle.dump(data, a_file)
    a_file.close()


def load_data(filename):
    a_file = open(filename, "rb")
    output = pickle.load(a_file)
    a_file.close()
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Datasplit')
    parser.add_argument('--experiment_name', type=str, required=True, help='Enter name of experiment (used to name outputs)')
    parser.add_argument('--iter', type=int, required=True, help='Enter Iteration number')
    args = parser.parse_args()

    if args.iter > 0:
        input_file = f'{args.experiment_name}/iteration' + str(args.iter) + '/fine_tune_' + str(args.iter) + '.pkl'    

        current_data = load_data(input_file)
        if args.iter > 1:
            prev_iter = str(args.iter - 1)
            previous_file = f'{args.experiment_name}/iteration' + prev_iter + '/selected_data_' + prev_iter + '.pkl'
            previous_data = load_data(previous_file)
            current_data += previous_data

        save_data(f'{args.experiment_name}/iteration' + str(args.iter) + '/selected_data_' + str(args.iter) + '.pkl', current_data)
