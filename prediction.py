#import os, sys
#from docopt import docopt
#args = docopt(__doc__)
#print("Command line args:\n", args)
#gpu_id = args['--gpu-id']
#print("Using GPU ", gpu_id)
#os.environ["CUDA_VISIBLE_DEVICES"]=gpu_id

import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import pickle
import numpy as np
import argparse
from transformers import pipeline
import numpy as np
import emoji
from collections import Counter
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from torch.utils.data import TensorDataset, random_split
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup
import time
import torch

print('Loaded all libraries')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

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

def read_data(input_file):
    data = load_data(input_file)

    sentences, labels, tweet_ids = [], [], []

    for (tweet_id, sentence, label) in data:
        sentences.append(sentence)
        labels.append(label)
        tweet_ids.append(tweet_id)

    return sentences, labels, tweet_ids


def predict(sentences, labels, tweet_ids, classifier):
    preds, output_labels = [], []
    pred_label_score = []
    mapping_dict = {'LABEL_0': 'negative', 'LABEL_1': 'neutral', 'LABEL_2': 'positive'}

    for i, (sentence, label, tweet_id) in enumerate(zip(sentences, labels, tweet_ids)):
        try:
            pred = classifier(sentence)
        except:
            print('Faced error in predicting sentiment for: {}. Moving on...'.format(sentence))
            continue

        pred_label = pred[0]['label']

        if pred_label in mapping_dict:
            pred_label = mapping_dict[pred_label]

        pred_label = pred_label.lower()
        label = label.lower()
        preds.append(pred_label)
        output_labels.append(label)
        pred_label_score.append((tweet_id, sentence, pred_label, label, pred[0]['score']))

        if i % 1000 == 0:
            print('Finished processing {}/{} sentences'.format(i, len(sentences)))  
        
        
    return pred_label_score



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Zeroshot on dataset')
    parser.add_argument('--iter', type=str, required=True, help='Enter Iteration number')
    parser.add_argument('--type', type=str, required=True, help='Enter if prediction from zero shot or loaded model')
    args = parser.parse_args()
    print('Entered', args.iter)
    
    #Read data
    input_file = 'iteration' + args.iter + '/processed_data_' + args.iter + '.pkl'
    print('Input File:', input_file)
    sentences, labels, tweet_ids = read_data(input_file)
    #sentences, labels = read_data(input_file)


    #Define classifier
    print()
    if args.type == 'zeroshot':
        classifier = pipeline('sentiment-analysis', 'cardiffnlp/twitter-roberta-base-sentiment', device = 0)
        #classifier = pipeline('sentiment-analysis', 'cardiffnlp/twitter-roberta-base-sentiment')
    else:
        saved_model = 'iteration' + args.iter + '/saved_model'
        print('Model Directory:', saved_model)
        tokenizer = AutoTokenizer.from_pretrained(saved_model)
        model = AutoModelForSequenceClassification.from_pretrained(saved_model)
        classifier = pipeline('sentiment-analysis', model=model, tokenizer=tokenizer, device = 0)
    
    pred_label_score = predict(sentences, labels, tweet_ids, classifier)
    print()


    #Save results
    output_file = 'iteration' + args.iter + '/iteration_' + args.iter + '.pkl'
    print('Output File:', output_file)
    save_data(output_file, pred_label_score)
