import os

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import argparse
import pickle
import time
from collections import Counter
from pathlib import Path

import emoji
import numpy as np
import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset, random_split)
from transformers import (AdamW, AutoModelForSequenceClassification,
                          AutoTokenizer, get_linear_schedule_with_warmup,
                          pipeline)


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
    
    try:
        for (tweet_id, sentence, label) in data:
            sentences.append(sentence)
            labels.append(label)
            tweet_ids.append(tweet_id)

    except:
        for (tweet_id, sentence, pred_label, label, score) in data:
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
    #validation file location is hard coded.
    parser.add_argument('--experiment_name', type=str, required=True, help='Enter name of experiment (used to name outputs)')
    parser.add_argument('--iter', type=int, required=True, help='Location of saved model')
    parser.add_argument('--output', type=str, required=True, help='Location of Output File')
    parser.add_argument('--input', type=str, required=True, help='Location of Output File')
    args = parser.parse_args()
    print('Entered', args.iter)
    
    #Read data
    input_file = args.input
    #input_file = 'sentiment_mal_train_binary.pkl'
    #input_file = 'processed_data_binary.pkl'
    print('Input File:', input_file)
    sentences, labels, tweet_ids = read_data(input_file)


    #Define classifier
    print()
    if args.iter == 0:
        # if english pretrain, load the base model without pretraining
        if args.experiment_name[:3] == "eng":
            classifier = pipeline('sentiment-analysis', 'cardiffnlp/twitter-roberta-base-sentiment', device = 0)
        else:
            # if other languages, load the model we pretrained
            model_path = list(Path(f"{args.experiment_name}/pretrain").glob(f"*.model"))[0] # get the only folder that has .model in its name (other models were deleted)
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            model = AutoModelForSequenceClassification.from_pretrained(model_path)
            classifier = pipeline('sentiment-analysis', model=model, tokenizer=tokenizer, device = 0)
    else:
        saved_model = f'{args.experiment_name}/iteration{args.iter}/saved_model'
        print('Model Directory:', saved_model)
        tokenizer = AutoTokenizer.from_pretrained(saved_model)
        model = AutoModelForSequenceClassification.from_pretrained(saved_model)
        classifier = pipeline('sentiment-analysis', model=model, tokenizer=tokenizer, device = 0)
    
    pred_label_score = predict(sentences, labels, tweet_ids, classifier)
    print()


    #Save results
    output_file = args.output + f'/iteration_{args.iter}.pkl'
    print('Output File:', output_file)
    save_data(output_file, pred_label_score)
