import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import pickle
import numpy as np
import argparse
import numpy as np
from collections import Counter
import pdb
import time
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from torch.utils.data import TensorDataset, random_split
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup
import datetime

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


def read_data(input_file, label_dict):
    data = load_data(input_file)

    sentences, labels = [], []
    
    for (_, sentence, pred_label, label, score) in data:
        sentences.append(sentence)
        labels.append(label_dict[pred_label])

    return sentences, labels

def read_data_test(input_file, label_dict):
    data = load_data(input_file)

    sentences, labels = [], []
    
    for (id, sentence, label) in data:
        sentences.append(sentence)
        labels.append(label_dict[label])

    return sentences, labels


def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))
    
    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))


def tokenize_text(tokeinzer, sentences, labels):

    data_input_ids, data_attention_masks, output_labels = [], [], []

    #TODO: can this be optimized?
    max_num_tokens = 0
    for sent, label in zip(sentences, labels):
        tokenized_output = tokeinzer(sent)
        sent_input_ids = torch.tensor(tokenized_output['input_ids'])
        sent_attention_mask = torch.tensor(tokenized_output['attention_mask'])
        if len(sent_input_ids) > max_num_tokens:
            max_num_tokens = len(sent_input_ids)

    for sent, label in zip(sentences, labels):
        tokenized_output = tokeinzer(sent, max_length=max_num_tokens, pad_to_max_length = True)
        sent_input_ids = torch.tensor(tokenized_output['input_ids'])
        sent_attention_mask = torch.tensor(tokenized_output['attention_mask'])
        data_input_ids.append(sent_input_ids)
        data_attention_masks.append(sent_attention_mask)
        output_labels.append(label)

    # Convert the lists into tensors.
    data_input_ids = torch.stack(data_input_ids, dim=0)
    data_attention_masks = torch.stack(data_attention_masks, dim=0)
    output_labels = torch.tensor(output_labels)

    return data_input_ids, data_attention_masks, output_labels


def create_loader(tokenizer, sentences, labels, batch_size):

    input_ids, attention_masks, labels = tokenize_text(tokenizer, sentences, labels)

    dataset = TensorDataset(input_ids, attention_masks, labels)

    train_size = len(dataset)

    train_dataset = dataset

    print('{:>5,} training samples'.format(train_size))

    train_dataloader = DataLoader(
            train_dataset,  
            sampler = RandomSampler(train_dataset),
            batch_size = batch_size 
        )

    return train_dataloader


def train_epoch(train_dataloader, model, optimizer, scheduler):

    t0 = time.time()
    model.train()
    total_train_loss = 0

    correct = 0
    total = 0

    for step, batch in enumerate(train_dataloader):

        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_labels = batch[2].to(device)
        b_size = b_labels.size(0)

        model.zero_grad()
        # pdb.set_trace()
        outputs = model(b_input_ids, 
                             token_type_ids=None, 
                             attention_mask=b_input_mask, 
                             labels=b_labels)
        
        loss = outputs.loss
        logits = outputs.logits
        total_train_loss += loss.item()
        loss.backward()
        #torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        smax = torch.softmax(logits, dim = 1)
        indices = torch.argmax(smax, dim = 1)
        correct += torch.sum(indices==b_labels).item()
        total += b_size


    avg_train_loss = total_train_loss / len(train_dataloader)  
    training_time = format_time(time.time() - t0)
    print("")
    print("  Average training loss: {0:.2f}".format(avg_train_loss))
    print("  Training epcoh took: {:}".format(training_time))
    val_accuracy = correct / total
    print(" Training Accuracy: {0:.4f}".format(val_accuracy))
    print('Time:', time.time() - t0)
    

def eval_epoch(validation_dataloader, model):
    t0 = time.time()
    model.eval()

    correct = 0
    total = 0

    for i, batch in enumerate(validation_dataloader):
        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_labels = batch[2].to(device)
        b_size = b_labels.size(0)

        with torch.no_grad():
            outputs = model(b_input_ids, 
                                   token_type_ids=None, 
                                   attention_mask=b_input_mask,
                                   labels=b_labels)
            
        loss = outputs.loss
        logits = outputs.logits
        smax = torch.softmax(logits, dim = 1)
        indices = torch.argmax(smax, dim = 1)

        correct += torch.sum(indices==b_labels).item()
        total += b_size

    val_accuracy = correct / total
    print(" Validation Accuracy: {0:.4f}".format(val_accuracy))
    print('Time:', time.time() - t0)
    
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Fine Tuning')
    parser.add_argument('--iter', type=str, required=True, help='Enter Iteration number')
    parser.add_argument('--n_epochs', type=int, required=True, help='Enter number of epochs')
    args = parser.parse_args()
    print('Entered', args.iter)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.iter == '1':
        tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")
        model = AutoModelForSequenceClassification.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")
        model = model.to(device)
    else:
        load_model_location = 'iteration' + str(int(args.iter)-1) + '/saved_model'
        tokenizer = AutoTokenizer.from_pretrained(load_model_location)
        model = AutoModelForSequenceClassification.from_pretrained(load_model_location)
        model = model.to(device)
    print('Loaded model and shifted to device', device)
    
    #load data for fine tuning
    current_dir = 'iteration' + args.iter + '/'
    fine_tune_file = current_dir + 'fine_tune_' + args.iter + '.pkl'
    print('Fine Tune File:', fine_tune_file)
    label_dict = {'positive': 2, 'neutral': 1, 'negative': 0}
    sentences, labels = read_data(fine_tune_file, label_dict)
   
    label_distribution = Counter(labels)
    print('Train Label Distribution:')
    print(label_distribution)


    #dataloader
    batch_size = 16
    num_epochs = args.n_epochs
    train_dataloader = create_loader(tokenizer, sentences, labels, batch_size)
    print('Train Loader:', len(train_dataloader))

    optimizer = AdamW(model.parameters(),
                  lr = 2e-5, # args.learning_rate - default is 5e-5, our notebook had 2e-5
                  eps = 1e-8 # args.adam_epsilon  - default is 1e-8.
                )
    total_steps = len(train_dataloader) * num_epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, 
                                            num_warmup_steps = 0, # Default value in run_glue.py
                                            num_training_steps = total_steps)
    
    
    #load data for testing
    val_file = current_dir + 'processed_data_' + args.iter + '.pkl'
    print('Validation File:', val_file)
    sentences, labels = read_data_test(val_file , label_dict)
    label_distribution = Counter(labels)
    print('Validation Label Distribution:')
    print(label_distribution)

    batch_size = 128
    val_dataloader = create_loader(tokenizer, sentences, labels, batch_size)
    print('Validation Loader:', len(val_dataloader))

    #fine tuning loop
    for i in range(num_epochs):
        train_epoch(train_dataloader, model, optimizer, scheduler)
        eval_epoch(val_dataloader, model)
        
    print ("\tSaving model at epoch: {}\t".format(i))
    save_directory = current_dir + 'saved_model'
    os.makedirs(save_directory, exist_ok=True)
    tokenizer.save_pretrained(save_directory)
    model.save_pretrained(save_directory)
    
    
    
    
