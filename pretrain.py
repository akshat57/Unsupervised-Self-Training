import argparse
import datetime
import os
import pdb
import pickle
import time
from collections import Counter
from tqdm import tqdm

import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset, random_split)
from transformers import (AdamW, AutoModelForSequenceClassification,
                          AutoTokenizer, RobertaForSequenceClassification,
                          RobertaTokenizer, get_linear_schedule_with_warmup,
                          pipeline)

device = 0

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

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

        # smax = torch.softmax(logits, dim = 1)
        # indices = torch.argmax(smax, dim = 1)
        indices = torch.argmax(logits, dim = 1)
        correct += torch.sum(indices==b_labels).item()
        total += b_size


    avg_train_loss = total_train_loss / len(train_dataloader)  
    training_time = format_time(time.time() - t0)
    f.write("")
    f.write("  Average training loss: {0:.2f}".format(avg_train_loss))
    f.write("  Training epoch took: {:}".format(training_time))
    val_accuracy = correct / total
    f.write(" Training Accuracy: {0:.4f}".format(val_accuracy))
    f.write(f' Time:{time.time() - t0}\n')
    
def eval_epoch(validation_dataloader, model):
    t0 = time.time()
    model.eval()
    correct = 0
    total = 0
    all_predictions = []
    all_labels = []

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
            
        logits = outputs.logits
        # smax = torch.softmax(logits, dim = 1)
        # indices = torch.argmax(smax, dim = 1)
        indices = torch.argmax(logits, dim = 1)

        correct += torch.sum(indices==b_labels).item()
        total += b_size

        #saving
        all_predictions += list(indices.cpu().numpy())
        all_labels += list(b_labels.cpu().numpy())

    try:
        f.write(classification_report(all_labels, all_predictions, target_names=['negative', 'positive'], digits = 5))
    except:
        f.write(classification_report(all_labels, all_predictions, target_names=['negative', 'neutral', 'positive'], digits = 5))
    val_accuracy = correct / total
    
    return val_accuracy



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Supervised pretraining (based on EACLsentiment2021.ipynb)')
    parser.add_argument('--experiment_name', type=str, required=True, help='Enter name of experiment (used to name output file)')
    parser.add_argument('--num_epochs', type=int, required=True, help='Enter num epochs of supervised training')
    parser.add_argument('--pretrain_data', type=str, required=True, help='Enter path to pretrain data file')
    parser.add_argument('--predev_data', type=str, required=True, help='Enter path to pretrain data file')
    parser.add_argument('--pretest_data', type=str, required=True, help='Enter path to test data file')

    args = parser.parse_args()
    
    # make folder to store pretrained model and tokenizer
    os.makedirs(f'{args.experiment_name}/pretrain', exist_ok=True)
    os.makedirs(f'{args.experiment_name}/check', exist_ok=True)
    f = open(f"{args.experiment_name}/check/pretrain_log_{args.experiment_name}.txt", "a")
    
    tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")
    model = AutoModelForSequenceClassification.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")
    model = model.to(device)
    
    f.write(f"Pretraining {args.experiment_name} for {args.num_epochs} epochs\n")
    print(f"Pretraining {args.experiment_name} for {args.num_epochs} epochs")
    
    label_dict = {'positive': 2, 'neutral': 1, 'negative': 0}
    #sentences, labels = read_data('sentiment__train.pkl', label_dict)
    sentences, labels = read_data(args.pretrain_data, label_dict)

    #for multilingual training
    # '''sentences2, labels2 = read_data('sentiment_tamil_train.pkl', label_dict)
    # print(len(sentences))
    # sentences += sentences2[:1000]
    # labels += labels2[:1000]
    # print(len(sentences))
    # '''
    # print(Counter(labels))
    
    batch_size = 16
    train_dataloader = create_loader(tokenizer, sentences, labels, batch_size)
    
    optimizer = AdamW(model.parameters(),
                  lr = 2e-5, # args.learning_rate - default is 5e-5, our notebook had 2e-5
                  eps = 1e-8 # args.adam_epsilon  - default is 1e-8.
                )
    total_steps = len(train_dataloader) * args.num_epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, 
                                                num_warmup_steps = 0, # Default value in run_glue.py
                                                num_training_steps = total_steps)
    
    batch_size = 128
    sentences, labels = read_data(args.predev_data, label_dict)
    val_dataloader = create_loader(tokenizer, sentences, labels, batch_size)

    #for multilingual
    # '''sentences, labels = read_data('sentiment_tamil_dev.pkl', label_dict)
    # val_dataloader2 = create_loader(tokenizer, sentences, labels, batch_size)'''
    
    #eval_epoch(val_dataloader, model)
    #print('End Zero Shot')
    val_accuracies = []
    for i in tqdm(range(1, args.num_epochs+1), total=args.num_epochs):
        f.write(f"Epoch #{i}\n")
        train_epoch(train_dataloader, model, optimizer, scheduler)
        val_accuracy = eval_epoch(val_dataloader, model)
        val_accuracies.append(val_accuracy)
        
    sentences, labels = read_data(args.pretest_data, label_dict)
    test_dataloader2 = create_loader(tokenizer, sentences, labels, batch_size)
    eval_epoch(test_dataloader2, model)

    print(f"Finished pretraining, epoch val accuracies: {val_accuracies}")
    f.write(f"Finished pretraining, epoch val accuracies: {val_accuracies}\n")
    
    # Save model
    save_directory = f'{args.experiment_name}/pretrain/{args.experiment_name}_epoch_{i}.model'
    #os.makedirs(save_directory, exist_ok=True)
    print(f"\tSaving pretrained model at epoch: {i}\t")
    f.write(f"\tSaving pretrained model at epoch: {i}\t\n")
    tokenizer.save_pretrained(save_directory)
    model.save_pretrained(save_directory)
    
    # print(f"Deleting other epoch models")
    # # delete models that aren't from best epoch
    # for i in range(1, args.num_epochs+1):
    #     if i == best_epoch:
    #         continue
    #     os.system(f"rm -r pretrain/{args.experiment_name}_epoch_{i}.model")
