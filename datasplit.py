import argparse
import pickle
import numpy as np
import os
import random

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

def count(data):
    predicted_labels = [pred_label for (_, sentence, pred_label, label, score) in data]
    n_pos = 0
    n_neg = 0
    n_neu = 0

    for label in predicted_labels:
        if label == 'positive':
            n_pos += 1
        elif label == 'negative':
            n_neg += 1
        elif label == 'neutral':
            n_neu += 1

    return n_pos, n_neg, n_neu


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Datasplit')
    parser.add_argument('--iter', type=str, required=True, help='Enter Iteration number')
    parser.add_argument('--positive', type=int, required=True, help='Enter percentage of data to be split for a class')
    parser.add_argument('--negative', type=int, required=True, help='Enter percentage of data to be split for a class')
    parser.add_argument('--neutral', type=int, required=True, help='Enter percentage of data to be split for a class')
    args = parser.parse_args()
    
    print('Entered', args.iter)
    
    current_directory = 'iteration' + args.iter + '/'
    input_file = 'iteration' + args.iter + '/iteration_' + args.iter + '.pkl'
        
    #Load and count data
    data = load_data(input_file)
    n_pos, n_neg, n_neu = count(data)
    
    #find overall accuracy:
    overall_accuracy = np.array([pred_label == label for (_, sentence, pred_label, label, score) in data])
    print('Overall Accuracy:', np.sum(overall_accuracy)/len(data))
    
    #Store in different dictionaries based on actual label
    pred_positive = []
    pred_negative = []
    pred_neutral = []
    for (tweet_id, sentence, pred_label, label, score) in data:
        if pred_label == 'positive':
            pred_positive.append((tweet_id, sentence, pred_label, label, score))
        elif pred_label == 'negative':
            pred_negative.append((tweet_id, sentence, pred_label, label, score))
        elif pred_label == 'neutral':
            pred_neutral.append((tweet_id, sentence, pred_label, label, score))
        
    print('Check:', (len(pred_positive) + len(pred_negative) + len(pred_neutral)) == len(data))
    print('Total Dataset:', len(data), 'Positive:', len(pred_positive), 'Negative:', len(pred_negative), 'Neutral:', len(pred_neutral))
    
    #Original Accuracy for predictions made
    positive_accuracy = np.array([pred_label == label for i, (_, sentence, pred_label, label, score) in enumerate(pred_positive)])
    negative_accuracy = np.array([pred_label == label for i, (_, sentence, pred_label, label, score) in enumerate(pred_negative)])
    neutral_accuracy = np.array([pred_label == label for i, (_, sentence, pred_label, label, score) in enumerate(pred_neutral)])

    print('Positive Accuray:', np.sum(positive_accuracy)/len(pred_positive))
    print('Negative Accuracy:', np.sum(negative_accuracy)/len(pred_negative))
    print('Neutral Accuracy:', np.sum(neutral_accuracy)/len(pred_neutral))

    
    ##Sorting by predicting confidence
    sorted_pred_positive = sorted(pred_positive, key=lambda k: k[4], reverse=True)
    sorted_pred_negative = sorted(pred_negative, key=lambda k: k[4], reverse=True)
    sorted_pred_neutral = sorted(pred_neutral, key=lambda k: k[4], reverse=True)
    
    
    ##Total number of samples in top10% high confidence predictions for each class
    n_top10_positive = min(args.positive, n_pos) #(n_pos * args.positive)//100
    n_top10_negative = min(args.negative, n_neg)#(n_neg * args.negative)//100
    n_top10_neutral = 0#(n_neu * args.neutral)//100
    
    print('Number chosen:', n_top10_positive, n_top10_negative, n_top10_neutral)
    
    
    #Find top 10 percent accuracy
    positive_top10_accuracy = np.array([pred_label == label for i, (_, sentence, pred_label, label, score) in enumerate(sorted_pred_positive) if i < n_top10_positive ])
    negative_top10_accuracy = np.array([pred_label == label for i, (_, sentence, pred_label, label, score) in enumerate(sorted_pred_negative) if i < n_top10_negative ])
    neutral_top10_accuracy = np.array([pred_label == label for i, (_, sentence, pred_label, label, score) in enumerate(sorted_pred_neutral) if i < n_top10_neutral ])

    print('Top10 Confidence Positive Accuray:', np.sum(positive_top10_accuracy)/n_top10_positive)
    print('Top10 Confidence Negative Accuracy:', np.sum(negative_top10_accuracy)/n_top10_negative)
    print('Top10 Confidence Neutral Accuracy:', np.sum(neutral_top10_accuracy)/n_top10_neutral)

    
    #Pick top chosen percent
    top_confidence_positive = sorted_pred_positive[:n_top10_positive]
    top_confidence_negative = sorted_pred_negative[:n_top10_negative]
    top_confidence_neutral = sorted_pred_neutral[:n_top10_neutral]
    
    #create new train data set that does not have saved sentences
    new_directory = 'iteration' + str(int(args.iter) + 1)
    os.mkdir(new_directory)
    os.mkdir(new_directory + '/logs')
    
    fine_tune_data = top_confidence_positive + top_confidence_negative + top_confidence_neutral
    random.shuffle(fine_tune_data)
    print('Check:', len(fine_tune_data) == len(top_confidence_positive) + len(top_confidence_negative) + len(top_confidence_neutral))
    save_data(new_directory + '/fine_tune_' + str(int(args.iter) + 1) + '.pkl', fine_tune_data)
    
    #Collecting chosen sentences and removing them from original dataset, and saving
    chosen_sentences = [sentence for (_, sentence, pred_label, label, score) in fine_tune_data]
    original_dataset = load_data(current_directory + 'processed_data_' + args.iter + '.pkl')
    
    next_iteration_data = []
    for (tweet_id, tweet, label) in original_dataset:
        if tweet not in chosen_sentences:
            next_iteration_data.append((tweet_id, tweet, label))
            
    save_data( new_directory + '/processed_data_' + str(int(args.iter) + 1) + '.pkl', next_iteration_data)

    
