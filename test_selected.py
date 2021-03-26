import os
import pickle
import time
import argparse

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import precision_recall_fscore_support as score
from tqdm import tqdm

#Create a new repository called iteration 0. Inside that create a repo called logs.
#It should contain 'processed_data_0.pkl': this is the entire dataset initially. This dataset could be binary or trinary class.

#THIS CODES FINDS THE ACCURACY OF THE SELECTED SENTENCES/TWEETS BASED ON MODEL AT THAT POINT

def load_data(filename):
    a_file = open(filename, "rb")
    output = pickle.load(a_file)
    a_file.close()
    return output

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Zeroshot on dataset')
    #validation file location is hard coded.
    parser.add_argument('--experiment_name', type=str, required=True, help='Enter name of experiment (used to name outputs)')
    parser.add_argument('--num_iters', type=int, required=False, default=15, help='Enter number of validation iters to run')

    args = parser.parse_args()

    check_dir = f'{args.experiment_name}/selected'
    os.makedirs(check_dir, exist_ok=True)

    f = open(check_dir + "/accuracies_selected.txt", "a")


    ##make sure i > 0
    for i in tqdm(range(1,args.num_iters+1), total=args.num_iters):
        start = time.time()

        # write iter number at top of each entry
        f.write(f'Iter #{i}:\n')

        #combine data files
        command = f'python3 combine.py --experiment_name {args.experiment_name} --iter {i}'
        print('\nLoaded prediction:', command)
        os.system( command + ' > ' + check_dir + '/combine_log')

        #Find accuracy of recursed files
        selected_file = f'{args.experiment_name}/iteration{i}/selected_data_{i}.pkl'
        data = load_data(selected_file)
        
        #find overall accuracy:
        overall_accuracy = np.array([pred_label == label for (_, sentence, pred_label, label, score) in data])
        overall_accuracy = np.sum(overall_accuracy)/len(data)
        print('Overall Accuracy:', overall_accuracy)
        #f.write('Overall Accuracy: 'iter + ' : ' + str(overall_accuracy) + '\n')
        
        #Find f1 score
        predicted = [pred_label for (_, sentence, pred_label, label, score) in data]
        actual = [label for (_, sentence, pred_label, label, score) in data]
        try:
            f.write(classification_report(actual, predicted, target_names=['negative', 'positive'], digits = 4))
        except:    
            f.write(classification_report(actual, predicted, target_names=['negative', 'neutral', 'positive'], digits = 4))
        f.write('\n') 

        f.write(f'Overall Accuracy:{i} : ' + str(overall_accuracy) + '\n')
        precision, recall, fscore, support = score(actual, predicted, average='weighted')
        print('F1:', fscore, '  Precision:', precision, '  Recall:', recall )
        f.write(f'{i} -->  ' + 'F1:' + str(fscore) +  '  Precision:' + str(precision) + '  Recall:' + str(recall) + '\n')

        positive_accuracy = np.array([pred_label == label for (_, sentence, pred_label, label, score) in data if label == 'positive'])
        positive_accuracy = np.sum(positive_accuracy)/positive_accuracy.shape[0]
        print('Positive Accuracy:', positive_accuracy)
        f.write(f'Positive Accuracy:{i} : ' + str(positive_accuracy) + '\n')

        positive_accuracy = np.array([pred_label == label for (_, sentence, pred_label, label, score) in data if label == 'negative'])
        positive_accuracy = np.sum(positive_accuracy)/positive_accuracy.shape[0]
        print(f'Negative Accuracy:', positive_accuracy)
        f.write(f'Negative Accuracy:{i} : ' + str(positive_accuracy) + '\n')

        positive_accuracy = np.array([pred_label == label for (_, sentence, pred_label, label, score) in data if label == 'neutral'])
        positive_accuracy = np.sum(positive_accuracy)/positive_accuracy.shape[0]
        print(f'Neutral Accuracy:', positive_accuracy)
        f.write(f'Neutral Accuracy:{i} : ' + str(positive_accuracy) + '\n\n')
        
        print('Time:', time.time() - start)
        
        f.write('='*20 + '\n')


    f.close()

