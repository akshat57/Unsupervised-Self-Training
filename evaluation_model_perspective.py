import argparse
import os
import pickle
import time

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import precision_recall_fscore_support as score
from tqdm import tqdm

#Create a new repository called iteration 0. Inside that create a repo called logs.
#It should contain 'processed_data_0.pkl': this is the entire dataset initially. This dataset could be binary or trinary class.

#THIS CHECKS THE ACCURACY OF ALL MODELS WITH THE ENTIRE DATASET

def load_data(filename):
    a_file = open(filename, "rb")
    output = pickle.load(a_file)
    a_file.close()
    return output

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Runs validation on test set for single experiment (similar to check_models_fulldata.py)')
    parser.add_argument('--experiment_name', type=str, required=True, help='Enter name of experiment (used to name output file)')
    parser.add_argument('--test_data', type=str, required=True, help='Enter path to test dataset')
    parser.add_argument('--num_iters', type=int, required=False, default=15, help='Enter number of validation iters to run')
    args = parser.parse_args()

    check_dir = f'{args.experiment_name}/check'
    os.makedirs(check_dir, exist_ok=True)
    
    out_file_name = check_dir + '/' + args.experiment_name + '_test.txt'
    
    f = open(out_file_name, "a")
    
    for i in tqdm(range(0,args.num_iters + 1), total=args.num_iters+1):
        start = time.time()
        
        # write iter number at top of each entry
        f.write(f'Iter #{i}:\n')

        #loaded predictions
        check_file = args.test_data
        command = f'python prediction_finalcheck.py --experiment_name {args.experiment_name} --iter {i} --output ' + check_dir + '  --input ' + check_file 
        os.system( command + ' > ' + check_dir + '/checking_log')
        
        #FIND ACCURACY
        pred_file = check_dir + f'/iteration_{i}.pkl'
        data = load_data(pred_file)
        
        #find overall accuracy:
        overall_accuracy = np.array([pred_label == label for (tweet_id, sentence, pred_label, label, score) in data])
        overall_accuracy = np.sum(overall_accuracy)/len(data)
        print('Overall Accuracy:', overall_accuracy)
        
        #Find f1 score
        predicted = [pred_label for (tweet_id, sentence, pred_label, label, score) in data]
        actual = [label for (tweet_id, sentence, pred_label, label, score) in data]
        try:
            f.write(classification_report(actual, predicted, target_names=['negative', 'positive'], digits = 4))
        except:    
            f.write(classification_report(actual, predicted, target_names=['negative', 'neutral', 'positive'], digits = 4))
        f.write('\n') 

        # Calculate overall accuracy
        f.write('Overall Accuracy:'+ str(overall_accuracy) + '\n')
        precision, recall, fscore, support = score(actual, predicted, average='weighted')
        print('F1:', fscore, '  Precision:', precision, '  Recall:', recall )
        f.write('F1:' + str(fscore) +  '  Precision:' + str(precision) + '  Recall:' + str(recall) + '\n')
        
        # Calculate positive accuracy
        positive_accuracy = np.array([pred_label == label for (tweet_id, sentence, pred_label, label, score) in data if label == 'positive'])
        positive_accuracy = np.sum(positive_accuracy)/positive_accuracy.shape[0]
        print('Positive Accuracy:', positive_accuracy)
        f.write('Positive Accuracy:'+ str(positive_accuracy) + '\n')

        # Calculate negative accuracy
        positive_accuracy = np.array([pred_label == label for (tweet_id, sentence, pred_label, label, score) in data if label == 'negative'])
        positive_accuracy = np.sum(positive_accuracy)/positive_accuracy.shape[0]
        print('Negative Accuracy:', positive_accuracy)
        f.write('Negative Accuracy:'+ str(positive_accuracy) + '\n')
        
        # Calculate neutral accuracy
        positive_accuracy = np.array([pred_label == label for (tweet_id, sentence, pred_label, label, score) in data if label == 'neutral'])
        positive_accuracy = np.sum(positive_accuracy)/positive_accuracy.shape[0]
        print('Neutral Accuracy:', positive_accuracy)
        f.write('Neutral Accuracy:'+ str(positive_accuracy) + '\n\n')
        
        print('Time:', time.time() - start)
        time.sleep(5)
     
        f.write('='*20 + '\n')

    f.close()
    print(f"Finished writing validation output at {out_file_name}")
