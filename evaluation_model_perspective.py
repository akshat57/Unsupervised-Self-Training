import os
import time
import pickle
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import precision_recall_fscore_support as score
#Create a new repository called iteration 0. Inside that create a repo called logs.
#It should contain 'processed_data_0.pkl': this is the entire dataset initially. This dataset could be binary or trinary class.

#THIS CHECKS THE ACCURACY OF ALL MODELS WITH THE ENTIRE DATASET


def load_data(filename):
    a_file = open(filename, "rb")
    output = pickle.load(a_file)
    a_file.close()
    return output

check_dir = 'check'
os.makedirs(check_dir, exist_ok=True)

f = open(check_dir + "/accuracies.txt", "a")


for i in range(0,14):
    start = time.time()
    iter = str(i)

    #loaded predictions
    check_file = 'sentiment_mal_train_binary.pkl'
    command = 'python3.6 prediction_finalcheck.py --iter ' + iter + ' --output ' + check_dir + '  --input ' + check_file 
    print('\nLoaded prediction:', command)
    os.system( command + ' > ' + check_dir + '/checking_log')
    
    #FIND ACCURACY
    pred_file = check_dir + '/iteration_' + iter + '.pkl'
    data = load_data(pred_file)
    
    #find overall accuracy:
    overall_accuracy = np.array([pred_label == label for (tweet_id, sentence, pred_label, label, score) in data])
    overall_accuracy = np.sum(overall_accuracy)/len(data)
    print('Overall Accuracy:', overall_accuracy)
    #f.write('Overall Accuracy: 'iter + ' : ' + str(overall_accuracy) + '\n')
    
    #Find f1 score
    predicted = [pred_label for (tweet_id, sentence, pred_label, label, score) in data]
    actual = [label for (tweet_id, sentence, pred_label, label, score) in data]
    try:
        f.write(classification_report(actual, predicted, target_names=['negative', 'positive'], digits = 4))
    except:    
        f.write(classification_report(actual, predicted, target_names=['negative', 'neutral', 'positive'], digits = 4))
    f.write('\n') 


    f.write('Overall Accuracy:' + iter + ' : ' + str(overall_accuracy) + '\n')
    precision, recall, fscore, support = score(actual, predicted, average='macro')
    print('F1:', fscore, '  Precision:', precision, '  Recall:', recall )
    f.write(iter + ' -->  ' + 'F1:' + str(fscore) +  '  Precision:' + str(precision) + '  Recall:' + str(recall) + '\n')

    positive_accuracy = np.array([pred_label == label for (tweet_id, sentence, pred_label, label, score) in data if label == 'positive'])
    positive_accuracy = np.sum(positive_accuracy)/positive_accuracy.shape[0]
    print('Positive Accuracy:', positive_accuracy)
    f.write('Positive Accuracy:' + iter + ' : ' + str(positive_accuracy) + '\n')

    positive_accuracy = np.array([pred_label == label for (tweet_id, sentence, pred_label, label, score) in data if label == 'negative'])
    positive_accuracy = np.sum(positive_accuracy)/positive_accuracy.shape[0]
    print('Negative Accuracy:', positive_accuracy)
    f.write('Negative Accuracy:' + iter + ' : ' + str(positive_accuracy) + '\n\n')

    
    print('Time:', time.time() - start)
    print('Time to break the loop begins:')
    time.sleep(5)
    print('Time to break the loop ends')
    
    f.write('='*20 + '\n')

f.close()

