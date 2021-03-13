import os
import time
import pickle
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import precision_recall_fscore_support as score
#Create a new repository called iteration 0. Inside that create a repo called logs.
#It should contain 'processed_data_0.pkl': this is the entire dataset initially. This dataset could be binary or trinary class.

#THIS CODES FINDS THE ACCURACY OF THE SELECTED SENTENCES TWEETS BASED ON MODEL AT THAT POINT

def load_data(filename):
    a_file = open(filename, "rb")
    output = pickle.load(a_file)
    a_file.close()
    return output

check_dir = 'selected'
os.makedirs(check_dir, exist_ok=True)

f = open(check_dir + "/accuracies_selected.txt", "a")


##make sure i > 0
for i in range(1,14):
    if i == 0:
        continue

    start = time.time()
    iter = str(i)

    #combine data files
    command = 'python3.6 combine.py --iter ' + iter
    print('\nLoaded prediction:', command)
    os.system( command + ' > ' + check_dir + '/combine_log')

    #Find accuracy of recursed files
    selected_file = 'iteration' + iter + '/selected_data_' + iter + '.pkl'
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


    f.write('Overall Accuracy:' + iter + ' : ' + str(overall_accuracy) + '\n')
    precision, recall, fscore, support = score(actual, predicted, average='macro')
    print('F1:', fscore, '  Precision:', precision, '  Recall:', recall )
    f.write(iter + ' -->  ' + 'F1:' + str(fscore) +  '  Precision:' + str(precision) + '  Recall:' + str(recall) + '\n')

    positive_accuracy = np.array([pred_label == label for (_, sentence, pred_label, label, score) in data if label == 'positive'])
    positive_accuracy = np.sum(positive_accuracy)/positive_accuracy.shape[0]
    print('Positive Accuracy:', positive_accuracy)
    f.write('Positive Accuracy:' + iter + ' : ' + str(positive_accuracy) + '\n')

    positive_accuracy = np.array([pred_label == label for (_, sentence, pred_label, label, score) in data if label == 'negative'])
    positive_accuracy = np.sum(positive_accuracy)/positive_accuracy.shape[0]
    print('Negative Accuracy:', positive_accuracy)
    f.write('Negative Accuracy:' + iter + ' : ' + str(positive_accuracy) + '\n\n')

    
    print('Time:', time.time() - start)
    
    f.write('='*20 + '\n')


f.close()

