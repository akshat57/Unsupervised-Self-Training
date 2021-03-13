import os
import time

#Create a new repository called iteration 0. Inside that create a repo called logs.
#It should contain 'processed_data_0.pkl': this is the entire dataset initially. This dataset could be binary or trinary class.

pos_ratio = 100#50
neg_ratio = 100#50
neu_ratio = 0
n_epochs = '1'

iter = '0'


#INITIALIZATION STEP
#Make zero shot predictions 
command = 'python3.6 prediction.py --iter ' + iter + ' --type zeroshot'
print('Zero shot prediction:', command)
os.system( command + ' > iteration' + iter + '/logs/prediction_zeroshot_log')

#Select top N sentences using selection criteria and split data
command = 'python3.6 datasplit.py --iter ' + iter + ' --positive ' + str(pos_ratio) + ' --negative ' + str(neg_ratio) + ' --neutral ' + str(neu_ratio)
print('Split Data:', command)
os.system(command + ' > iteration' + iter + '/logs/datasplit_logs')



for i in range(1, 15):
    start = time.time()
    iter = str(i)

    print('='*20)
    print()
    print('Working on iteration ' + iter)
    
    #FINE TUNE MODEL (FINE TUNE BLOCK)
    command = 'python3.6 finetune.py --iter ' + iter + ' --n_epochs ' + n_epochs
    print('\nFine Tuning:', command)
    os.system(command + ' > iteration' + iter + '/logs/fine_tune_logs')

    #MAKING PREDICTIONS FROM FINE TUNED MODEL (PREDICTION BLOCK)
    command = 'python3.6 prediction.py --iter ' + iter + ' --type load'
    print('\nLoaded prediction:', command)
    os.system( command + ' > iteration' + iter + '/logs/prediction_load_log')
    
    
    #SELECT AND SPLIT DATA (SELECTION BLOCK)
    command = 'python3.6 datasplit.py --iter ' + iter + ' --positive ' + str(pos_ratio) + ' --negative ' + str(neg_ratio) + ' --neutral ' + str(neu_ratio)
    print('\nSplit Data:', command)
    os.system(command + ' > iteration' + iter + '/logs/datasplit_logs')
    
    
    print('Time for 1 epoch:', time.time() - start)
    print()

