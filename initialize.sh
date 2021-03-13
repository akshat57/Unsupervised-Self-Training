#Enter the location of your data file in the variable below. The datafile should be a pickle file that can be processed by the code.

datafile="entiment_spanglish_train_binary.pkl"

mkdir iteration0
mkdir iteration0/logs
cp $datafile iteration0/processed_data_0.pkl 
