# Unsupervised-Self-Training

The code contains the pipeline for Unsupervised Self Training.

Run the main.py file to run the pipeline. The pipeline very closely resembles the pipeline described in the paper. The code is split into the following steps:

A. INITIALIZATION STEP
  1. Make zero shot predictions
  2. Select top N sentences using selection criteria and split data

B. ITERATIVELY FINE TUNE MODEL
  1. Fine tune model based on selected sentences (FINE TUNE BLOCK)
  2. Make predictions from fine tuned model (PREDICTION BLOCK)
  3. Select and Split data (SELECTION BLOCK)
  

NOTE ON DATA FILE:
The input data file for the code is a .pkl file, where the pkl file contains a list. 
Each element of the list is a training sample, where data is stored in the form of a tuple
The schema of the tuple should be : (sample_id, sample, label)
