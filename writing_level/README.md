# Writing Level Classifier

This is a set of scripts to help build and test a model that can classify the "writing level" of a student based on a text paragraph into 16 different levels of English proficiency as provided in the data. The structure is as follows:

1. xml_to_dataframe.py:
This file converts xml data to a dataframe and splits it into three groups
data_test, data_dev and data_train. Requires the xml data file to be in the 'data/' folder of the project. Execute this file first to create necessary data files required by the model.

2. data_processing.py:
This file contains all the necessary functions for data preprocessing and 
feeding the final vectorised data to the model.

3. model_functions.py:
This code has all the necessary functions to execute all model related 
functions from build, train to predict and more additional modules.

4. run_project.py:
This code runs the entire experiment for training and evaluating the model. Execute this code after the data creation mentioned in step 1. It will excecute the entire pipeline and display final metrics after completion.

5. Clustering.py:
This is a trail experiment to implement an unsupervised clustering algorithm using DBSCAN. After execution the clustering results are stored in a pickle file for inspection.