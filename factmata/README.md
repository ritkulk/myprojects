# Subjectivity Classifier

This project builds a binary classifier model to predict whether a sentence is subjective or objective. Sentences are obtained from the dataset released by Pang et.al, available here: http://www.cs.cornell.edu/people/pabo/movie-review-data/. Refer to the documentation for more details.

The code has been developed to work with Python 3.6.4 while other dependencies can be found in the “requirements.txt”. It is advised to create a python virtual environment and install the dependencies before running the experiment. 

The functionality of the code has been divided into three main files.
1) data_functions.py :
	This file contains all the necessary modules to obtain and pre-process data, up to the point of vectorising it in order to be fed into the neural network model for training-testing purposes

2) model_functions.py :
	This file contains all the necessary modules to build a Keras neural network model with a Tensorflow backend, along with modules to train, evaluate and get predictions using the model. There are two architectures available to experiment, a) CNN+LSTM network and b) 3 channel CNN network

3) subjectivity_classifier_run.py :
	This is the main script that runs the entire pipeline from data loading, processing, training and evaluating a cross validation experiment for the subjectivity classifier. You will only be required to run this file as ‘python3 subjectivity_classifier_run.py’ to execute a cross validation experiment. Parameters for this experiment can be set in this file itself.  For the current task only 3 parameters are set by the user while all others are internally defined. The three user set parameters with values for reported results  are 
a) batch_size = 100
b) epoch = 20
c) n_folds = 10

The folder structure of the project is as follows:  
-factmata/  
--setup.py  
--requirements.txt  
--README.md  
--main_project/  
----data/        ( holds the datafiles)  
----models/   ( models are saved here as h5 files)  
----data_functions.py  
----model_functions.py  
----subjectivity_classifier_run.py  
