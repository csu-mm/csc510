'''
MS - Artificial Intelligence and Machine Learning
Course: CSC510: Foundations of Artificial Intelligence
Module 6: Critical Thinking
Professor: Dr. Bingdong Li
Created by Mukul Mondal
February 18, 2026

Problem Statement: 
Naive Bayes classifiers are quick and easy to code in Python and are very efficient. 
Naive Bayes classifiers are based on Bayes' Theorem and assume independence among predictors 
(hence the "Naive" terminology). Not only are Naive Bayes classifiers handy and straightforward in a pinch, 
but they also outperform many other methods without the need for advanced feature engineering of the data.
Check out the following for further information on Naive Bayes classificationLinks to an external site.

Using scikit-learn, write a Naive Bayes classifier in Python. It can be single or multiple features. 
Submit the classifier in the form of an executable Python script alongside basic instructions for testing.
Your Naive Bayes classification script should allow you to do the following:
Calculate the posterior probability by converting the dataset into a frequency table.
Create a "Likelihood" table by finding relevant probabilities.
Calculate the posterior probability for each class.
Correct Zero Probability errors using Laplacian correction.
Your classifier may use a Gaussian, Multinomial, or Bernoulli model, depending on your chosen function. 
Your classifier must properly display its probability prediction based on its input data.

My solution: 
In this implementation, I created a Naive Bayes classifier that can handle both boolean and categorical features. 
It generates a synthetic dataset with boolean features (prior_imaging, heart_stroke, copd) and a categorical feature (pain_location). 
The target variable is "Is_MRI_SCAN_Needed", which is determined based on specific conditions. 
The classifier is then applied to predict whether an MRI scan is needed based on the input features.
I've created a class `NaiveBayesClassifier` in the `NaiveBayes.py` file that implements the Naive Bayes algorithm from scratch, 
as well as using scikit-learn's GaussianNB and BernoulliNB for comparison.
-to be installed:
pip install numpy
pip install pandas
pip install scikit-learn
'''

import os
import math
import json
import numpy as np 
import pandas as pd
import NaiveBayes as nbClassifier



def clearScreen():    
    if os.name == 'nt':  # For windows system
        _ = os.system('cls')    
    else:             # for non-windows system
        _ = os.system('clear')
    return


class App:
    def __init__(self) -> None:        
        return

    def ApplyNaiveBayesClassifier(self, dfInputLabled: pd.DataFrame, labledColumnName: str, new_sample: dict):
        bool_Feature_Names = ["prior_imaging","heart_stroke", "copd"]
        enum_Feature_names = ["pain_location"]
        enum_Feature_values = [1,2,3]

        nvbClassifier = nbClassifier.NaiveBayesClassifier()
        nvbClassifier.SetBoolFeatures(bool_Feature_Names)
        nvbClassifier.SetEnumFeatures(enum_Feature_names, enum_Feature_values)
        
        prediction = nvbClassifier.FindPrediction(dfInputLabled, labledColumnName, new_sample) # No library, basic Python implementation of: NaiveBayes Classifier Algorithm.
        print("Prediction using custom Python implementation:", prediction)
        print("Prediction using sklearn.GaussianNB:", nvbClassifier.Prediction_GaussianNB(dfInputLabled, labledColumnName, new_sample))
        print("Prediction using sklearn.BernoulliNB:", nvbClassifier.Prediction_BernoulliNB(dfInputLabled, labledColumnName, new_sample))
        return

    def CreateLabledData(self, N: int):
        # N: int = HowManyRows # 100
        rng = np.random.default_rng(42) 
        data = {
            "prior_imaging": rng.integers(0, 2, size=N),
            "heart_stroke": rng.integers(0, 2, size=N),
            "copd": rng.integers(0, 2, size=N),
            "pain_location": rng.integers(1, 4, size=N) # 1: Head, 2: Chest, 3: Lower body
        } 
        df = pd.DataFrame(data)
        df["Is_MRI_SCAN_Needed"] = False
        condition1 = (df["pain_location"] == 3) # MRI_NotNeeded 
        condition2 = ((df["prior_imaging"] == 1) | (df["heart_stroke"] == 1)) # MRI_Needed
        df.loc[~condition1 & condition2, "Is_MRI_SCAN_Needed"] = True
        return df

    def CreateRandomUnLabledData(self) -> dict:
        # N: int = HowManyRows # 100
        rng = np.random.default_rng(42) 
        data = {
            "prior_imaging": rng.integers(0, 2, size=1)[0],
            "heart_stroke": rng.integers(0, 2, size=1)[0],
            "copd": rng.integers(0, 2, size=1)[0],
            "pain_location": rng.integers(1, 4, size=1)[0] # 1: Head, 2: Chest, 3: Lower body
        }        
        return data
    
    def TakeInputFromUser(self) -> dict: 
        print("\nPlease enter your 'to be tested model's feature values:")
        img = (input("Do you have prior imaging? (y|Yes|yes for 'Yes', anything else for 'No'): ")).strip().lower()
        prior_imaging = 1 if img in ['y', 'yes'] else 0
        strk = (input("Do you have heart stroke? (y|Yes|yes for 'Yes', anything else for 'No'): ")).strip().lower()
        heart_stroke = 1 if strk in ['y', 'yes'] else 0
        cpd = (input("Do you have COPD? (y|Yes|yes for 'Yes', anything else for 'No'): ")).strip().lower()
        copd = 1 if cpd in ['y', 'yes'] else 0
        pain_loc = (input("Where is your pain located? (enter: 1 for Head, 2 for Chest, '3 or anything else' for Lower body): ")).strip().lower()
        pain_location = 3 # default value for 'anything else' input
        if pain_loc == '1':
            pain_location = 1
        elif pain_loc == '2':
            pain_location = 2
        return {
            "prior_imaging": prior_imaging,
            "heart_stroke": heart_stroke,
            "copd": copd,
            "pain_location": pain_location
        }

if __name__ == "__main__":
    clearScreen()
    print("\n ==== CSC510: Foundations of Artificial Intelligence ==== ")
    print(" ==== Module 6: Critical Thinking ==== ")
    print(" == Naive Bayes Classifier: Using scikit-learn, and custom python based implementation from scratch == ")

    app: App = App()

    # create Labeled dataset with 100 rows
    dfLabeled = app.CreateLabledData(1000)

    new_sample1 = app.CreateRandomUnLabledData()
    new_sample2 = {
                "prior_imaging": 1,
                "heart_stroke": 0,
                "copd": 1,
                "pain_location": 2 # 1: Head, 2: Chest, 3: Lower body
                }

    # Apply Naive Bayes Classifier on the dataset and predict for the new sample
    # app.ApplyNaiveBayesClassifier(dfInputLabled=dfLabeled, labledColumnName="Is_MRI_SCAN_Needed", new_sample=app.CreateRandomUnLabledData()) # works ok
    # app.ApplyNaiveBayesClassifier(dfInputLabled=dfLabeled, labledColumnName="Is_MRI_SCAN_Needed", new_sample=new_sample2) # works ok
    app.ApplyNaiveBayesClassifier(dfInputLabled=dfLabeled, labledColumnName="Is_MRI_SCAN_Needed", new_sample=app.TakeInputFromUser())
    
