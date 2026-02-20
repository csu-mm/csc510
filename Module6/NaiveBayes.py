'''
MS - Artificial Intelligence and Machine Learning
Course: CSC510: Foundations of Artificial Intelligence
Module 6: Critical Thinking
Professor: Dr. Bingdong Li
Created by Mukul Mondal
February 18, 2026

In this source file, I've created a Naive Bayes classifier that can handle both boolean and categorical features. 
It generates a synthetic dataset with boolean features (prior_imaging, heart_stroke, copd) and a categorical feature (pain_location). 
The target variable is "Is_MRI_SCAN_Needed", which is determined based on specific conditions. 
The classifier is then applied to predict whether an MRI scan is needed based on the input features.
I've created a class `NaiveBayesClassifier` in the `NaiveBayes.py` file that implements the Naive Bayes algorithm from scratch, 
as well as using scikit-learn's GaussianNB and BernoulliNB for comparison.
'''


import math
import json

import numpy as np 
import pandas as pd


from sklearn.naive_bayes import GaussianNB 
from sklearn.model_selection import train_test_split 
from sklearn.metrics import accuracy_score

from sklearn.naive_bayes import BernoulliNB # BernoulliNB is suitable for binary features
from sklearn.preprocessing import LabelEncoder          


class NaiveBayesClassifier:
    def __init__(self):
        self.bool_features = {} # key: feature name, value: [False, True])
        self.enum_features = {} # key: feature name, value: possible enum values of that feature

    # This function sets the boolean features and their possible values for the Naive Bayes classifier
    # Input:
    #   bfeature_names: list of boolean feature names (e.g., ["feature1", "feature2"])
    def SetBoolFeatures(self, bfeature_names):
        if bfeature_names is None or len(bfeature_names) == 0:
            return
        for f in bfeature_names:            
            self.bool_features.setdefault(f.lower().strip(), False) # [False, True]
        return
    
    # This function sets the enum features and their possible values for the Naive Bayes classifier
    # Input:
    #   efeature_names: list of enum feature names (e.g., ["feature1", "feature2"])
    #   efeature_values: list of possible values for each enum feature
    def SetEnumFeatures(self, efeature_names, efeature_values):
        if efeature_names is None or len(efeature_names) == 0:
            return
        if efeature_values is None or len(efeature_values) == 0:
            return
        
        for f in efeature_names:
            f = f.lower().strip()
            self.enum_features.setdefault(f, efeature_values)
            for val in efeature_values:
                self.bool_features.setdefault(f"is_{f}_{val}", False)  # [False, True], add boolean features for each possible enum values
        return
    
    '''
    Example:--
      enumfeature_names = ["pain_location"]
      enumfeature_values = [1, 2, 3] # 1: Head, 2: Chest, 3: Lower body
    enum_features => bool_features = ["Is_pain_location_1", "Is_pain_location_2", "Is_pain_location_3"]
    '''

    # This function creates a DataFrame with all boolean features from dfInputLbled, which will be used for training the Naive Bayes model.
    # This dataframe will be used to calculate probabilities for Naive Bayes
    # Input:
    #   dfInputLbled: DataFrame with the input features and labels
    #   lblColName: name of the label column
    # Output:
    #   df: DataFrame with columns: boolfeature_names, enum features converted to boolean, label column
    # Feature engineering:
    def CreateTrainingDataFrame(self, dfInputLbled: pd.DataFrame, lblColName: str):
        if dfInputLbled.empty or lblColName is None or len(lblColName) == 0 or lblColName not in dfInputLbled.columns: 
            print("Input DataFrame is empty or label column not found. Please try again with valid input data.")
            exit(0)
        dfRet = pd.DataFrame()
        for f in [col for col in dfInputLbled.columns if col.strip().lower() != lblColName.strip().lower()]:
            retf = f.lower().strip()
            if retf in self.bool_features:
                dfRet[retf] = dfInputLbled[f].astype(bool)
            elif retf in self.enum_features:
                for val in self.enum_features[retf]:
                    dfRet[f"is_{retf}_{val}"] = (dfInputLbled[f] == val).astype(bool)
        return pd.concat([dfRet, dfInputLbled[lblColName]], axis=1)

    # This function creates a DataFrame for the new patient features with all boolean features, which will be used for prediction.
    # Input:
    #   dfXLabeled: the training dataframe with all boolean features
    #   new_patient_features: a dictionary with the new patient's feature values
    # Output:   df: DataFrame with columns: as in dfXLabeled.columns.drop('Is_MRI_SCAN_Needed').
    def CreatePredictionDataFrame(self, dfXLabeled, new_patient_features: dict):
        dict_new_patient_bool_features = {}
        for f in new_patient_features:
            ret2f = f.lower().strip()
            if ret2f in self.bool_features:
                dict_new_patient_bool_features[ret2f] = bool(new_patient_features.get(f, False))
            elif ret2f in self.enum_features:
                for val in self.enum_features[ret2f]:
                    dict_new_patient_bool_features[f"is_{ret2f}_{val}"] = (new_patient_features.get(f, None) == val)
        
        # feature engineering: ensure the new patient dataframe has the same columns as the training dataframe (except the label column)
        #missing_in_df = [x for x in self.bool_features.keys() if x not in dfRet.columns.tolist()]
        retDf = pd.DataFrame([dict_new_patient_bool_features])        
        return retDf
    
    # scikit-learn's GaussianNB Naive Bayes classification.
    def Prediction_GaussianNB(self, dfLabled: pd.DataFrame, labledColName: str, new_patient_features: dict):
        df = self.CreateTrainingDataFrame(dfLabled, labledColName)
        if df.empty:
            print("Training DataFrame is empty. Please try again with valid input data.")
            exit(0)
        #print(df.columns.tolist()) # has 'Is_MRI_SCAN_Needed'
        #pd.set_option("display.max_columns", None) # to display all columns in the dataframe.
        target = df[labledColName] # this is the target variable for training the model, which is the column 'Is_MRI_SCAN_Needed'.
        dfInput = df.drop(labledColName, axis=1) # this column should not be present in the new patient data for training the model.
        dfNewPatientInput = self.CreatePredictionDataFrame(df, new_patient_features)
        
        X_train, X_test, y_train, y_test = train_test_split( dfInput, target, test_size=0.2, train_size=0.8, random_state=42 )
        # Train the GaussianNB model
        model = GaussianNB()
        model.fit(X_train, y_train)        
        prediction_probability = model.predict_proba(dfNewPatientInput)[0]
        px = prediction_probability[0] + prediction_probability[1]
        prediction_probability = [(100*prediction_probability[0])/px, (100*prediction_probability[1])/px]
        model.score(X_test, y_test)
        y_pred = model.predict(X_test)        
        print(f"\nGaussianNB: Accuracy : {accuracy_score(y_test, y_pred):.4f}")
        y_pred2 = model.predict(dfNewPatientInput)        
        #print("\nGaussianNB: Predicted class label:", y_pred2)
        print(f"GaussianNB: Predicted class label: {y_pred2[0]} , probabilities: {prediction_probability}")
        return y_pred2[0]
    
    # scikit-learn's: BernoulliNB Naive Bayes classification.
    def Prediction_BernoulliNB(self, dfLabled: pd.DataFrame, labledColName: str, new_patient_features: dict):
        df = self.CreateTrainingDataFrame(dfLabled, labledColName)
        if df.empty:
            print("Training DataFrame is empty. Please try again with valid input data.")
            exit(0)        
        target = df[labledColName] # this is the target variable for training the model, which is the column 'Is_MRI_SCAN_Needed'.
        dfInput = df.drop(labledColName, axis=1) # this column should not be present in the new patient data for training the model.
        dfNewPatientInput = self.CreatePredictionDataFrame(df, new_patient_features)
        
        X_train, X_test, y_train, y_test = train_test_split( dfInput, target, test_size=0.2, train_size=0.8, random_state=42 )

        # Encode the target variable
        le = LabelEncoder()
        y_train_encoded = le.fit_transform(y_train)
        # Train the BernoulliNB model
        model = BernoulliNB()
        model.fit(X_train, y_train_encoded)
        
        # Make the prediction
        prediction_probability = model.predict_proba(dfNewPatientInput)[0]
        px = prediction_probability[0] + prediction_probability[1]
        prediction_probability = [(100*prediction_probability[0])/px, (100*prediction_probability[1])/px]
        #model.score(X_test, le.transform(y_test))
        prediction_label = le.inverse_transform([model.predict(dfNewPatientInput)[0]])[0]
        print(f"\nBernoulliNB: Predicted class label: {prediction_label} , probabilities: {prediction_probability}")
        return prediction_label



    # This function does all the work to predict the class label for a new patient using principle of 'Naive Bayes classifier'.
    #   python based implementation of Naive Bayes classifier for binary classification without using any external libraries.
    # Input:
    #   dfLabled: DataFrame with columns: feature1, feature2, feature3, ... and the label column
    #   labledColName: name of the label column
    #   unLabledSample: dictionary, ...user input features of the new patient for prediction.
    # Output:
    #   prediction: predicted class label for the new patient (True/False for MRI_Needed)    
    def FindPrediction(self, dfLabled: pd.DataFrame, labledColName: str, unLabledSample: dict):
        df = self.CreateTrainingDataFrame(dfLabled, labledColName)
        if df.empty:
            print("Training DataFrame is empty. Please try again with valid input data.")
            exit(0)
        #pd.set_option("display.max_columns", None)
        #print(df.head(10)) # ok        

        # Convert input new_patient_features to all boolean features in dictionary for prediction
        # For boolean features, the value will be directly used as True/False.
        # For enum features, we will create new boolean features for each possible value of the enum feature. 
        # For example, if we have an enum feature "pain_location" with possible values [1, 2, 3], 
        #    we will create boolean features "Is_pain_location_1", "Is_pain_location_2", "Is_pain_location_3". 
        # The value of these boolean features will be True for the corresponding enum value and False for others.
        dict_new_patient_bool_features = {}
        for f in unLabledSample:
            ret2f = f.lower().strip()
            if ret2f in self.bool_features:
                dict_new_patient_bool_features[ret2f] = bool(unLabledSample.get(f, False))
            elif ret2f in self.enum_features:
                for val in self.enum_features[ret2f]:
                    dict_new_patient_bool_features[f"is_{ret2f}_{val}"] = (unLabledSample.get(f, None) == val)
        
        print("\nNew patient features (converted to boolean features for prediction):")
        for k, v in dict_new_patient_bool_features.items():
            print(f"{k}: {v}")
        print("\nCalculating probabilities using Naive Bayes prediction Algorithm...")
        df_TestNeeded_Yes = df[df[labledColName] == True]
        df_TestNeeded_No = df[df[labledColName] == False]
        count_MRI_Needed_Y = df_TestNeeded_Yes.shape[0]  # count of rows where Test Needed is True
        count_MRI_Needed_N = df_TestNeeded_No.shape[0]  # count of rows where Test Needed is False
        pYes = count_MRI_Needed_Y / df.shape[0]
        pNo = count_MRI_Needed_N / df.shape[0]

        for f in dict_new_patient_bool_features.keys():
            if df_TestNeeded_Yes[df_TestNeeded_Yes[f] == dict_new_patient_bool_features[f]].shape[0] == 0 \
             or df_TestNeeded_No[df_TestNeeded_No[f] == dict_new_patient_bool_features[f]].shape[0] == 0:
                #If a feature has zero count for one of the classes, apply Laplace smoothing
                pYes *= (df_TestNeeded_Yes[df_TestNeeded_Yes[f] == dict_new_patient_bool_features[f]].shape[0] + 1) / (count_MRI_Needed_Y + 2)
                pNo *= (df_TestNeeded_No[df_TestNeeded_No[f] == dict_new_patient_bool_features[f]].shape[0] + 1) / (count_MRI_Needed_N + 2)
            else:
                pYes *= (df_TestNeeded_Yes[df_TestNeeded_Yes[f] == dict_new_patient_bool_features[f]].shape[0]) / (count_MRI_Needed_Y)
                pNo *= (df_TestNeeded_No[df_TestNeeded_No[f] == dict_new_patient_bool_features[f]].shape[0]) / (count_MRI_Needed_N)
        
        #print(f"\nUnnormalized probabilities: (MRI_Needed=True): {pYes} , (MRI_Needed=False): {pNo}")
        
        # Calculate the percentage for each class prediction/probability
        percent_Y = 100 * (pYes / (pYes + pNo))
        percent_N = 100 * (pNo / (pYes + pNo))
        
        print(f"Normalized Probabilities:  (Test_Needed=True): {percent_Y:.2f}% , (Test_Needed=False): {percent_N:.2f}%")
        
        return pYes > pNo  #prediction
    
# End of NaiveBayes.py