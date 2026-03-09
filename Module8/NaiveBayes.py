'''
MS - Artificial Intelligence and Machine Learning
Course: CSC510: Foundations of Artificial Intelligence
Module 1-8: Portfolio Project
Professor: Dr. Bingdong Li
Created by Mukul Mondal
January February, 2026

Portfolio Project statement: 
Your Portfolio Project will be a fully-functioning AI program built to solve a real-world 
problem of your choosing, utilizing the tools and techniques outlined in this course. 
Your program will interact with human beings to support decision-making processes by 
delivering relevant information about the problem.

My solution: 
I chose to develop an AI based project intended to support a doctor’s office in 
identifying the diagnostic procedures and scan tests a patient may need before treatment. 
This portfolio project for CSC 510 – Foundations of Artificial Intelligence focuses on 
designing an AI‑assisted decision‑support system to help medical offices determine which 
diagnostic tests or scans a patient may require before treatment. The goal is not to replace clinical 
judgment, but to demonstrate how AI techniques—specifically classification and image 
analysis—can support early decision‑making in a healthcare workflow. 

I'll divided the whole project into two major logical components or parts. 
Part (a): Patient Questionnaire and Data Collection
    In this part, I will create a user-friendly interface to collect patient information through a structured questionnaire. 
    The questionnaire will cover various aspects of the patient's health, symptoms, and medical history relevant to 
    diagnostic procedures. The collected data will be stored in a structured format for further analysis.
    Apply Naive Bayes classifier to predict the diagnostic test needs based on the collected patient data.
    At the end of this part, I will predict if the patient requires specific diagnostic tests, MRI Scan.
Part (b): AI-Based analysis of MRI Scan Images
    In this part, I will develop an AI model that can analyze MRI scan images to identify potential abnormalities 
    or conditions. I will use a pre-trained convolutional neural network (CNN) architecture, 
    such as ResNet or VGG, and fine-tune it on a dataset of labeled MRI images. 
    The model will be trained to classify the images into categories such as "Normal," "Abnormal," or 
    specific conditions like "Tumor," "Fracture," etc. The AI model will provide insights and 
    recommendations based on the analysis of the MRI scan images, supporting the doctor's decision-making process
    in diagnosing and treating patients effectively.    
'''


import math
import json
from typing import List, Dict, Tuple
import numpy as np 
import pandas as pd

# use scikit-learn's MultinomialNB or BernoulliNB to calculate probabilities and make prediction
# (venv)...)>pip install scikit-learn
from sklearn.naive_bayes import GaussianNB 
from sklearn.model_selection import train_test_split 
from sklearn.metrics import accuracy_score

from sklearn.naive_bayes import BernoulliNB # BernoulliNB is suitable for binary features
from sklearn.preprocessing import LabelEncoder


# Part (a): Prediction based on Patient's Medical History using Naive Bayes Classifier
# Implementation class for 'Naive Bayes Classifier'
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
            f = f.lower().strip()
            self.bool_features.setdefault(f, False) # [False, True]
        return
    
    # This function sets the enum features and their possible values for the Naive Bayes classifier
    # For each enum feature, it also creates corresponding boolean features for each possible enum value (e.g., "Is_Feature1_Value1", "Is_Feature1_Value2", etc.)
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
      enumfeature_names = ["AGE_GROUP"]
        "AGE_GROUP": {
        "?": "How old are you?",
        "1": "Age 17 to 30",
        "2": "Age 31 to 40",
        "3": "Age 41 to 50",
        "4": "Age 51 to 65",
        "5": "Age 66 or Older",        
        "99": "Skip this question"
    },
    enum_features => bool_features = ["Is_AGE_GROUP_1", "Is_AGE_GROUP_2", "Is_AGE_GROUP_3", "Is_AGE_GROUP_4", "Is_AGE_GROUP_5"]
    '''

    # This function creates a DataFrame with all boolean features from X and y
    # This dataframe will be used to calculate probabilities for Naive Bayes
    # Input:
    #   X: features : list of dicts with keys: feature1, feature2, feature3, ...
    #   y: labels : list of class labels
    # Output:
    #   df: DataFrame with columns: boolfeature_names + enum2boolfeature_names + ['Is_MRI_SCAN_Needed']
    def CreateTrainingDataFrame(self, X, y):
        dfX = pd.DataFrame(X)        
        dfRet = pd.DataFrame()
        for f in dfX.columns:
            retf = f.lower().strip()
            if retf in self.bool_features:
                dfRet[retf] = dfX[f].astype(bool)
            elif retf in self.enum_features:
                for val in self.enum_features[retf]:
                    dfRet[f"is_{retf}_{val}"] = (dfX[f] == val).astype(bool)
        
        dfLabel = pd.DataFrame(y, columns=['Is_MRI_SCAN_Needed'])
        # Feature engineering:         
        # Ensure dfRet and dfLabel have the same number of rows before concatenation
        if dfRet.shape[0] < dfLabel.shape[0]:
            dfLabel = dfLabel[:(-1) * (dfLabel.shape[0] - dfRet.shape[0])]
        elif dfRet.shape[0] > dfLabel.shape[0]:
            dfRet = dfRet[:(-1) * (dfRet.shape[0] - dfLabel.shape[0])]
        
        return pd.concat([dfRet, dfLabel], axis=1)

    # This function creates a DataFrame for the new patient features with all boolean features, which will be used for prediction
    # Input:
    #   dfXLabeled: the training dataframe with all boolean features
    #   new_patient_features: a dictionary with the new patient's feature values
    #                         for any boolean feature, the value should be True or False (e.g., {"feature1": True, "feature2": False}).
    #                         for any enum feature, the value should be one of the possible enum values (e.g., {"feature1": "value1"})
    #                         e.g., for age_group, if the value is: age_group:3, then this will get converted to boolean features like:
    #                           "Is_AGE_GROUP_1": False, "Is_AGE_GROUP_2": False,"Is_AGE_GROUP_3": True, "Is_AGE_GROUP_4": False, "Is_AGE_GROUP_5": False.
    # Output:   df: DataFrame with columns: as in dfXLabeled.columns.drop('Is_MRI_SCAN_Needed').
    def CreatePredictionDataFrame(self, dfXLabeled, new_patient_features) -> pd.DataFrame :
        dict_new_patient_bool_features = {}
        for f in new_patient_features:
            ret2f = f.lower().strip()
            if ret2f in self.bool_features:
                dict_new_patient_bool_features[ret2f] = bool(new_patient_features.get(f, False))
            elif ret2f in self.enum_features:
                for val in self.enum_features[ret2f]:
                    dict_new_patient_bool_features[f"is_{ret2f}_{val}"] = (new_patient_features.get(f, None) == val)
        
        # feature engineering: ensure the new patient dataframe has the same columns as the training dataframe (except the label column)
        #  missing_in_df = [x for x in self.bool_features.keys() if x not in dfRet.columns.tolist()]
        retDf = pd.DataFrame([dict_new_patient_bool_features])
        retDf = retDf.reindex(columns=dfXLabeled.columns.drop('Is_MRI_SCAN_Needed'), fill_value=False)
        return retDf
    
    # scikit-learn's GaussianNB Naive Bayes classification.
    def Prediction_GaussianNB(self, X, y, new_patient_features):
        df = self.CreateTrainingDataFrame(X, y)
        #print(df.columns.tolist()) # has 'Is_MRI_SCAN_Needed'
        #pd.set_option("display.max_columns", None) # to display all columns in the dataframe.
        target = df.Is_MRI_SCAN_Needed # this is the target variable for training the model, which is the column 'Is_MRI_SCAN_Needed'.
        dfInput = df.drop('Is_MRI_SCAN_Needed', axis=1) # this column should not be present in the new patient data for training the model.
        dfNewPatientInput = self.CreatePredictionDataFrame(df, new_patient_features)
        
        X_train, X_test, y_train, y_test = train_test_split( dfInput, target, test_size=0.2, train_size=0.8, random_state=42 )
        
        # Train the GaussianNB model
        model = GaussianNB()
        model.fit(X_train, y_train)
        #print("model.predict_proba(X_test)", model.predict_proba(X_test)) # to show the predicted probabilities for the test set
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
    def Prediction_BernoulliNB(self, X, y, new_patient_features):
        df = self.CreateTrainingDataFrame(X, y)        
        target = df.Is_MRI_SCAN_Needed # this is the target variable for training the model, which is the column 'Is_MRI_SCAN_Needed'.
        dfInput = df.drop('Is_MRI_SCAN_Needed', axis=1) # this column should not be present in the new patient data for training the model.
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
    #   X: list of dicts with keys: feature1, feature2, feature3, ...
    #   y: list of class labels
    #   new_patient_features: dict with keys: feature1, feature2, feature3, ...user input features for a new patient
    # Output:
    #   prediction: display the predicted class label for the new patient along with the probabilities for each class label
    #                  (e.g., "MRI_Needed (70.00%)" or "MRI_NotNeeded (30.00%)")
    #   return: boolean value indicating whether MRI scan is needed (True) or not needed (False) for the new patient.
    def FindPrediction(self, X, y, new_patient_features) -> bool:   
        df = self.CreateTrainingDataFrame(X, y)
        #pd.set_option("display.max_columns", None)
        #print(df.head(25)) # ok

        # Convert input new_patient_features to all boolean features in dictionary for prediction
        dict_new_patient_bool_features = {}
        for f in new_patient_features:
            ret2f = f.lower().strip()
            if ret2f in self.bool_features:
                dict_new_patient_bool_features[ret2f] = bool(new_patient_features.get(f, False))
            elif ret2f in self.enum_features:
                for val in self.enum_features[ret2f]:
                    dict_new_patient_bool_features[f"is_{ret2f}_{val}"] = (new_patient_features.get(f, None) == val)

        print("\nCalculating probabilities using Naive Bayes prediction Algorithm...")
        dfMriNeededY = df[df['Is_MRI_SCAN_Needed'] == True]
        dfMriNeededN = df[df['Is_MRI_SCAN_Needed'] == False]
        count_MRI_Needed_Y = dfMriNeededY.shape[0]  # count of rows where MRI_Needed is True
        count_MRI_Needed_N = dfMriNeededN.shape[0]  # count of rows where MRI_Needed is False
        p_MRI_Needed_Y = count_MRI_Needed_Y / df.shape[0]
        p_MRI_Needed_N = count_MRI_Needed_N / df.shape[0]        
        #print(f"Count(MRI_Needed=True): {count_MRI_Needed_Y} , Count(MRI_Needed=False): {count_MRI_Needed_N}")
        #print(f"Probability(MRI_Needed=True): {p_MRI_Needed_Y:.4f} , Probability(MRI_Needed=False): {p_MRI_Needed_N:.4f}")
        
        pYes = p_MRI_Needed_Y
        pNo = p_MRI_Needed_N
        for f in dict_new_patient_bool_features.keys():
            if dfMriNeededY[dfMriNeededY[f] == dict_new_patient_bool_features[f]].shape[0] == 0 \
            or dfMriNeededN[dfMriNeededN[f] == dict_new_patient_bool_features[f]].shape[0] == 0:
                #print(f"\nFeature '{f}' has zero count for one of the classes. Applying Laplace smoothing.")
                pYes *= (dfMriNeededY[dfMriNeededY[f] == dict_new_patient_bool_features[f]].shape[0] + 1) / (count_MRI_Needed_Y + 2)
                pNo *= (dfMriNeededN[dfMriNeededN[f] == dict_new_patient_bool_features[f]].shape[0] + 1) / (count_MRI_Needed_N + 2)                
            else:
                pYes *= (dfMriNeededY[dfMriNeededY[f] == dict_new_patient_bool_features[f]].shape[0]) / (count_MRI_Needed_Y)
                pNo *= (dfMriNeededN[dfMriNeededN[f] == dict_new_patient_bool_features[f]].shape[0]) / (count_MRI_Needed_N)
        
        print(f"\nUnnormalized probabilities: (MRI_Needed=True): {pYes} , (MRI_Needed=False): {pNo}")
        
        # Calculate the percentage for each class prediction/probability
        percent_Y = 100 * (pYes / (pYes + pNo))
        percent_N = 100 * (pNo / (pYes + pNo))

        #prediction = "MRI_Needed ({:.2f}%)".format(percent_Y) if pYes > pNo else "MRI_NotNeeded ({:.2f}%)".format(percent_N)
        print(f"Normalized Probabilities:  (MRI_Needed=True): {percent_Y:.2f}% , (MRI_Needed=False): {percent_N:.2f}%")
        
        return pYes > pNo  #prediction
    
# End of NaiveBayes.py