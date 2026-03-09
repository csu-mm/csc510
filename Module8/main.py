'''
MS - Artificial Intelligence and Machine Learning
Course: CSC510: Foundations of Artificial Intelligence
Module 1-8: Portfolio Project
Professor: Dr. Bingdong Li
Created by Mukul Mondal
January February, 2026
'''

import os
from os import system, name
import json
from typing import List, Dict, Tuple
from pathlib import Path
import tensorflow as tf
from tensorflow.python.keras import Sequential
keras = tf.keras   #from tensorflow import keras
import matplotlib.pyplot as plt
#%matplotlib inline
import numpy as np
#import seaborn as sn
import pandas as pd
from sklearn.model_selection import train_test_split  # pip install scikit-learn
from sklearn.preprocessing import OneHotEncoder
from PIL import Image   # needed: pip install Pillow

import NaiveBayes as nbClassifier
from cnn_brain_tumor import MriBrainTumorCNN

import tkinter as tk
from tkinter import messagebox


# This function just clears the current screen
def clearScreen():
    if name == 'nt':  # For windows
        _ = system('cls')
    else:             # For mac and linux(here, os.name is 'posix')
        _ = system('clear')
    return

# This class represents a patient and their details, including basic profile information and medical history.
# It includes methods to initiate and run a questionnaire to collect relevant data for diagnostic test prediction.
# Part (a): Patient Questionnaire and Data Collection
class Patient:
    def __init__(self, pName: str, pAddress: str, pEthnicity: str) -> None:
        self.patient_name = pName # Patient's name, part of the basic profile information.
        self.patient_address = pAddress # Patient's address, part of the basic profile information.
        self.patient_ethnicity = pEthnicity # Patient's ethnicity, part of the basic profile information.
        self.patient_details = {} # Patient medical history. 
                                  # A dictionary to store the patient's responses to the questionnaire, which will be used for diagnostic test prediction.
        return

    # This method initializes the patient medical history details.
    def initiateQuestionnaire(self) -> None:
        self.patient_details = self.RunQuestionnaire()
        return
    
    # This method runs the questionnaire to collect patient details. 
    # It interacts with the user to gather responses to a series of questions defined in a JSON file. 
    # The responses are stored in a dictionary and returned at the end of the questionnaire.
    def RunQuestionnaire(self) -> dict:
        print("Loading initial questionnaires. . .")
        user_input = UserInput()
        patientDetails = {}
        patientDetails["name"] = self.patient_name
        patientDetails["address"] = self.patient_address
        patientDetails["ethnicity"] = self.patient_ethnicity
        bProceed: bool = True

        question_variables = []
        statFeaturesEnum, _ = user_input.GetStatFeaturesEnum()
        sStatFeaturesBool, _ = user_input.GetStatFeaturesBool()
        question_variables = statFeaturesEnum + sStatFeaturesBool

        qIndex: int = 0
        while bProceed and qIndex < len(question_variables):
            userAns = user_input.GetUserResponse(question_variables[qIndex]).strip().lower()
            if userAns == 'p':  # go to previous question
                if qIndex > 0:
                    qIndex -= 1
            elif userAns == 'n' or userAns == '99':  # go to next question
                qIndex += 1
            elif userAns.lower() == 'q':
                bProceed = False
            else:
                uAnsInt: int = -1
                try:
                    uAnsInt = int(userAns)
                    if uAnsInt != 99:
                        patientDetails[question_variables[qIndex]] = uAnsInt
                except:
                        pass #patientDetails[question_variables[qIndex]] = -1
                qIndex += 1
        return patientDetails
    
# This class handles user input for the patient questionnaire. It loads questions from a JSON file and 
# provides methods to retrieve non-statistical and statistical features, as well as to get user responses to specific questions.
# This helper class is designed to facilitate the collection of patient data through a structured questionnaire, 
#             which will be used for diagnostic test prediction.
# Part (a): Patient Questionnaire and Data Collection
class UserInput:
    def __init__(self) -> None:        
        self.questions_dict = json.load(open('data/patient_questionnaires.json'))
        return
    
    # These are: non-statistical features, not used for prediction.
    # Shows features that are part of the Patient Profile information.
    def GetNonStatFeatures(self) -> list:
        return ["name", "state", "address", "interview_status", "ethnicity"]
    
    # These are: statistical features, used for prediction.
    # Shows features that are part of the Patient Medical History and the possible feature values.
    # These features are enum type.
    def GetStatFeaturesEnum(self):
        featureNames: list = ["age_group", "general_health", "weight_category", 
                              "recent_surgery_days_group", "pain_level", 
                              "pain_location", "anxiety_level"]
        featureValues: list = [1,2,3,4,5]
        return featureNames, featureValues
    
    # These are: statistical features, used for prediction.
    # Shows features that are part of the Patient Medical History and the possible feature values.
    # These features are boolean type.
    def GetStatFeaturesBool(self):
        featureNames: list = ["gender", "pregnant", "claustrophobic", "metal_implants", 
                              "pacemaker", "prior_imaging", "contrast_allergy", 
                              "can_remain_still", "heart_coronary", "heart_stroke", 
                              "asthma", "cancer", "diabetes", "copd"]
        featureValues: list = [0,1]
        return featureNames, featureValues
    
    # This method defines the common structure for prompting the user with a specific question based on the provided question and collecting the user's response.
    # Ask user/patient a question and return the user response as a string
    def GetUserResponse(self, qKey: str) -> str:
        if qKey is None or len(qKey.strip()) == 0:
            print("Question key cannot be empty. Please provide a valid question key.")
            return ""
        qKey = qKey.strip().upper()
        if qKey not in self.questions_dict:
            print(f"Question key '{qKey}' not found in the questions dictionary.")
            return ""
        
        uResponse: str = ""
        while len(uResponse.strip()) == 0:
            print("\n" + self.questions_dict[qKey]["?"])
            print("Your options are: ")
            kkeys = self.questions_dict[qKey].keys()
            for k1 in kkeys :
                if k1 != "?":
                    print(f"{k1}: {self.questions_dict[qKey][k1]}")
            print("Enter: p or P : to go to Previous Question")
            print("Enter: n or N : to go to Next Question")
            print("Enter: q or Q : to Quit the Questionnaire")
            uResponse = input("  Please enter your response(choice number): ").strip().lower()
            if len(uResponse) < 1 or not (uResponse == 'p' or uResponse == 'n' or uResponse == 'q' or uResponse in kkeys):
                print("Input cannot be empty or invalid. Please try again.")
                uResponse = ""
        return uResponse
    
# This class coordinates with all other classes and executes methods as needed for the Application.
class App:
    def __init__(self) -> None:
        clearScreen()
        print("\n ==== CSC510: Foundations of Artificial Intelligence ==== ")
        print(" ==== Portfolio Project ==== January February, 2026 ==== \n")
        print("|-----------------------------------------------------------------------------------------------|")
        print("|  This AI based application/project intended to support a doctor’s office in identifying the   |")
        print("|  diagnostic procedures and scan tests a patient may need before treatment.                    |")
        print("|-----------------------------------------------------------------------------------------------|\n")
        print("\nLoading necessary program components...")
        print("Press enter key to continue. . .")
        input()
        self.patnt: Patient = None        
        return
    
    # Creates Patient profile
    def CreatePatientProfile(self) -> Patient:
        print("\n ==== Enter Basic Information of the Patient ==== \n")
        bProceed: bool = True
        self.dict_questionaries: dict = json.load(open('data/patient_questionnaires.json'))
        strAns: str = ""
        while len(strAns) == 0 and strAns.lower() != 'q':
            strAns = input(self.dict_questionaries["NAME"]["?"] + ": ").strip()
            if strAns.lower() == 'q':
                bProceed = False

        if not bProceed:
            print("Not enough information from the Patient. Exiting the application. Goodbye!")
            return None
        pName: str = strAns

        strAns = ""
        while bProceed and len(strAns) == 0:
            print("Please enter the state code from the following options: ")
            kkeys = self.dict_questionaries["STATE"].keys()
            for k1 in kkeys :
                if k1 != "?":
                    print(f"{k1}: {self.dict_questionaries["STATE"][k1]}")
            strAns = input("\tYour selected option: ").strip()
            if strAns.lower() == 'q':
                bProceed = False
            elif strAns.lower() not in self.dict_questionaries["STATE"].keys():
                print("Invalid state. Please try again.")
                strAns = ""

        if not bProceed:
            print("Not enough information from the Patient. Exiting the application. Goodbye!")
            return None
        pAddressState: str = self.dict_questionaries["STATE"][strAns]

        strAns = ""
        bProceed = True
        while bProceed and len(strAns) == 0:
            strAns = input(self.dict_questionaries["ADDRESS"]["?"] + ": ").strip()
            if strAns.lower() == 'q':
                bProceed = False
        pAddressState = strAns + ", " + pAddressState
        strAns = ""
        while bProceed and len(strAns) == 0:
            print("Please enter your ETHINICITY from the following options: ")
            kkeys = self.dict_questionaries["ETHINICITY"].keys()
            for k1 in kkeys :
                if k1 != "?":
                    print(f"{k1}: {self.dict_questionaries["ETHINICITY"][k1]}")
            strAns = input("\tYour selected option: ").strip()
            if strAns.lower() == 'q':
                bProceed = False
            elif strAns.lower() not in self.dict_questionaries["ETHINICITY"].keys():
                print("Invalid ethnicity. Please try again.")
                strAns = ""

        if not bProceed:
            print("Not enough information from the Patient. Exiting the application. Goodbye!")
            return None
        pEthnicity: str = self.dict_questionaries["ETHINICITY"][strAns]
        
        self.patnt = Patient(pName,  pAddressState, pEthnicity)
        #print(self.patnt.patient_name) #ok
        return self.patnt

    # Collects Patient data related to the patient’s medical history and current symptoms.
    def InitializePatientForMRIQuestionnaire(self):
        if self.patnt is None:
            print("No patient profile found. Please create a patient profile first.")
            return None
        print("\n = Hello, " + self.patnt.patient_name + " ! = \n")
        print("\n ==== Welcome to the MRI Scan Eligibility Questionnaire ==== \n")
        print("Anytime, Enter: q or Q : to Quit the Application.\n")
        self.patnt.initiateQuestionnaire()
        print("\n ==== Thank you for completing the MRI Scan Eligibility Questionnaire ==== \n")        
        return
    
    # This method applies a Naive Bayes classifier to predict whether a patient needs an MRI scan based on their responses to the questionnaire.
    # It uses a custom implementation of the Naive Bayes classifier and also compares the prediction with scikit-learn's GaussianNB and BernoulliNB classifiers.
    # Input:
    #   X: list of dicts with keys: feature1, feature2, feature3, ... (training data features)
    #   y: list of class labels (training data labels)
    #   new_sample: dict with keys: feature1, feature2, feature3, ... (new patient data for prediction)
    # Output:
    #   bool: True if the patient needs an MRI scan, False otherwise
    def ApplyNaiveBayesClassifier(self, X: list, y: list, new_sample: dict) -> bool:
        nvbClassifier = nbClassifier.NaiveBayesClassifier()        
        user_input = UserInput()
        enumFeature_names, efeature_values = user_input.GetStatFeaturesEnum()
        boolFeature_names, bfeature_values = user_input.GetStatFeaturesBool()

        #nvbClassifier.SetBoolFeatures(boolFeature_names, bfeature_values)
        nvbClassifier.SetBoolFeatures(boolFeature_names)
        nvbClassifier.SetEnumFeatures(enumFeature_names, efeature_values)

        prediction_Berno = nvbClassifier.Prediction_BernoulliNB(X, y, new_sample) == True
        print("Prediction using sklearn.BernoulliNB:", prediction_Berno)
        prediction_Gauss = nvbClassifier.Prediction_GaussianNB(X, y, new_sample) == True
        print("Prediction using sklearn.GaussianNB:", prediction_Gauss)
        prediction = nvbClassifier.FindPrediction(X, y, new_sample)
        print("Prediction using my custom implementation:", prediction)
        overallPrediction = prediction | prediction_Berno | prediction_Gauss
        print("Prediction overall:", overallPrediction)
        return overallPrediction

    # This method creates a labeled dataset for training the Naive Bayes classifier.
    # By randomizing, it generates synthetic patient data with various features and labels them based on certain 
    #     conditions that determine whether an MRI scan is needed or not.
    def CreateLabledData(self, N: int) -> pd.DataFrame :
        # N: int = How Many Rows of data we need to create.
        rng = np.random.default_rng(42) 
        data = {
            "age_group": rng.integers(1, 6, size=N), 
            "gender": rng.integers(0, 2, size=N), # 0=female, 1=male
            "general_health": rng.integers(1, 6, size=N), 
            "claustrophobic": rng.integers(0, 2, size=N), 
            "metal_implants": rng.integers(0, 2, size=N), 
            "pacemaker": rng.integers(0, 2, size=N), 
            "prior_imaging": rng.integers(0, 2, size=N), 
            "contrast_allergy": rng.integers(0, 2, size=N), 
            "can_remain_still": rng.integers(0, 2, size=N), 
            "heart_coronary": rng.integers(0, 2, size=N), 
            "heart_stroke": rng.integers(0, 2, size=N), 
            "asthma": rng.integers(0, 2, size=N), 
            "cancer": rng.integers(0, 2, size=N), 
            "diabetes": rng.integers(0, 2, size=N), 
            "copd": rng.integers(0, 2, size=N), 
            "pregnant": rng.integers(0, 2, size=N), 
            "weight_category": rng.integers(1, 6, size=N),
            "recent_surgery_days_group": rng.integers(1, 6, size=N),
            "pain_level": rng.integers(1, 6, size=N), 
            "pain_location": rng.integers(1, 6, size=N),
            "anxiety_level": rng.integers(1, 6, size=N)
        } 
        df = pd.DataFrame(data)
        df["Is_MRI_SCAN_Needed"] = False        

        condition1 = (df["metal_implants"] == 1) | (df["pacemaker"] == 1) | (df["can_remain_still"] == 0) | ((df["pregnant"] == 1) & (df["gender"] == 0)) # MRI_NotNeeded 
        condition2 = ((df["heart_stroke"] == 1) | (df["heart_coronary"] == 1)) | (df["cancer"] == 1) | ((df["pain_level"] > 2) & ((df["pain_location"] == 2) | (df["pain_location"] == 3 )))
        condition3 = (df["general_health"] > 2) | (df["contrast_allergy"] == 1)  # MRI_Needed 

        df.loc[~condition1 & condition2 & condition3, "Is_MRI_SCAN_Needed"] = True
        # we can create more appropriate conditions for labeling the data based on the domain knowledge and the features we have.

        return df

    # This is for Part(b) test and CNN based classificationas as mentioned in the architecture diagram.
    def test_mribraintumorcnn(self, patientMRIImageFilePath: str):
        if patientMRIImageFilePath is None or len(patientMRIImageFilePath.strip()) < 2:
            print("Please provide valid path for patient MRI Scan image files.")
            return
        cnnbraintumor = MriBrainTumorCNN()
        cnnbraintumor.SetImgDataPaths(notumorImgDir="./data/mri/train/no_tumor", tumorImgDir="./data/mri/train/tumor")
        cnnbraintumor.PreprocessAndLoadData()
        cnnbraintumor.InitModelAndTrain(epchs=30) # default 30

        # do predictions
        print("\n === Predictions ====")        
        patientMRIImageFilePath = patientMRIImageFilePath.strip()                
        for afile in os.listdir(patientMRIImageFilePath):
            if '.jpg' in afile:
                mriScanImgFile: str = os.path.join(patientMRIImageFilePath, afile)
                yn, pcntConfidence = cnnbraintumor.predictImage(imgFile=mriScanImgFile)
                print(mriScanImgFile, " ==> " ,"Tumor: " + yn + ", Confidence: " + pcntConfidence + '%')        
        return

    # confirms that Patient MRI scan data files are uploaded in correct path.
    def mriScanFileLoaded(self) -> bool:
        root = tk.Tk()
        root.withdraw()  # hide the main window
        return messagebox.askyesno("Confirm MRI Scan files loaded", "Empty the folder: \data\mri\patient\nThen upload patient's MRI Scan files here.\n Do you want to continue?")


# The main entry point of the application. 
# It initializes the application, creates a patient profile, collects patient details through a questionnaire, 
#   generates labeled data for training a Naive Bayes classifier, and applies the classifier to predict 
#   whether the patient needs an MRI scan based on their responses.
if __name__ == "__main__":
    app: App = App()
    
    # Part (a): Patient Questionnaire, Data Collection and Naive Bayes Classifier Prediction : Starts here.
    app.CreatePatientProfile()
    #print(app.patnt.patient_name) #ok
    #print(app.patnt.patient_address) #ok
    
    app.InitializePatientForMRIQuestionnaire()
    if app.patnt is None or app.patnt.patient_details is None:
        exit(-1) # user terminated
    print("User Input:\n", app.patnt.patient_details) #ok

    # Create a new sample data for the patient medical history based on the collected patient 
    #        details from the questionnaire.
    new_sample1 = {"age_group": app.patnt.patient_details.get("age_group", 3), 
                   "gender": app.patnt.patient_details.get("gender", 0), 
                   "general_health": app.patnt.patient_details.get("general_health", 4), 
                   "claustrophobic": app.patnt.patient_details.get("claustrophobic", 0), 
                   "metal_implants": app.patnt.patient_details.get("metal_implants", 0), 
                   "pacemaker": app.patnt.patient_details.get("pacemaker", 0),
                   "asthma": app.patnt.patient_details.get("asthma", 0),  
                   "pregnant": app.patnt.patient_details.get("pregnant", 0), 
                   "recent_surgery_days_group": app.patnt.patient_details.get("recent_surgery_days_group", 3), 
                   "pain_level": app.patnt.patient_details.get("pain_level", 4), 
                   "weight_category": app.patnt.patient_details.get("weight_category", 3), 
                   "anxiety_level": app.patnt.patient_details.get("anxiety_level", 4),
                   "prior_imaging": app.patnt.patient_details.get("prior_imaging", 0), 
                   "contrast_allergy": app.patnt.patient_details.get("contrast_allergy", 0), 
                   "can_remain_still": app.patnt.patient_details.get("can_remain_still", 1), 
                   "heart_coronary": app.patnt.patient_details.get("heart_coronary", 1), 
                   "heart_stroke": app.patnt.patient_details.get("heart_stroke", 1), 
                   "cancer": app.patnt.patient_details.get("cancer", 1), 
                   "diabetes": app.patnt.patient_details.get("diabetes", 0), 
                   "copd": app.patnt.patient_details.get("copd", 0), 
                   "pain_location": app.patnt.patient_details.get("pain_location", 3)}
    
    '''
    new_sample2 = {"age_group": 3, "gender": 1, "general_health": 4, "claustrophobic": 0, "metal_implants": 0, "pacemaker": 0,
                   "asthma": 0, "pregnant": 0,"recent_surgery_days_group": 3, "pain_level": 4, "weight_category": 3, "anxiety_level": 4,
                   "prior_imaging": 0, "contrast_allergy": 0, "can_remain_still": 1, "heart_coronary": 1, "heart_stroke": 1, "cancer": 1,
                   "diabetes": 0, "copd": 0, "pain_location": 3}
    '''
    
    ddf2 = app.CreateLabledData(1000) # create a labeled dataset with 1000 rows for training the Naive Bayes classifier.

    y2 = ddf2["Is_MRI_SCAN_Needed"].tolist()
    user_input = UserInput()
    efeature_names, _ = user_input.GetStatFeaturesEnum()
    bfeature_names, _ = user_input.GetStatFeaturesBool()
    X2 = ddf2[efeature_names + bfeature_names].to_dict(orient='records')

    prediction_NaiveBayes = app.ApplyNaiveBayesClassifier(X=X2, y=y2, new_sample=new_sample1)
    # Part (a): Patient Questionnaire, Data Collection and Naive Bayes Classifier Prediction : Ends here.
    print("\n")

    # Part (b): CNN on Patient's mri scan data files : Starts here.
    if prediction_NaiveBayes == True:
        # confirm that patient mri scan files are available here
        if app.mriScanFileLoaded() == True:
            app.test_mribraintumorcnn(patientMRIImageFilePath= "./data/mri/patient")
        else:
            pass   # if MRI Scan files are not uploaded then we should not proceed.
    else:
        print("MRI Scan Not needed.")
    # Part (b): CNN on Patient's mri scan data files : Ends here.

    print("\nApplication closing. Press any key...")
    input()
    
    
# (.venv) ....>pip install numpy pandas matplotlib seaborn tensorflow 
