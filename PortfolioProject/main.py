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
I chose to develop an AI‑based project intended to support a doctor’s office in 
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

import os
import math
import json
import numpy as np 
import pandas as pd



def clearScreen():    
    if os.name == 'nt':  # For windows system
        _ = os.system('cls')    
    else:             # for non-windows system
        _ = os.system('clear')
    return

class Patient:
    def __init__(self, pName:str, pAddress:str) -> None:
        self.patient_name = pName
        self.patient_address = pAddress
        self.patient_details = {}
        return

    def initiateQuestionnaire(self) -> None:
        self.patient_details = self.RunQuestionnaire()
        return
    
    def RunQuestionnaire(self) -> dict:
        try:
            print("Loading initial questionnaires. . .")            
            question_variables = ['_NAME', '_STATE', '_ADDRESS','ETHINICITY','GENDER', 'AGE_GROUP', 'GENERAL_HEALTH', 'CLAUSTROPHOBIC',
                                  'METAL_IMPLANTS', 'PACEMAKER', 'PRIOR_IMAGING', 'CONTRAST_ALLERGY', 'CAN_REMAIN_STILL',
                                  'HEART_CORONARY', 'HEART_STROKE', 'ASTHMA', 'CANCER','DIABETES', 'COPD', 'PREGNANT', 
                                  'WEIGHT_CATEGORY', 'RECENT_SURGERY_DAYS', 'PAIN_LEVEL', 'PAIN_LOCATION_SCORE','ANXIETY_LEVEL']
        except:
            print("ERROR: Unable to load initial questionnaires for the patient. Exiting program.")
            exit(0)
        user_input = UserInput()
        patientDetails = {}
        patientDetails["name"] = self.patient_name
        patientDetails["address"] = self.patient_address
        bProceed: bool = True
        qIndex: int = 3
        while bProceed and qIndex < len(question_variables):            
            userAns = user_input.GetUserResponse(question_variables[qIndex]).strip().lower()
            if userAns == 'p':  # go to previous question
                if qIndex > 3:
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
                        if question_variables[qIndex] == 'AGE_GROUP':
                            if uAnsInt in [1,2,3,4,5,6]:
                                patientDetails["age"] = 18 + 10*(uAnsInt -1)
                                if userAns == '6':
                                    patientDetails["age"] = 80
                        elif question_variables[qIndex] == 'RECENT_SURGERY_DAYS':
                            if uAnsInt == 2:
                                patientDetails["recent_surgery_days"] = 7
                            elif uAnsInt == 3:
                                patientDetails["recent_surgery_days"] = 15
                            elif uAnsInt == 4:
                                patientDetails["recent_surgery_days"] = 30
                            elif uAnsInt == 5:
                                patientDetails["recent_surgery_days"] = 90
                            elif uAnsInt == 6:
                                patientDetails["recent_surgery_days"] = 180
                        elif question_variables[qIndex] == 'ETHINICITY':
                            patientDetails["ethnicity"] = question_variables[qIndex][userAns]
                        elif question_variables[qIndex] in ['GENERAL_HEALTH','PAIN_LEVEL','PAIN_LOCATION_SCORE','ANXIETY_LEVEL','WEIGHT_CATEGORY']:
                            patientDetails[question_variables[qIndex].lower()] = uAnsInt
                        elif (uAnsInt > -1 and uAnsInt < 2) and question_variables[qIndex] in ['CLAUSTROPHOBIC','METAL_IMPLANTS','PACEMAKER','PRIOR_IMAGING',
                                                                                               'CONTRAST_ALLERGY','CAN_REMAIN_STILL','HEART_CORONARY','HEART_STROKE',
                                                                                               'ASTHMA','CANCER','DIABETES','COPD','PREGNANT']:
                            patientDetails[question_variables[qIndex].lower()] = uAnsInt
                except:
                        pass #patientDetails[question_variables[qIndex]] = -1
                qIndex += 1
        return patientDetails

class UserInput():
    def __init__(self) -> None:        
        self.questions_dict = json.load(open('Data/patient_questionnaires.json'))
        return
    
    def GetUserResponse(self, qKey:str) -> str:
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

class App():
    def __init__(self) -> None:
        clearScreen()
        print("\n ==== CSC510: Foundations of Artificial Intelligence ==== ")
        print(" ==== CSC510: Portfolio Project ==== ")
        print(" ==== This AI‑based application/project intended to support a doctor’s office in identifying the " \
                    "diagnostic procedures and scan tests a patient may need before treatment.  ==== \n")
        print("\nLoading necessary program components...")
        #print("Press enter key to continue. . .")
        #input()
        #self.CreatePreprocessData(500)
        self.patnt: Patient = None        
        return
    
    def CreatePatientProfile(self):
        print("\n ==== Enter Basic Information of the Patient ==== \n")
        bProceed: bool = True
        self.dict_questionaries: dict = json.load(open('Data/patient_questionnaires.json'))
        strAns: str = ""
        while len(strAns) == 0 and strAns.lower() != 'q':
            strAns = input(self.dict_questionaries["_NAME"]["?"] + ": ").strip()
            if strAns.lower() == 'q':
                bProceed = False

        if not bProceed:
            print("Not enough information from the Patient. Exiting the application. Goodbye!")
            return -1;
        pName: str = strAns

        strAns = ""
        while bProceed and len(strAns) == 0:
            print("Please enter the state code from the following options: ")
            kkeys = self.dict_questionaries["_STATE"].keys()
            for k1 in kkeys :
                if k1 != "?":
                    print(f"{k1}: {self.dict_questionaries["_STATE"][k1]}")
            strAns = input("\tYour selected option: ").strip()
            if strAns.lower() == 'q':
                bProceed = False
            elif strAns.lower() not in self.dict_questionaries["_STATE"].keys():
                print("Invalid state. Please try again.")
                strAns = ""

        if not bProceed:
            print("Not enough information from the Patient. Exiting the application. Goodbye!")
            return -1;
        pAddressState: str = self.dict_questionaries["_STATE"][strAns]

        strAns = ""
        bProceed = True
        while bProceed and len(strAns) == 0:
            strAns = input(self.dict_questionaries["_ADDRESS"]["?"] + ": ").strip()
            if strAns.lower() == 'q':
                bProceed = False

        if not bProceed:
            print("Not enough information from the Patient. Exiting the application. Goodbye!")
            return -1;
        
        pAddressState = strAns + ", " + pAddressState        
        self.patnt = Patient(pName,  pAddressState)
        #print(self.patnt.patient_name) #ok
        return self.patnt

    def InitializePatientForMRIQuestionnaire(self):
        if self.patnt is None:
            print("No patient profile found. Please create a patient profile first.")
            return -1
        print("\n ==== Welcome to the MRI Scan Eligibility Questionnaire ==== \n")
        print("Anytime, Enter: q or Q : to Quit the Application.\n")
        self.patnt.initiateQuestionnaire()
        print("\n ==== Thank you for completing the MRI Scan Eligibility Questionnaire ==== \n")        
        return
    
    def CreatePreprocessData(self, N:int):
        # number of rows 
        #N: int = HowManyRows # 5000 
        rng = np.random.default_rng(42) 
        data = { 
            "age": rng.integers(18, 90, size=N), 
            "sex": rng.integers(0, 2, size=N), # 0=female, 1=male 
            "claustrophobic": rng.integers(0, 2, size=N), 
            "metal_implants": rng.integers(0, 2, size=N), 
            "pacemaker": rng.integers(0, 2, size=N), 
            "recent_surgery_days": rng.integers(0, 60, size=N), 
            "pain_level": rng.integers(1, 11, size=N), 
            "pain_location_score": rng.integers(0, 6, size=N), 
            "prior_imaging": rng.integers(0, 2, size=N), 
            "contrast_allergy": rng.integers(0, 2, size=N), 
            "can_remain_still": rng.integers(0, 2, size=N), 
            "anxiety_level": rng.integers(1, 11, size=N), 
            "general_health": rng.integers(1, 6, size=N), 
            "heart_coronary": rng.integers(0, 2, size=N), 
            "heart_stroke": rng.integers(0, 2, size=N), 
            "asthma": rng.integers(0, 2, size=N), 
            "cancer": rng.integers(0, 2, size=N), 
            "diabetes": rng.integers(0, 2, size=N), 
            "copd": rng.integers(0, 2, size=N), 
            "pregnant": rng.integers(0, 2, size=N), 
            "weight_category": rng.integers(1, 5, size=N), 
        } 

        df = pd.DataFrame(data) 

        # simple synthetic rule for labels (you can change this) 
        """ df["label"] = np.where( 
            (df["pain_level"].between(1,3)) | 
            (df["recent_surgery_days"] < 10) | 
            (df["anxiety_level"] > 7), 
            "MRI1_Needed", 
            "MRI_NotNeeded" ) 
        
        df["label"] = np.where( 
            (df["pain_level"].between(3,6)) | 
            (df["recent_surgery_days"] < 10) | 
            (df["anxiety_level"] > 7), 
            "MRI2_Needed",
            "MRI_NotNeeded" ) """ 
        
        df["label"] = np.where( 
            (df["pain_level"] > 7) &            # 1-10
            (df["recent_surgery_days"] < 10) |  # 0-59
            (df["anxiety_level"] > 7) |         # 1-10
            (df["can_remain_still"] > 1),       # 0-1
            "MRI_Needed",
            "MRI_NotNeeded" ) 

        print(df[df["label"] == "MRI_Needed"].shape)
        print(df[df["label"] == "MRI_Needed"])
        #print(df.head()) 
        print(df.shape) 
        return



if __name__ == "__main__":
    clearScreen()
    app: App = App()
    app.CreatePatientProfile()
    #print(app.patnt.patient_name) #ok
    #print(app.patnt.patient_address) #ok
    app.InitializePatientForMRIQuestionnaire()
    print(app.patnt.patient_details) #ok
