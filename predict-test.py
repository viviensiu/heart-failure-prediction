#!/usr/bin/env python
# coding: utf-8

import requests

url = 'http://localhost:9696/predict_risk'

def get_cholesterol_level(c):
    '''
    Feature engineering on cholesterol_level
    '''
    if c < 200:
        return 'Normal'
    elif c <= 239:
        return 'Borderline high'
    else:
        return 'High'
    
patient_info = {"Age": 30,
                    "Sex": "M",
                    "ChestPainType": "ASY",
                    "RestingBP": 127,
                    "Cholesterol": 239,
                    "FastingBS": 1,
                    "RestingECG": "Normal",
                    "MaxHR": 99,
                    "ExerciseAngina": "N",
                    "Oldpeak": 1.0,
                    "ST_Slope": "Up",
                    "Cholesterol_Level": get_cholesterol_level(239)
                    }


prediction = requests.post(url, json=patient_info).json()
print(prediction)

# if prediction['hasHeartDisease'] == True
if prediction['hasHeartDisease']:
    print("Potentially at risk of heart disease. Follow-up examination recommended.")
else:
    print("Negative.")