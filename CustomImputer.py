import numpy as np
import pandas as pd

def Impute(patient) : 
       
    is_categorical_channel = {
        'Capillary refill rate': True,
        'Diastolic blood pressure': False,
        'Fraction inspired oxygen': False,
        'Glascow coma scale eye opening': True,
        'Glascow coma scale motor response': True,
        'Glascow coma scale total': True,
        'Glascow coma scale verbal response': True,
        'Glucose': False,
        'Heart Rate': False,
        'Height': False,
        'Mean blood pressure': False,
        'Oxygen saturation': False,
        'Respiratory rate': False,
        'Systolic blood pressure': False,
        'Temperature': False,
        'Weight': False,
        'pH': False,
        'Mortality': False, 
        'Episode': False}

        
    normal_values = {
        'Capillary refill rate': '0.0',
        'Diastolic blood pressure': '59.0',
        'Fraction inspired oxygen': '0.21',
        'Glascow coma scale eye opening': 'Spontaneously',
        'Glascow coma scale motor response': 'Obeys Commands',
        'Glascow coma scale total': '15',
        'Glascow coma scale verbal response': 'Oriented',
        'Glucose': '128.0',
        'Heart Rate': '86',
        'Height': '170.0',
        'Mean blood pressure': '77.0',
        'Oxygen saturation': '98.0',
        'Respiratory rate': '19',
        'Systolic blood pressure': '118.0',
        'Temperature': '36.6',
        'Weight': '81.0',
        'pH': '7.4',
    }
    for col in patient.columns:
        if patient.loc[0, col] == '' or str(patient.loc[0, col]) == 'nan' :
            patient.loc[0, col] = normal_values[col]
        for (index, row) in enumerate(patient[col]):
            if row == '' or str(row) == 'nan':
                patient.loc[index, col] = patient[col][index-1]
    return patient
                
        