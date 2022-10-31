#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
from pandas.api.types import CategoricalDtype

from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import KNNImputer

from matplotlib import pyplot as plt


# In[ ]:


#Open csv file.

data = pd.read_csv("../input/st-combined/st_combined_cpt_only.csv", index_col=0, na_values = -99)
data.head()


# In[ ]:


#See all columns.

print(list(data.columns))


# In[ ]:


#Check data shape.

data.shape


# In[ ]:


#Define variables of interest (predictor variables, inclusion/exclusion criteria, outcomes of interest).

variables = ['SEX', 'RACE_NEW', 'ETHNICITY_HISPANIC', 'CPT', 'INOUT', 'TRANST', 'AGE', 'OPERYR', 'DISCHDEST', 'ANESTHES', 'SURGSPEC', 'ELECTSURG', 'HEIGHT', 'WEIGHT', 'DIABETES', 'SMOKE', 'DYSPNEA', 'FNSTATUS2', 'VENTILAT', 'HXCOPD', 'ASCITES', 'HXCHF', 'HYPERMED', 'RENAFAIL', 'DIALYSIS', 'DISCANCR', 'WNDINF', 'STEROID', 'WTLOSS', 'BLEEDDIS', 'TRANSFUS', 'PRSEPIS', 'PRSODM', 'PRBUN', 'PRCREAT', 'PRALBUM', 'PRBILI', 'PRSGOT', 'PRALKPH', 'PRWBC', 'PRHCT', 'PRPLATE', 'PRPTT', 'PRINR', 'PRPT', 'OTHERCPT1', 'OTHERCPT2', 'OTHERCPT3', 'OTHERCPT4', 'OTHERCPT5', 'OTHERCPT6', 'OTHERCPT7', 'OTHERCPT8', 'OTHERCPT9', 'OTHERCPT10', 'CONCPT1', 'CONCPT2', 'CONCPT3', 'CONCPT4', 'CONCPT5', 'CONCPT6', 'CONCPT7', 'CONCPT8', 'CONCPT9', 'CONCPT10', 'EMERGNCY', 'WNDCLAS', 'ASACLAS', 'OPTIME', 'TOTHLOS', 'HTOODAY', 'NSUPINFEC', 'NWNDINFD', 'NORGSPCSSI', 'NDEHIS', 'NOUPNEUMO', 'NREINTUB', 'NPULEMBOL', 'NFAILWEAN', 'NRENAINSF', 'NOPRENAFL', 'NURNINFEC', 'NCNSCVA', 'NCDARREST', 'NCDMI', 'NOTHBLEED', 'NOTHDVT', 'NOTHSYSEP', 'NOTHSESHOCK', 'PODIAG', 'PODIAG10', 'STILLINHOSP', 'REOPERATION1', 'READMISSION1']


# In[ ]:


#Remove unwanted columns and check data shape.

data = data[variables]

data.shape


# In[ ]:


#Check data for elective surgeries.

data['ELECTSURG'].value_counts(normalize=False, dropna=False)


# In[ ]:


#Apply inclusion criteria for elective surgeries.

data = data[(data['ELECTSURG'] == 'Yes')]

data['ELECTSURG'].value_counts(normalize=False, dropna=False)


# In[ ]:


#Check data for inpatient vs. outpatient procedures.

data['INOUT'].value_counts(normalize=False, dropna=False)


# In[ ]:


#Apply inclusion criteria for inpatient procedures.

data = data[(data['INOUT'] == 'Inpatient')]

data['INOUT'].value_counts(normalize=False, dropna=False)


# In[ ]:


#Check data for anesthesia type.

data['ANESTHES'].value_counts(normalize=False, dropna=False)


# In[ ]:


#Apply inclusion criteria for general anesthesia.

data = data[(data['ANESTHES'] == 'General')]

data['ANESTHES'].value_counts(normalize=False, dropna=False)


# In[ ]:


#Check data for surgical specialties.

data['SURGSPEC'].value_counts(normalize=False, dropna=False)


# In[ ]:


#Apply inclusion criteria for surgical specialties.

data = data[(data['SURGSPEC'] == 'Neurosurgery') | (data['SURGSPEC'] == 'Orthopedics')]

data['SURGSPEC'].value_counts(normalize=False, dropna=False)


# In[ ]:


#Check data for emergency surgery.

data['EMERGNCY'].value_counts(normalize=False, dropna=False)


# In[ ]:


#Apply exclusion criteria for emergency surgery.

data = data[(data['EMERGNCY'] == 'No')]

data['EMERGNCY'].value_counts(normalize=False, dropna=False)


# In[ ]:


#Check data for ventilator dependency.

data['VENTILAT'].value_counts(normalize=False, dropna=False)


# In[ ]:


#Apply exclusion criteria for ventilator dependency.

data = data[(data['VENTILAT'] == 'No')]

data['VENTILAT'].value_counts(normalize=False, dropna=False)


# In[ ]:


#Check data for wound class.

data['WNDCLAS'].value_counts(normalize=False, dropna=False)


# In[ ]:


#Apply exclusion criteria for wound class.

data = data[(data['WNDCLAS'] == '1-Clean')]

data['WNDCLAS'].value_counts(normalize=False, dropna=False)


# In[ ]:


#Check data for preoperative sepsis.

data['PRSEPIS'].value_counts(normalize=False, dropna=False)


# In[ ]:


#Apply exclusion criteria for preoperative sepsis.

data = data[(data['PRSEPIS'] == 'None')]

data['PRSEPIS'].value_counts(normalize=False, dropna=False)


# In[ ]:


#Check data for ASA Class.

data['ASACLAS'].value_counts(normalize=False, dropna=False)


# In[ ]:


#Apply exclusion criteria for ASA class.

data = data[(data['ASACLAS'] != '4-Life Threat') & (data['ASACLAS'] != '5-Moribund') & (data['ASACLAS'] != 'None assigned')]

data['ASACLAS'].value_counts(normalize=False, dropna=False)


# In[ ]:


#Check data for patients still in hospital after 30 days postoperatively.

data['STILLINHOSP'].value_counts(normalize=False, dropna=False)


# In[ ]:


#Apply exclusion criteria for patients still in hospital after 30 days postoperatively.

data = data[(data['STILLINHOSP'] == 'No')]

data['STILLINHOSP'].value_counts(normalize=False, dropna=False)


# In[ ]:


#Create BMI column.

lbs_to_kg_ratio = 0.453592
inch_to_meter_ratio = 0.0254

data['HEIGHT'] *= inch_to_meter_ratio
data['WEIGHT'] *= lbs_to_kg_ratio

data['BMI'] = data['WEIGHT']/(data['HEIGHT']**2)
print(min(data['BMI']))
print(max(data['BMI']))


# In[ ]:


#Check data for race.

data['RACE_NEW'].value_counts(normalize=False, dropna=False)


# In[ ]:


#Check data for ethnicity.

data['ETHNICITY_HISPANIC'].value_counts(normalize=False, dropna=False)


# In[ ]:


#Simplify race and ethnicity columns.

data.loc[data['RACE_NEW'] == 'White', 'RACE'] = 'White'
data.loc[data['RACE_NEW'] == 'Black or African American', 'RACE'] = 'Black or African American'
data.loc[data['RACE_NEW'] == 'Asian', 'RACE'] = 'Asian'
data.loc[data['RACE_NEW'] == 'Native Hawaiian or Pacific Islander', 'RACE'] = 'Other'
data.loc[data['RACE_NEW'] == 'American Indian or Alaska Native', 'RACE'] = 'Other'
data.loc[data['RACE_NEW'] == 'Native Hawaiian or Other Pacific Islander', 'RACE'] = 'Other'

data.loc[data['ETHNICITY_HISPANIC'] == 'Yes', 'RACE'] = 'Hispanic'

data['RACE'].value_counts(normalize=False, dropna=False)


# In[ ]:


#Check data for transfer status.

data['TRANST'].value_counts(normalize=False, dropna=False)


# In[ ]:


#Simplify transfer status column.

data.loc[data['TRANST'] == 'Not transferred (admitted from home)', 'TRANST'] = 'Not transferred'
data.loc[data['TRANST'] == 'From acute care hospital inpatient', 'TRANST'] = 'Transferred'
data.loc[data['TRANST'] == 'Outside emergency department', 'TRANST'] = 'Transferred'
data.loc[data['TRANST'] == 'Nursing home - Chronic care - Intermediate care', 'TRANST'] = 'Transferred'
data.loc[data['TRANST'] == 'Transfer from other', 'TRANST'] = 'Transferred'

data['TRANST'].value_counts(normalize=False, dropna=False)


# In[ ]:


#Check data for dyspnea.

data['DYSPNEA'].value_counts(normalize=False, dropna=False)


# In[ ]:


#Simplify dyspnea column.

data.loc[data['DYSPNEA'] == 'No', 'DYSPNEA'] = 'No'
data.loc[data['DYSPNEA'] == 'MODERATE EXERTION', 'DYSPNEA'] = 'Yes'
data.loc[data['DYSPNEA'] == 'AT REST', 'DYSPNEA'] = 'Yes'

data['DYSPNEA'].value_counts(normalize=False, dropna=False)


# In[ ]:


#Check data for diabetes status.

data['DIABETES'].value_counts(normalize=False, dropna=False)


# In[ ]:


#Simplify diabetes column.

data.loc[data['DIABETES'] == 'NO', 'DIABETES'] = 'No'
data.loc[data['DIABETES'] == 'NON-INSULIN', 'DIABETES'] = 'Yes'
data.loc[data['DIABETES'] == 'INSULIN', 'DIABETES'] = 'Yes'

data['DIABETES'].value_counts(normalize=False, dropna=False)


# In[ ]:


#Cast ASA class as ordered categorical.

cat_type1 = CategoricalDtype(categories=['1-No Disturb','2-Mild Disturb','3-Severe Disturb'], ordered=True)
data['ASACLAS'].astype(cat_type1)


# In[ ]:


#Cast functional status as ordered categorical.

cat_type2 = CategoricalDtype(categories=['Unknown','Independent','Partiallly Dependent', 'Totally Dependent'], ordered=True)
data['FNSTATUS2'].astype(cat_type2)


# In[ ]:


#Convert 90+ to 91 and AGE column to integer.

data.loc[data['AGE'] == '90+', 'AGE'] = 91
pd.to_numeric(data['AGE'], downcast='integer')


# In[ ]:


#Show patients for each CPT code.

data['CPT'].value_counts()


# In[ ]:


#Classify tumors into extradural vs. intradural and create a column named 'IEDUR' for it.

data.loc[data['CPT'] == 63275, 'IEDUR'] = 'Extradural'
data.loc[data['CPT'] == 63276, 'IEDUR'] = 'Extradural'
data.loc[data['CPT'] == 63277, 'IEDUR'] = 'Extradural'
data.loc[data['CPT'] == 63278, 'IEDUR'] = 'Extradural'

data.loc[data['CPT'] == 63280, 'IEDUR'] = 'Intradural'
data.loc[data['CPT'] == 63281, 'IEDUR'] = 'Intradural'
data.loc[data['CPT'] == 63282, 'IEDUR'] = 'Intradural'
data.loc[data['CPT'] == 63283, 'IEDUR'] = 'Intradural'

data.loc[data['CPT'] == 63285, 'IEDUR'] = 'Intradural'
data.loc[data['CPT'] == 63286, 'IEDUR'] = 'Intradural'
data.loc[data['CPT'] == 63287, 'IEDUR'] = 'Intradural'
data.loc[data['CPT'] == 63290, 'IEDUR'] = 'Intradural'

data.loc[data['CPT'] == 63300, 'IEDUR'] = 'Extradural'
data.loc[data['CPT'] == 63301, 'IEDUR'] = 'Extradural'
data.loc[data['CPT'] == 63302, 'IEDUR'] = 'Extradural'
data.loc[data['CPT'] == 63303, 'IEDUR'] = 'Extradural'

data.loc[data['CPT'] == 63304, 'IEDUR'] = 'Intradural'
data.loc[data['CPT'] == 63305, 'IEDUR'] = 'Intradural'
data.loc[data['CPT'] == 63306, 'IEDUR'] = 'Intradural'
data.loc[data['CPT'] == 63307, 'IEDUR'] = 'Intradural'

data.loc[data['CONCPT1'] == 63275, 'IEDUR'] = 'Extradural'
data.loc[data['CONCPT1'] == 63276, 'IEDUR'] = 'Extradural'
data.loc[data['CONCPT1'] == 63277, 'IEDUR'] = 'Extradural'
data.loc[data['CONCPT1'] == 63278, 'IEDUR'] = 'Extradural'

data.loc[data['CONCPT1'] == 63280, 'IEDUR'] = 'Intradural'
data.loc[data['CONCPT1'] == 63281, 'IEDUR'] = 'Intradural'
data.loc[data['CONCPT1'] == 63282, 'IEDUR'] = 'Intradural'
data.loc[data['CONCPT1'] == 63283, 'IEDUR'] = 'Intradural'

data.loc[data['CONCPT1'] == 63285, 'IEDUR'] = 'Intradural'
data.loc[data['CONCPT1'] == 63286, 'IEDUR'] = 'Intradural'
data.loc[data['CONCPT1'] == 63287, 'IEDUR'] = 'Intradural'
data.loc[data['CONCPT1'] == 63290, 'IEDUR'] = 'Intradural'

data.loc[data['CONCPT1'] == 63300, 'IEDUR'] = 'Extradural'
data.loc[data['CONCPT1'] == 63301, 'IEDUR'] = 'Extradural'
data.loc[data['CONCPT1'] == 63302, 'IEDUR'] = 'Extradural'
data.loc[data['CONCPT1'] == 63303, 'IEDUR'] = 'Extradural'

data.loc[data['CONCPT1'] == 63304, 'IEDUR'] = 'Intradural'
data.loc[data['CONCPT1'] == 63305, 'IEDUR'] = 'Intradural'
data.loc[data['CONCPT1'] == 63306, 'IEDUR'] = 'Intradural'
data.loc[data['CONCPT1'] == 63307, 'IEDUR'] = 'Intradural'

data.loc[data['CONCPT2'] == 63275, 'IEDUR'] = 'Extradural'
data.loc[data['CONCPT2'] == 63276, 'IEDUR'] = 'Extradural'
data.loc[data['CONCPT2'] == 63277, 'IEDUR'] = 'Extradural'
data.loc[data['CONCPT2'] == 63278, 'IEDUR'] = 'Extradural'

data.loc[data['CONCPT2'] == 63280, 'IEDUR'] = 'Intradural'
data.loc[data['CONCPT2'] == 63281, 'IEDUR'] = 'Intradural'
data.loc[data['CONCPT2'] == 63282, 'IEDUR'] = 'Intradural'
data.loc[data['CONCPT2'] == 63283, 'IEDUR'] = 'Intradural'

data.loc[data['CONCPT2'] == 63285, 'IEDUR'] = 'Intradural'
data.loc[data['CONCPT2'] == 63286, 'IEDUR'] = 'Intradural'
data.loc[data['CONCPT2'] == 63287, 'IEDUR'] = 'Intradural'
data.loc[data['CONCPT2'] == 63290, 'IEDUR'] = 'Intradural'

data.loc[data['CONCPT2'] == 63300, 'IEDUR'] = 'Extradural'
data.loc[data['CONCPT2'] == 63301, 'IEDUR'] = 'Extradural'
data.loc[data['CONCPT2'] == 63302, 'IEDUR'] = 'Extradural'
data.loc[data['CONCPT2'] == 63303, 'IEDUR'] = 'Extradural'

data.loc[data['CONCPT2'] == 63304, 'IEDUR'] = 'Intradural'
data.loc[data['CONCPT2'] == 63305, 'IEDUR'] = 'Intradural'
data.loc[data['CONCPT2'] == 63306, 'IEDUR'] = 'Intradural'
data.loc[data['CONCPT2'] == 63307, 'IEDUR'] = 'Intradural'

data.loc[data['CONCPT3'] == 63275, 'IEDUR'] = 'Extradural'
data.loc[data['CONCPT3'] == 63276, 'IEDUR'] = 'Extradural'
data.loc[data['CONCPT3'] == 63277, 'IEDUR'] = 'Extradural'
data.loc[data['CONCPT3'] == 63278, 'IEDUR'] = 'Extradural'

data.loc[data['CONCPT3'] == 63280, 'IEDUR'] = 'Intradural'
data.loc[data['CONCPT3'] == 63281, 'IEDUR'] = 'Intradural'
data.loc[data['CONCPT3'] == 63282, 'IEDUR'] = 'Intradural'
data.loc[data['CONCPT3'] == 63283, 'IEDUR'] = 'Intradural'

data.loc[data['CONCPT3'] == 63285, 'IEDUR'] = 'Intradural'
data.loc[data['CONCPT3'] == 63286, 'IEDUR'] = 'Intradural'
data.loc[data['CONCPT3'] == 63287, 'IEDUR'] = 'Intradural'
data.loc[data['CONCPT3'] == 63290, 'IEDUR'] = 'Intradural'

data.loc[data['CONCPT3'] == 63300, 'IEDUR'] = 'Extradural'
data.loc[data['CONCPT3'] == 63301, 'IEDUR'] = 'Extradural'
data.loc[data['CONCPT3'] == 63302, 'IEDUR'] = 'Extradural'
data.loc[data['CONCPT3'] == 63303, 'IEDUR'] = 'Extradural'

data.loc[data['CONCPT3'] == 63304, 'IEDUR'] = 'Intradural'
data.loc[data['CONCPT3'] == 63305, 'IEDUR'] = 'Intradural'
data.loc[data['CONCPT3'] == 63306, 'IEDUR'] = 'Intradural'
data.loc[data['CONCPT3'] == 63307, 'IEDUR'] = 'Intradural'

data.loc[data['CONCPT4'] == 63275, 'IEDUR'] = 'Extradural'
data.loc[data['CONCPT4'] == 63276, 'IEDUR'] = 'Extradural'
data.loc[data['CONCPT4'] == 63277, 'IEDUR'] = 'Extradural'
data.loc[data['CONCPT4'] == 63278, 'IEDUR'] = 'Extradural'

data.loc[data['CONCPT4'] == 63280, 'IEDUR'] = 'Intradural'
data.loc[data['CONCPT4'] == 63281, 'IEDUR'] = 'Intradural'
data.loc[data['CONCPT4'] == 63282, 'IEDUR'] = 'Intradural'
data.loc[data['CONCPT4'] == 63283, 'IEDUR'] = 'Intradural'

data.loc[data['CONCPT4'] == 63285, 'IEDUR'] = 'Intradural'
data.loc[data['CONCPT4'] == 63286, 'IEDUR'] = 'Intradural'
data.loc[data['CONCPT4'] == 63287, 'IEDUR'] = 'Intradural'
data.loc[data['CONCPT4'] == 63290, 'IEDUR'] = 'Intradural'

data.loc[data['CONCPT4'] == 63300, 'IEDUR'] = 'Extradural'
data.loc[data['CONCPT4'] == 63301, 'IEDUR'] = 'Extradural'
data.loc[data['CONCPT4'] == 63302, 'IEDUR'] = 'Extradural'
data.loc[data['CONCPT4'] == 63303, 'IEDUR'] = 'Extradural'

data.loc[data['CONCPT4'] == 63304, 'IEDUR'] = 'Intradural'
data.loc[data['CONCPT4'] == 63305, 'IEDUR'] = 'Intradural'
data.loc[data['CONCPT4'] == 63306, 'IEDUR'] = 'Intradural'
data.loc[data['CONCPT4'] == 63307, 'IEDUR'] = 'Intradural'

data.loc[data['CONCPT5'] == 63275, 'IEDUR'] = 'Extradural'
data.loc[data['CONCPT5'] == 63276, 'IEDUR'] = 'Extradural'
data.loc[data['CONCPT5'] == 63277, 'IEDUR'] = 'Extradural'
data.loc[data['CONCPT5'] == 63278, 'IEDUR'] = 'Extradural'

data.loc[data['CONCPT5'] == 63280, 'IEDUR'] = 'Intradural'
data.loc[data['CONCPT5'] == 63281, 'IEDUR'] = 'Intradural'
data.loc[data['CONCPT5'] == 63282, 'IEDUR'] = 'Intradural'
data.loc[data['CONCPT5'] == 63283, 'IEDUR'] = 'Intradural'

data.loc[data['CONCPT5'] == 63285, 'IEDUR'] = 'Intradural'
data.loc[data['CONCPT5'] == 63286, 'IEDUR'] = 'Intradural'
data.loc[data['CONCPT5'] == 63287, 'IEDUR'] = 'Intradural'
data.loc[data['CONCPT5'] == 63290, 'IEDUR'] = 'Intradural'

data.loc[data['CONCPT5'] == 63300, 'IEDUR'] = 'Extradural'
data.loc[data['CONCPT5'] == 63301, 'IEDUR'] = 'Extradural'
data.loc[data['CONCPT5'] == 63302, 'IEDUR'] = 'Extradural'
data.loc[data['CONCPT5'] == 63303, 'IEDUR'] = 'Extradural'

data.loc[data['CONCPT5'] == 63304, 'IEDUR'] = 'Intradural'
data.loc[data['CONCPT5'] == 63305, 'IEDUR'] = 'Intradural'
data.loc[data['CONCPT5'] == 63306, 'IEDUR'] = 'Intradural'
data.loc[data['CONCPT5'] == 63307, 'IEDUR'] = 'Intradural'

data.loc[data['CONCPT6'] == 63275, 'IEDUR'] = 'Extradural'
data.loc[data['CONCPT6'] == 63276, 'IEDUR'] = 'Extradural'
data.loc[data['CONCPT6'] == 63277, 'IEDUR'] = 'Extradural'
data.loc[data['CONCPT6'] == 63278, 'IEDUR'] = 'Extradural'

data.loc[data['CONCPT6'] == 63280, 'IEDUR'] = 'Intradural'
data.loc[data['CONCPT6'] == 63281, 'IEDUR'] = 'Intradural'
data.loc[data['CONCPT6'] == 63282, 'IEDUR'] = 'Intradural'
data.loc[data['CONCPT6'] == 63283, 'IEDUR'] = 'Intradural'

data.loc[data['CONCPT6'] == 63285, 'IEDUR'] = 'Intradural'
data.loc[data['CONCPT6'] == 63286, 'IEDUR'] = 'Intradural'
data.loc[data['CONCPT6'] == 63287, 'IEDUR'] = 'Intradural'
data.loc[data['CONCPT6'] == 63290, 'IEDUR'] = 'Intradural'

data.loc[data['CONCPT6'] == 63300, 'IEDUR'] = 'Extradural'
data.loc[data['CONCPT6'] == 63301, 'IEDUR'] = 'Extradural'
data.loc[data['CONCPT6'] == 63302, 'IEDUR'] = 'Extradural'
data.loc[data['CONCPT6'] == 63303, 'IEDUR'] = 'Extradural'

data.loc[data['CONCPT6'] == 63304, 'IEDUR'] = 'Intradural'
data.loc[data['CONCPT6'] == 63305, 'IEDUR'] = 'Intradural'
data.loc[data['CONCPT6'] == 63306, 'IEDUR'] = 'Intradural'
data.loc[data['CONCPT6'] == 63307, 'IEDUR'] = 'Intradural'

data.loc[data['CONCPT7'] == 63275, 'IEDUR'] = 'Extradural'
data.loc[data['CONCPT7'] == 63276, 'IEDUR'] = 'Extradural'
data.loc[data['CONCPT7'] == 63277, 'IEDUR'] = 'Extradural'
data.loc[data['CONCPT7'] == 63278, 'IEDUR'] = 'Extradural'

data.loc[data['CONCPT7'] == 63280, 'IEDUR'] = 'Intradural'
data.loc[data['CONCPT7'] == 63281, 'IEDUR'] = 'Intradural'
data.loc[data['CONCPT7'] == 63282, 'IEDUR'] = 'Intradural'
data.loc[data['CONCPT7'] == 63283, 'IEDUR'] = 'Intradural'

data.loc[data['CONCPT7'] == 63285, 'IEDUR'] = 'Intradural'
data.loc[data['CONCPT7'] == 63286, 'IEDUR'] = 'Intradural'
data.loc[data['CONCPT7'] == 63287, 'IEDUR'] = 'Intradural'
data.loc[data['CONCPT7'] == 63290, 'IEDUR'] = 'Intradural'

data.loc[data['CONCPT7'] == 63300, 'IEDUR'] = 'Extradural'
data.loc[data['CONCPT7'] == 63301, 'IEDUR'] = 'Extradural'
data.loc[data['CONCPT7'] == 63302, 'IEDUR'] = 'Extradural'
data.loc[data['CONCPT7'] == 63303, 'IEDUR'] = 'Extradural'

data.loc[data['CONCPT7'] == 63304, 'IEDUR'] = 'Intradural'
data.loc[data['CONCPT7'] == 63305, 'IEDUR'] = 'Intradural'
data.loc[data['CONCPT7'] == 63306, 'IEDUR'] = 'Intradural'
data.loc[data['CONCPT7'] == 63307, 'IEDUR'] = 'Intradural'

data.loc[data['CONCPT8'] == 63275, 'IEDUR'] = 'Extradural'
data.loc[data['CONCPT8'] == 63276, 'IEDUR'] = 'Extradural'
data.loc[data['CONCPT8'] == 63277, 'IEDUR'] = 'Extradural'
data.loc[data['CONCPT8'] == 63278, 'IEDUR'] = 'Extradural'

data.loc[data['CONCPT8'] == 63280, 'IEDUR'] = 'Intradural'
data.loc[data['CONCPT8'] == 63281, 'IEDUR'] = 'Intradural'
data.loc[data['CONCPT8'] == 63282, 'IEDUR'] = 'Intradural'
data.loc[data['CONCPT8'] == 63283, 'IEDUR'] = 'Intradural'

data.loc[data['CONCPT8'] == 63285, 'IEDUR'] = 'Intradural'
data.loc[data['CONCPT8'] == 63286, 'IEDUR'] = 'Intradural'
data.loc[data['CONCPT8'] == 63287, 'IEDUR'] = 'Intradural'
data.loc[data['CONCPT8'] == 63290, 'IEDUR'] = 'Intradural'

data.loc[data['CONCPT8'] == 63300, 'IEDUR'] = 'Extradural'
data.loc[data['CONCPT8'] == 63301, 'IEDUR'] = 'Extradural'
data.loc[data['CONCPT8'] == 63302, 'IEDUR'] = 'Extradural'
data.loc[data['CONCPT8'] == 63303, 'IEDUR'] = 'Extradural'

data.loc[data['CONCPT8'] == 63304, 'IEDUR'] = 'Intradural'
data.loc[data['CONCPT8'] == 63305, 'IEDUR'] = 'Intradural'
data.loc[data['CONCPT8'] == 63306, 'IEDUR'] = 'Intradural'
data.loc[data['CONCPT8'] == 63307, 'IEDUR'] = 'Intradural'

data.loc[data['CONCPT9'] == 63275, 'IEDUR'] = 'Extradural'
data.loc[data['CONCPT9'] == 63276, 'IEDUR'] = 'Extradural'
data.loc[data['CONCPT9'] == 63277, 'IEDUR'] = 'Extradural'
data.loc[data['CONCPT9'] == 63278, 'IEDUR'] = 'Extradural'

data.loc[data['CONCPT9'] == 63280, 'IEDUR'] = 'Intradural'
data.loc[data['CONCPT9'] == 63281, 'IEDUR'] = 'Intradural'
data.loc[data['CONCPT9'] == 63282, 'IEDUR'] = 'Intradural'
data.loc[data['CONCPT9'] == 63283, 'IEDUR'] = 'Intradural'

data.loc[data['CONCPT9'] == 63285, 'IEDUR'] = 'Intradural'
data.loc[data['CONCPT9'] == 63286, 'IEDUR'] = 'Intradural'
data.loc[data['CONCPT9'] == 63287, 'IEDUR'] = 'Intradural'
data.loc[data['CONCPT9'] == 63290, 'IEDUR'] = 'Intradural'

data.loc[data['CONCPT9'] == 63300, 'IEDUR'] = 'Extradural'
data.loc[data['CONCPT9'] == 63301, 'IEDUR'] = 'Extradural'
data.loc[data['CONCPT9'] == 63302, 'IEDUR'] = 'Extradural'
data.loc[data['CONCPT9'] == 63303, 'IEDUR'] = 'Extradural'

data.loc[data['CONCPT9'] == 63304, 'IEDUR'] = 'Intradural'
data.loc[data['CONCPT9'] == 63305, 'IEDUR'] = 'Intradural'
data.loc[data['CONCPT9'] == 63306, 'IEDUR'] = 'Intradural'
data.loc[data['CONCPT9'] == 63307, 'IEDUR'] = 'Intradural'

data.loc[data['CONCPT10'] == 63275, 'IEDUR'] = 'Extradural'
data.loc[data['CONCPT10'] == 63276, 'IEDUR'] = 'Extradural'
data.loc[data['CONCPT10'] == 63277, 'IEDUR'] = 'Extradural'
data.loc[data['CONCPT10'] == 63278, 'IEDUR'] = 'Extradural'

data.loc[data['CONCPT10'] == 63280, 'IEDUR'] = 'Intradural'
data.loc[data['CONCPT10'] == 63281, 'IEDUR'] = 'Intradural'
data.loc[data['CONCPT10'] == 63282, 'IEDUR'] = 'Intradural'
data.loc[data['CONCPT10'] == 63283, 'IEDUR'] = 'Intradural'

data.loc[data['CONCPT10'] == 63285, 'IEDUR'] = 'Intradural'
data.loc[data['CONCPT10'] == 63286, 'IEDUR'] = 'Intradural'
data.loc[data['CONCPT10'] == 63287, 'IEDUR'] = 'Intradural'
data.loc[data['CONCPT10'] == 63290, 'IEDUR'] = 'Intradural'

data.loc[data['CONCPT10'] == 63300, 'IEDUR'] = 'Extradural'
data.loc[data['CONCPT10'] == 63301, 'IEDUR'] = 'Extradural'
data.loc[data['CONCPT10'] == 63302, 'IEDUR'] = 'Extradural'
data.loc[data['CONCPT10'] == 63303, 'IEDUR'] = 'Extradural'

data.loc[data['CONCPT10'] == 63304, 'IEDUR'] = 'Intradural'
data.loc[data['CONCPT10'] == 63305, 'IEDUR'] = 'Intradural'
data.loc[data['CONCPT10'] == 63306, 'IEDUR'] = 'Intradural'
data.loc[data['CONCPT10'] == 63307, 'IEDUR'] = 'Intradural'

data.loc[data['OTHERCPT1'] == 63275, 'IEDUR'] = 'Extradural'
data.loc[data['OTHERCPT1'] == 63276, 'IEDUR'] = 'Extradural'
data.loc[data['OTHERCPT1'] == 63277, 'IEDUR'] = 'Extradural'
data.loc[data['OTHERCPT1'] == 63278, 'IEDUR'] = 'Extradural'

data.loc[data['OTHERCPT1'] == 63280, 'IEDUR'] = 'Intradural'
data.loc[data['OTHERCPT1'] == 63281, 'IEDUR'] = 'Intradural'
data.loc[data['OTHERCPT1'] == 63282, 'IEDUR'] = 'Intradural'
data.loc[data['OTHERCPT1'] == 63283, 'IEDUR'] = 'Intradural'

data.loc[data['OTHERCPT1'] == 63285, 'IEDUR'] = 'Intradural'
data.loc[data['OTHERCPT1'] == 63286, 'IEDUR'] = 'Intradural'
data.loc[data['OTHERCPT1'] == 63287, 'IEDUR'] = 'Intradural'
data.loc[data['OTHERCPT1'] == 63290, 'IEDUR'] = 'Intradural'

data.loc[data['OTHERCPT1'] == 63300, 'IEDUR'] = 'Extradural'
data.loc[data['OTHERCPT1'] == 63301, 'IEDUR'] = 'Extradural'
data.loc[data['OTHERCPT1'] == 63302, 'IEDUR'] = 'Extradural'
data.loc[data['OTHERCPT1'] == 63303, 'IEDUR'] = 'Extradural'

data.loc[data['OTHERCPT1'] == 63304, 'IEDUR'] = 'Intradural'
data.loc[data['OTHERCPT1'] == 63305, 'IEDUR'] = 'Intradural'
data.loc[data['OTHERCPT1'] == 63306, 'IEDUR'] = 'Intradural'
data.loc[data['OTHERCPT1'] == 63307, 'IEDUR'] = 'Intradural'

data.loc[data['OTHERCPT2'] == 63275, 'IEDUR'] = 'Extradural'
data.loc[data['OTHERCPT2'] == 63276, 'IEDUR'] = 'Extradural'
data.loc[data['OTHERCPT2'] == 63277, 'IEDUR'] = 'Extradural'
data.loc[data['OTHERCPT2'] == 63278, 'IEDUR'] = 'Extradural'

data.loc[data['OTHERCPT2'] == 63280, 'IEDUR'] = 'Intradural'
data.loc[data['OTHERCPT2'] == 63281, 'IEDUR'] = 'Intradural'
data.loc[data['OTHERCPT2'] == 63282, 'IEDUR'] = 'Intradural'
data.loc[data['OTHERCPT2'] == 63283, 'IEDUR'] = 'Intradural'

data.loc[data['OTHERCPT2'] == 63285, 'IEDUR'] = 'Intradural'
data.loc[data['OTHERCPT2'] == 63286, 'IEDUR'] = 'Intradural'
data.loc[data['OTHERCPT2'] == 63287, 'IEDUR'] = 'Intradural'
data.loc[data['OTHERCPT2'] == 63290, 'IEDUR'] = 'Intradural'

data.loc[data['OTHERCPT2'] == 63300, 'IEDUR'] = 'Extradural'
data.loc[data['OTHERCPT2'] == 63301, 'IEDUR'] = 'Extradural'
data.loc[data['OTHERCPT2'] == 63302, 'IEDUR'] = 'Extradural'
data.loc[data['OTHERCPT2'] == 63303, 'IEDUR'] = 'Extradural'

data.loc[data['OTHERCPT2'] == 63304, 'IEDUR'] = 'Intradural'
data.loc[data['OTHERCPT2'] == 63305, 'IEDUR'] = 'Intradural'
data.loc[data['OTHERCPT2'] == 63306, 'IEDUR'] = 'Intradural'
data.loc[data['OTHERCPT2'] == 63307, 'IEDUR'] = 'Intradural'

data.loc[data['OTHERCPT3'] == 63275, 'IEDUR'] = 'Extradural'
data.loc[data['OTHERCPT3'] == 63276, 'IEDUR'] = 'Extradural'
data.loc[data['OTHERCPT3'] == 63277, 'IEDUR'] = 'Extradural'
data.loc[data['OTHERCPT3'] == 63278, 'IEDUR'] = 'Extradural'

data.loc[data['OTHERCPT3'] == 63280, 'IEDUR'] = 'Intradural'
data.loc[data['OTHERCPT3'] == 63281, 'IEDUR'] = 'Intradural'
data.loc[data['OTHERCPT3'] == 63282, 'IEDUR'] = 'Intradural'
data.loc[data['OTHERCPT3'] == 63283, 'IEDUR'] = 'Intradural'

data.loc[data['OTHERCPT3'] == 63285, 'IEDUR'] = 'Intradural'
data.loc[data['OTHERCPT3'] == 63286, 'IEDUR'] = 'Intradural'
data.loc[data['OTHERCPT3'] == 63287, 'IEDUR'] = 'Intradural'
data.loc[data['OTHERCPT3'] == 63290, 'IEDUR'] = 'Intradural'

data.loc[data['OTHERCPT3'] == 63300, 'IEDUR'] = 'Extradural'
data.loc[data['OTHERCPT3'] == 63301, 'IEDUR'] = 'Extradural'
data.loc[data['OTHERCPT3'] == 63302, 'IEDUR'] = 'Extradural'
data.loc[data['OTHERCPT3'] == 63303, 'IEDUR'] = 'Extradural'

data.loc[data['OTHERCPT3'] == 63304, 'IEDUR'] = 'Intradural'
data.loc[data['OTHERCPT3'] == 63305, 'IEDUR'] = 'Intradural'
data.loc[data['OTHERCPT3'] == 63306, 'IEDUR'] = 'Intradural'
data.loc[data['OTHERCPT3'] == 63307, 'IEDUR'] = 'Intradural'

data.loc[data['OTHERCPT4'] == 63275, 'IEDUR'] = 'Extradural'
data.loc[data['OTHERCPT4'] == 63276, 'IEDUR'] = 'Extradural'
data.loc[data['OTHERCPT4'] == 63277, 'IEDUR'] = 'Extradural'
data.loc[data['OTHERCPT4'] == 63278, 'IEDUR'] = 'Extradural'

data.loc[data['OTHERCPT4'] == 63280, 'IEDUR'] = 'Intradural'
data.loc[data['OTHERCPT4'] == 63281, 'IEDUR'] = 'Intradural'
data.loc[data['OTHERCPT4'] == 63282, 'IEDUR'] = 'Intradural'
data.loc[data['OTHERCPT4'] == 63283, 'IEDUR'] = 'Intradural'

data.loc[data['OTHERCPT4'] == 63285, 'IEDUR'] = 'Intradural'
data.loc[data['OTHERCPT4'] == 63286, 'IEDUR'] = 'Intradural'
data.loc[data['OTHERCPT4'] == 63287, 'IEDUR'] = 'Intradural'
data.loc[data['OTHERCPT4'] == 63290, 'IEDUR'] = 'Intradural'

data.loc[data['OTHERCPT4'] == 63300, 'IEDUR'] = 'Extradural'
data.loc[data['OTHERCPT4'] == 63301, 'IEDUR'] = 'Extradural'
data.loc[data['OTHERCPT4'] == 63302, 'IEDUR'] = 'Extradural'
data.loc[data['OTHERCPT4'] == 63303, 'IEDUR'] = 'Extradural'

data.loc[data['OTHERCPT4'] == 63304, 'IEDUR'] = 'Intradural'
data.loc[data['OTHERCPT4'] == 63305, 'IEDUR'] = 'Intradural'
data.loc[data['OTHERCPT4'] == 63306, 'IEDUR'] = 'Intradural'
data.loc[data['OTHERCPT4'] == 63307, 'IEDUR'] = 'Intradural'

data.loc[data['OTHERCPT5'] == 63275, 'IEDUR'] = 'Extradural'
data.loc[data['OTHERCPT5'] == 63276, 'IEDUR'] = 'Extradural'
data.loc[data['OTHERCPT5'] == 63277, 'IEDUR'] = 'Extradural'
data.loc[data['OTHERCPT5'] == 63278, 'IEDUR'] = 'Extradural'

data.loc[data['OTHERCPT5'] == 63280, 'IEDUR'] = 'Intradural'
data.loc[data['OTHERCPT5'] == 63281, 'IEDUR'] = 'Intradural'
data.loc[data['OTHERCPT5'] == 63282, 'IEDUR'] = 'Intradural'
data.loc[data['OTHERCPT5'] == 63283, 'IEDUR'] = 'Intradural'

data.loc[data['OTHERCPT5'] == 63285, 'IEDUR'] = 'Intradural'
data.loc[data['OTHERCPT5'] == 63286, 'IEDUR'] = 'Intradural'
data.loc[data['OTHERCPT5'] == 63287, 'IEDUR'] = 'Intradural'
data.loc[data['OTHERCPT5'] == 63290, 'IEDUR'] = 'Intradural'

data.loc[data['OTHERCPT5'] == 63300, 'IEDUR'] = 'Extradural'
data.loc[data['OTHERCPT5'] == 63301, 'IEDUR'] = 'Extradural'
data.loc[data['OTHERCPT5'] == 63302, 'IEDUR'] = 'Extradural'
data.loc[data['OTHERCPT5'] == 63303, 'IEDUR'] = 'Extradural'

data.loc[data['OTHERCPT5'] == 63304, 'IEDUR'] = 'Intradural'
data.loc[data['OTHERCPT5'] == 63305, 'IEDUR'] = 'Intradural'
data.loc[data['OTHERCPT5'] == 63306, 'IEDUR'] = 'Intradural'
data.loc[data['OTHERCPT5'] == 63307, 'IEDUR'] = 'Intradural'

data.loc[data['OTHERCPT6'] == 63275, 'IEDUR'] = 'Extradural'
data.loc[data['OTHERCPT6'] == 63276, 'IEDUR'] = 'Extradural'
data.loc[data['OTHERCPT6'] == 63277, 'IEDUR'] = 'Extradural'
data.loc[data['OTHERCPT6'] == 63278, 'IEDUR'] = 'Extradural'

data.loc[data['OTHERCPT6'] == 63280, 'IEDUR'] = 'Intradural'
data.loc[data['OTHERCPT6'] == 63281, 'IEDUR'] = 'Intradural'
data.loc[data['OTHERCPT6'] == 63282, 'IEDUR'] = 'Intradural'
data.loc[data['OTHERCPT6'] == 63283, 'IEDUR'] = 'Intradural'

data.loc[data['OTHERCPT6'] == 63285, 'IEDUR'] = 'Intradural'
data.loc[data['OTHERCPT6'] == 63286, 'IEDUR'] = 'Intradural'
data.loc[data['OTHERCPT6'] == 63287, 'IEDUR'] = 'Intradural'
data.loc[data['OTHERCPT6'] == 63290, 'IEDUR'] = 'Intradural'

data.loc[data['OTHERCPT6'] == 63300, 'IEDUR'] = 'Extradural'
data.loc[data['OTHERCPT6'] == 63301, 'IEDUR'] = 'Extradural'
data.loc[data['OTHERCPT6'] == 63302, 'IEDUR'] = 'Extradural'
data.loc[data['OTHERCPT6'] == 63303, 'IEDUR'] = 'Extradural'

data.loc[data['OTHERCPT6'] == 63304, 'IEDUR'] = 'Intradural'
data.loc[data['OTHERCPT6'] == 63305, 'IEDUR'] = 'Intradural'
data.loc[data['OTHERCPT6'] == 63306, 'IEDUR'] = 'Intradural'
data.loc[data['OTHERCPT6'] == 63307, 'IEDUR'] = 'Intradural'

data.loc[data['OTHERCPT7'] == 63275, 'IEDUR'] = 'Extradural'
data.loc[data['OTHERCPT7'] == 63276, 'IEDUR'] = 'Extradural'
data.loc[data['OTHERCPT7'] == 63277, 'IEDUR'] = 'Extradural'
data.loc[data['OTHERCPT7'] == 63278, 'IEDUR'] = 'Extradural'

data.loc[data['OTHERCPT7'] == 63280, 'IEDUR'] = 'Intradural'
data.loc[data['OTHERCPT7'] == 63281, 'IEDUR'] = 'Intradural'
data.loc[data['OTHERCPT7'] == 63282, 'IEDUR'] = 'Intradural'
data.loc[data['OTHERCPT7'] == 63283, 'IEDUR'] = 'Intradural'

data.loc[data['OTHERCPT7'] == 63285, 'IEDUR'] = 'Intradural'
data.loc[data['OTHERCPT7'] == 63286, 'IEDUR'] = 'Intradural'
data.loc[data['OTHERCPT7'] == 63287, 'IEDUR'] = 'Intradural'
data.loc[data['OTHERCPT7'] == 63290, 'IEDUR'] = 'Intradural'

data.loc[data['OTHERCPT7'] == 63300, 'IEDUR'] = 'Extradural'
data.loc[data['OTHERCPT7'] == 63301, 'IEDUR'] = 'Extradural'
data.loc[data['OTHERCPT7'] == 63302, 'IEDUR'] = 'Extradural'
data.loc[data['OTHERCPT7'] == 63303, 'IEDUR'] = 'Extradural'

data.loc[data['OTHERCPT7'] == 63304, 'IEDUR'] = 'Intradural'
data.loc[data['OTHERCPT7'] == 63305, 'IEDUR'] = 'Intradural'
data.loc[data['OTHERCPT7'] == 63306, 'IEDUR'] = 'Intradural'
data.loc[data['OTHERCPT7'] == 63307, 'IEDUR'] = 'Intradural'

data.loc[data['OTHERCPT8'] == 63275, 'IEDUR'] = 'Extradural'
data.loc[data['OTHERCPT8'] == 63276, 'IEDUR'] = 'Extradural'
data.loc[data['OTHERCPT8'] == 63277, 'IEDUR'] = 'Extradural'
data.loc[data['OTHERCPT8'] == 63278, 'IEDUR'] = 'Extradural'

data.loc[data['OTHERCPT8'] == 63280, 'IEDUR'] = 'Intradural'
data.loc[data['OTHERCPT8'] == 63281, 'IEDUR'] = 'Intradural'
data.loc[data['OTHERCPT8'] == 63282, 'IEDUR'] = 'Intradural'
data.loc[data['OTHERCPT8'] == 63283, 'IEDUR'] = 'Intradural'

data.loc[data['OTHERCPT8'] == 63285, 'IEDUR'] = 'Intradural'
data.loc[data['OTHERCPT8'] == 63286, 'IEDUR'] = 'Intradural'
data.loc[data['OTHERCPT8'] == 63287, 'IEDUR'] = 'Intradural'
data.loc[data['OTHERCPT8'] == 63290, 'IEDUR'] = 'Intradural'

data.loc[data['OTHERCPT8'] == 63300, 'IEDUR'] = 'Extradural'
data.loc[data['OTHERCPT8'] == 63301, 'IEDUR'] = 'Extradural'
data.loc[data['OTHERCPT8'] == 63302, 'IEDUR'] = 'Extradural'
data.loc[data['OTHERCPT8'] == 63303, 'IEDUR'] = 'Extradural'

data.loc[data['OTHERCPT8'] == 63304, 'IEDUR'] = 'Intradural'
data.loc[data['OTHERCPT8'] == 63305, 'IEDUR'] = 'Intradural'
data.loc[data['OTHERCPT8'] == 63306, 'IEDUR'] = 'Intradural'
data.loc[data['OTHERCPT8'] == 63307, 'IEDUR'] = 'Intradural'

data.loc[data['OTHERCPT9'] == 63275, 'IEDUR'] = 'Extradural'
data.loc[data['OTHERCPT9'] == 63276, 'IEDUR'] = 'Extradural'
data.loc[data['OTHERCPT9'] == 63277, 'IEDUR'] = 'Extradural'
data.loc[data['OTHERCPT9'] == 63278, 'IEDUR'] = 'Extradural'

data.loc[data['OTHERCPT9'] == 63280, 'IEDUR'] = 'Intradural'
data.loc[data['OTHERCPT9'] == 63281, 'IEDUR'] = 'Intradural'
data.loc[data['OTHERCPT9'] == 63282, 'IEDUR'] = 'Intradural'
data.loc[data['OTHERCPT9'] == 63283, 'IEDUR'] = 'Intradural'

data.loc[data['OTHERCPT9'] == 63285, 'IEDUR'] = 'Intradural'
data.loc[data['OTHERCPT9'] == 63286, 'IEDUR'] = 'Intradural'
data.loc[data['OTHERCPT9'] == 63287, 'IEDUR'] = 'Intradural'
data.loc[data['OTHERCPT9'] == 63290, 'IEDUR'] = 'Intradural'

data.loc[data['OTHERCPT9'] == 63300, 'IEDUR'] = 'Extradural'
data.loc[data['OTHERCPT9'] == 63301, 'IEDUR'] = 'Extradural'
data.loc[data['OTHERCPT9'] == 63302, 'IEDUR'] = 'Extradural'
data.loc[data['OTHERCPT9'] == 63303, 'IEDUR'] = 'Extradural'

data.loc[data['OTHERCPT9'] == 63304, 'IEDUR'] = 'Intradural'
data.loc[data['OTHERCPT9'] == 63305, 'IEDUR'] = 'Intradural'
data.loc[data['OTHERCPT9'] == 63306, 'IEDUR'] = 'Intradural'
data.loc[data['OTHERCPT9'] == 63307, 'IEDUR'] = 'Intradural'

data.loc[data['OTHERCPT10'] == 63275, 'IEDUR'] = 'Extradural'
data.loc[data['OTHERCPT10'] == 63276, 'IEDUR'] = 'Extradural'
data.loc[data['OTHERCPT10'] == 63277, 'IEDUR'] = 'Extradural'
data.loc[data['OTHERCPT10'] == 63278, 'IEDUR'] = 'Extradural'

data.loc[data['OTHERCPT10'] == 63280, 'IEDUR'] = 'Intradural'
data.loc[data['OTHERCPT10'] == 63281, 'IEDUR'] = 'Intradural'
data.loc[data['OTHERCPT10'] == 63282, 'IEDUR'] = 'Intradural'
data.loc[data['OTHERCPT10'] == 63283, 'IEDUR'] = 'Intradural'

data.loc[data['OTHERCPT10'] == 63285, 'IEDUR'] = 'Intradural'
data.loc[data['OTHERCPT10'] == 63286, 'IEDUR'] = 'Intradural'
data.loc[data['OTHERCPT10'] == 63287, 'IEDUR'] = 'Intradural'
data.loc[data['OTHERCPT10'] == 63290, 'IEDUR'] = 'Intradural'

data.loc[data['OTHERCPT10'] == 63300, 'IEDUR'] = 'Extradural'
data.loc[data['OTHERCPT10'] == 63301, 'IEDUR'] = 'Extradural'
data.loc[data['OTHERCPT10'] == 63302, 'IEDUR'] = 'Extradural'
data.loc[data['OTHERCPT10'] == 63303, 'IEDUR'] = 'Extradural'

data.loc[data['OTHERCPT10'] == 63304, 'IEDUR'] = 'Intradural'
data.loc[data['OTHERCPT10'] == 63305, 'IEDUR'] = 'Intradural'
data.loc[data['OTHERCPT10'] == 63306, 'IEDUR'] = 'Intradural'
data.loc[data['OTHERCPT10'] == 63307, 'IEDUR'] = 'Intradural'

data['IEDUR'].value_counts(dropna=False)


# In[ ]:


#Simplify CPT codes and assign other CPT codes as 'other'.

data.loc[data['CPT'] == 63275, 'CPTx'] = '63275'
data.loc[data['CPT'] == 63276, 'CPTx'] = '63276'
data.loc[data['CPT'] == 63277, 'CPTx'] = '63277'
data.loc[data['CPT'] == 63278, 'CPTx'] = '63278'
data.loc[data['CPT'] == 63280, 'CPTx'] = '63280'
data.loc[data['CPT'] == 63281, 'CPTx'] = '63281'
data.loc[data['CPT'] == 63282, 'CPTx'] = '63282'
data.loc[data['CPT'] == 63283, 'CPTx'] = '63283'
data.loc[data['CPT'] == 63285, 'CPTx'] = '63285'
data.loc[data['CPT'] == 63286, 'CPTx'] = '63286'
data.loc[data['CPT'] == 63287, 'CPTx'] = '63287'
data.loc[data['CPT'] == 63290, 'CPTx'] = '63290'
data.loc[data['CPT'] == 63300, 'CPTx'] = '63300'
data.loc[data['CPT'] == 63301, 'CPTx'] = '63301'
data.loc[data['CPT'] == 63302, 'CPTx'] = '63302'
data.loc[data['CPT'] == 63303, 'CPTx'] = '63303'
data.loc[data['CPT'] == 63304, 'CPTx'] = '63304'
data.loc[data['CPT'] == 63305, 'CPTx'] = '63305'
data.loc[data['CPT'] == 63306, 'CPTx'] = '63306'
data.loc[data['CPT'] == 63307, 'CPTx'] = '63307'

data['CPTx'].fillna(value='Other', inplace=True)
data['CPTx'].value_counts(dropna=False)


# In[ ]:


#Define major complications.

data['MAJRCOMP'] = data['NWNDINFD'] + data['NORGSPCSSI'] + data['NDEHIS'] + data['NREINTUB'] + data['NPULEMBOL'] + data['NFAILWEAN'] + data['NRENAINSF'] + data['NOPRENAFL'] + data['NCNSCVA'] + data['NCDARREST'] + data['NCDMI'] + data['NOTHBLEED'] + data['NOTHDVT'] + data['NOTHSYSEP'] + data['NOTHSESHOCK']


# In[ ]:


#Show number of major complications per patient.

data['MAJRCOMP'].value_counts()


# In[ ]:


#Convert major complications into categorical data in a column named 'COMP'.

data.loc[data['MAJRCOMP'] == 0, 'COMP'] = 'No'
data.loc[data['MAJRCOMP'] >= 1, 'COMP'] = 'Yes'


# In[ ]:


#Show major complications as categorical.

data['COMP'].value_counts()


# In[ ]:


#See LOS per patient.

data['TOTHLOS'].value_counts()


# In[ ]:


#See 75th percentile of LOS.
data.TOTHLOS.quantile(0.75)


# In[ ]:


#Convert total length of stay into categorical data in a column named 'LOS'.

data.loc[data['TOTHLOS'] <= data.TOTHLOS.quantile(0.75), 'LOS'] = 'No'
data.loc[data['TOTHLOS'] > data.TOTHLOS.quantile(0.75), 'LOS'] = 'Yes'


# In[ ]:


#Show major complications as categorical.

data['LOS'].value_counts(dropna=False)


# In[ ]:


#Show readmission status.

data['READMISSION1'].value_counts(dropna=False)


# In[ ]:


#Drop patients with unknown readmission status.

data = data[data['READMISSION1'].notna()]


# In[ ]:


#Show readmission status after dropping patients with unknown readmission status.

data['READMISSION1'].value_counts(dropna=False)


# In[ ]:


#Show readmission status.

data['DISCHDEST'].value_counts(dropna=False)


# In[ ]:


#Convert discharge destination into binary data (home vs. non-home discharge) in a column named 'DISCHARGE'.

data.loc[data['DISCHDEST'] == 'Home', 'DISCHARGE'] = 'No'
data.loc[data['DISCHDEST'] == 'Facility Which was Home', 'DISCHARGE'] = 'No'
data.loc[data['DISCHDEST'] == 'Rehab', 'DISCHARGE'] = 'Yes'
data.loc[data['DISCHDEST'] == 'Skilled Care, Not Home', 'DISCHARGE'] = 'Yes'
data.loc[data['DISCHDEST'] == 'Separate Acute Care', 'DISCHARGE'] = 'Yes'
data.loc[data['DISCHDEST'] == 'Unskilled Facility Not Home', 'DISCHARGE'] = 'Yes'


# In[ ]:


#Show discharge destination status after converting it to binary data.

data['DISCHARGE'].value_counts(dropna=False)


# In[ ]:


#Drop patients with unknown discharge status.

data = data[data['DISCHARGE'].notna()]


# In[ ]:


#Show discharge status after dropping patients with unknown discharge status.

data['DISCHARGE'].value_counts(dropna=False)


# In[ ]:


#Check data.

data.shape


# In[ ]:


#Save data.

data.to_csv('st_cpt_only_clean.csv')


# In[ ]:


#Read data again.

data = pd.read_csv('st_cpt_only_clean.csv', index_col = 0)
data.head()


# In[ ]:


#See all columns.

print(list(data.columns))


# In[ ]:


#Drop unwanted columns.

drop = ['RACE_NEW', 'ETHNICITY_HISPANIC', 'CPT', 'INOUT', 'DISCHDEST', 'ANESTHES','ELECTSURG', 'VENTILAT', 'PRSEPIS', 'OTHERCPT1', 'OTHERCPT2', 'OTHERCPT3', 'OTHERCPT4', 'OTHERCPT5', 'OTHERCPT6', 'OTHERCPT7', 'OTHERCPT8', 'OTHERCPT9', 'OTHERCPT10', 'CONCPT1', 'CONCPT2', 'CONCPT3', 'CONCPT4', 'CONCPT5', 'CONCPT6', 'CONCPT7', 'CONCPT8', 'CONCPT9', 'CONCPT10', 'EMERGNCY', 'WNDCLAS', 'OPTIME','OPTIME', 'TOTHLOS', 'NSUPINFEC', 'NWNDINFD', 'NORGSPCSSI', 'NDEHIS', 'NOUPNEUMO', 'NREINTUB', 'NPULEMBOL', 'NFAILWEAN', 'NRENAINSF', 'NOPRENAFL', 'NURNINFEC', 'NCNSCVA', 'NCDARREST', 'NCDMI', 'NOTHBLEED', 'NOTHDVT', 'NOTHSYSEP', 'NOTHSESHOCK', 'PODIAG', 'PODIAG10', 'STILLINHOSP', 'MAJRCOMP']
data.drop(drop, axis=1, inplace=True)


# In[ ]:


#Check data shape.

data.shape


# In[ ]:


#See categorical and continuous variables.

print('Numerical columns: {}'.format(list(data.select_dtypes('number').columns)))
print()
print('Categorical columns: {}'.format(list(data.select_dtypes('object').columns)))


# In[ ]:


#Define numerical and categorical columns.

num_cols = ['AGE', 'HEIGHT', 'WEIGHT', 'PRSODM', 'PRBUN', 'PRCREAT', 'PRALBUM', 'PRBILI', 'PRSGOT', 'PRALKPH', 'PRWBC', 'PRHCT', 'PRPLATE', 'PRPTT', 'PRINR', 'PRPT', 'BMI', 'HTOODAY']

cat_cols = ['SEX', 'OPERYR', 'TRANST', 'SURGSPEC', 'DIABETES', 'SMOKE', 'DYSPNEA', 'FNSTATUS2', 'HXCOPD', 'ASCITES', 'HXCHF', 'HYPERMED', 'RENAFAIL', 'DIALYSIS', 'DISCANCR', 'WNDINF', 'STEROID', 'WTLOSS', 'BLEEDDIS', 'TRANSFUS', 'ASACLAS', 'REOPERATION1', 'READMISSION1', 'RACE', 'IEDUR', 'CPTx', 'COMP', 'LOS', 'DISCHARGE']


# In[ ]:


#Check missing values for numerical variables.

data[num_cols].isnull().mean().round(4).mul(100).sort_values(ascending=False)


# In[ ]:


#Drop columns with missing values over 25%.

drop = ['PRPT', 'PRBILI', 'PRALKPH', 'PRSGOT', 'PRALBUM', 'PRPTT', 'PRINR']
data.drop(drop, axis=1, inplace=True)


# In[ ]:


#Redefine new numerical columns.

num_cols = ['AGE', 'HEIGHT', 'WEIGHT', 'PRSODM', 'PRBUN', 'PRCREAT', 'PRWBC', 'PRHCT', 'PRPLATE', 'BMI']


# In[ ]:


#Impute missing numerical values.
num_imputer = KNNImputer(n_neighbors=5, weights='uniform', metric='nan_euclidean')
data[num_cols] = num_imputer.fit_transform(data[num_cols])


# In[ ]:


#Check numerical variables with missing values after imputation.

data[num_cols].isnull().mean().round(4).mul(100).sort_values(ascending=False)


# In[ ]:


#Check missing values for categorical variables.

data[cat_cols].isnull().mean().round(4).mul(100).sort_values(ascending=False)


# In[ ]:


#Drop columns with missing values over 25%.

drop = []
data.drop(drop, axis=1, inplace=True)


# In[ ]:


#Redefine categorical variables.

cat_cols = ['SEX', 'OPERYR', 'TRANST', 'SURGSPEC', 'DIABETES', 'SMOKE', 'DYSPNEA', 'FNSTATUS2', 'HXCOPD', 'ASCITES', 'HXCHF', 'HYPERMED', 'RENAFAIL', 'DIALYSIS', 'DISCANCR', 'WNDINF', 'STEROID', 'WTLOSS', 'BLEEDDIS', 'TRANSFUS', 'ASACLAS', 'REOPERATION1', 'READMISSION1', 'RACE', 'IEDUR', 'CPTx', 'COMP', 'LOS', 'DISCHARGE']


# In[ ]:


#Replace missing categorical values with 'Unknown'.

for col in cat_cols:
    data[col].fillna(value='Unknown', inplace=True)


# In[ ]:


#Check missing values after imputation.

data[cat_cols].isnull().mean().round(4).mul(100).sort_values(ascending=False)


# In[ ]:


#Save imputed data.

data.to_csv('st_cpt_only_imputed.csv')


# In[ ]:


#RobustScale data.

data[num_cols] = RobustScaler().fit_transform(data[num_cols])


# In[ ]:


#Normalize data.

data[num_cols] = MinMaxScaler().fit_transform(data[num_cols])


# In[ ]:


#Save scaled data.

data.to_csv('st_cpt_only_scaled.csv')


# In[ ]:


#One hot encoding for categorical values.

data_final = pd.get_dummies(data, columns = cat_cols, drop_first = False)


# In[ ]:


#Save final data.

data_final.to_csv('st_cpt_only_final.csv')

