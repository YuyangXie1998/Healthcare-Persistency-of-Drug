
# Importing the libraries
import numpy as np
import pandas as pd
import pickle

import warnings
warnings.filterwarnings("ignore")

# Loading the dataset
dataset = pd.read_csv('Healthcare_dataset.csv')

cat_list = ['Region',
 'Ntm_Speciality',
 'Ntm_Specialist_Flag',
 'Ntm_Speciality_Bucket',
 'Gluco_Record_During_Rx',
 'Dexa_During_Rx',
 'Frag_Frac_During_Rx',
 'Risk_Segment_During_Rx',
 'Tscore_Bucket_During_Rx',
 'Change_T_Score',
 'Change_Risk_Segment',
 'Adherent_Flag',
 'Idn_Indicator',
 'Injectable_Experience_During_Rx',
 'Comorb_Encounter_For_Screening_For_Malignant_Neoplasms',
 'Comorb_Encounter_For_Immunization',
 'Comorb_Encntr_For_General_Exam_W_O_Complaint,_Susp_Or_Reprtd_Dx',
 'Comorb_Vitamin_D_Deficiency',
 'Comorb_Other_Joint_Disorder_Not_Elsewhere_Classified',
 'Comorb_Encntr_For_Oth_Sp_Exam_W_O_Complaint_Suspected_Or_Reprtd_Dx',
 'Comorb_Long_Term_Current_Drug_Therapy',
 'Comorb_Dorsalgia',
 'Comorb_Personal_History_Of_Other_Diseases_And_Conditions',
 'Comorb_Other_Disorders_Of_Bone_Density_And_Structure',
 'Comorb_Disorders_of_lipoprotein_metabolism_and_other_lipidemias',
 'Comorb_Osteoporosis_without_current_pathological_fracture',
 'Comorb_Personal_history_of_malignant_neoplasm',
 'Comorb_Gastro_esophageal_reflux_disease',
 'Concom_Cholesterol_And_Triglyceride_Regulating_Preparations',
 'Concom_Narcotics',
 'Concom_Systemic_Corticosteroids_Plain',
 'Concom_Anti_Depressants_And_Mood_Stabilisers',
 'Concom_Fluoroquinolones',
 'Concom_Cephalosporins',
 'Concom_Macrolides_And_Similar_Types',
 'Concom_Broad_Spectrum_Penicillins',
 'Concom_Anaesthetics_General',
 'Concom_Viral_Vaccines',
 'Risk_Rheumatoid_Arthritis',
 'Risk_Untreated_Chronic_Hypogonadism',
 'Risk_Smoking_Tobacco',
 'Risk_Vitamin_D_Insufficiency']

num_list = ['Dexa_Freq_During_Rx', 'Count_Of_Risks']

X = pd.get_dummies(dataset[cat_list + num_list])

dataset['Persistency_Flag'] = dataset['Persistency_Flag'].str.replace('Non-Persistent', '0')
dataset['Persistency_Flag'] = dataset['Persistency_Flag'].str.replace('Persistent', '1')
y = pd.to_numeric(dataset['Persistency_Flag'])

# Importing the model
from xgboost import XGBClassifier

import re
regex = re.compile(r"\[|\]|<", re.IGNORECASE)
X.columns = [regex.sub("_", col) if any(x in str(col) for x in set(('[', ']', '<'))) else col for col in X.columns.values]

regressor = XGBClassifier()

# Fitting model with trainig data
regressor.fit(X, y)

# Saving model to disk
pickle.dump(regressor, open('model_healthcare.pkl','wb'))

# Loading model to compare the results
model = pickle.load(open('model_healthcare.pkl','rb'))

# Try the model
deploy_list = [['West', 'GENERAL PRACTITIONER', 'Specialist', 'OB/GYN/Others/PCP/Unknown', 'Y', 'Y', 'Y', 'VLR_LR',
'<=-2.5', 'Improved', 'Improved', 'Adherent', 'Y', 'Y', 'Y', 'Y', 'Y', 'Y', 'N', 'Y', 'Y', 'Y', 'Y', 'N', 'Y', 'Y', 'Y',
'Y', 'Y', 'N', 'Y', 'Y', 'Y', 'Y', 'N', 'Y', 'Y', 'Y', 'Y', 'Y', 'Y', 'Y', 45, 2]]
deploy_df = pd.DataFrame(deploy_list, columns = cat_list + num_list)

# Save columns for later use
model_columns = pd.DataFrame(X.columns, columns = ['Features'])['Features']
 
# The next function will add missing columns (in response to df_train)
def add_missing_dummy_columns(df, columns):
  missing_cols = set(columns) - set(df.columns)
  for c in missing_cols:
    df[c] = 0

# The next function will delete extra columns (in response to df_train)
def fix_columns(df, columns):
  add_missing_dummy_columns(df, columns)
  # make sure we have all the columns we need
  assert(set(columns) - set(df.columns) == set())
  extra_cols = set(df.columns) - set(columns)
  if extra_cols:
    df = df[columns]
  return df
 
# Execute get_dummies to One-Hot the deploy dataset
deploy_df_enc = pd.get_dummies(deploy_df)

# Run the above functions to pad the deploy dataset
fixed_deploy_df = fix_columns(deploy_df_enc, model_columns)
fixed_deploy_df.columns = [regex.sub("_", col) if any(x in str(col) for x in set(('[', ']', '<'))) else col for col in fixed_deploy_df.columns.values]

print(model.predict(fixed_deploy_df[X.columns]))