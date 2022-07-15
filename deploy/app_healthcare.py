
import numpy as np
import pandas as pd
from flask import Flask, request, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('model_healthcare.pkl', 'rb'))

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

import re
regex = re.compile(r"\[|\]|<", re.IGNORECASE)
X.columns = [regex.sub("_", col) if any(x in str(col) for x in set(('[', ']', '<'))) else col for col in X.columns.values]

#deploy_features_enc = pd.DataFrame()
#fixed_deploy_df = pd.DataFrame()

@app.route('/')
def home():
    return render_template('index_healthcare.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    #int_features = [int(x) for x in request.form.values()]
    get_features = request.form
    
    
    deploy_features = pd.DataFrame(get_features, index=[0], columns = cat_list + num_list)

    #Save columns for later use
    model_columns = pd.DataFrame(X.columns, columns = ['Features'])['Features']
     
    #The next function will add missing columns (in response to df_train)
    def add_missing_dummy_columns(df, columns):
      missing_cols = set(columns) - set(df.columns)
      for c in missing_cols:
        df[c] = 0

    #The next function will delete extra columns (in response to df_train)
    def fix_columns(df, columns):
      add_missing_dummy_columns(df, columns)
      # make sure we have all the columns we need
      assert(set(columns) - set(df.columns) == set())
      extra_cols = set(df.columns) - set(columns)
      if extra_cols:
        df = df[columns]
      return df
     
    #Execute get_dummies to One-Hot the deploy dataset
    deploy_features_enc = pd.get_dummies(deploy_features)

    #Run the above functions to pad the deploy dataset
    fixed_deploy_df = fix_columns(deploy_features_enc, model_columns)

    fixed_deploy_df.columns = [regex.sub("_", col) if any(x in str(col) for x in set(('[', ']', '<'))) else col for col in fixed_deploy_df.columns.values]

    #deploy_pred = xgb.predict(fixed_deploy_df[X.columns])
    #print(model.predict(fixed_deploy_df[X.columns]))
    #final_features = [np.array(int_features)]
    
    final_features = fixed_deploy_df[X.columns]
    prediction = model.predict(final_features)

    output = np.round(prediction[0], 2)

    return render_template('index_healthcare.html', prediction_text='This patient will be {} (1-Persistent/0-Non-Persistent)'.format(output))

if __name__ == "__main__":
    app.run(debug=True)