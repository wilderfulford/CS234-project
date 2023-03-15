import pandas as pd
import numpy as np
import re


def preprocess_data(filepath):
    """
    Generator that randomly shuffles dataset and yields 31-dimensional
    np  vectors, where age, height, weight, Target INR are integers and 
    everything else is one-hot encoded:
        gender, race, ethnicity, specific comorbidity, specific medication.

    """
    
    # Read the CSV file
    df = pd.read_csv(filepath, na_values=['NA', ''])
    
    # Convert age ranges to appropriate numeric values
    def age_to_numeric(age):
        if pd.isna(age):
            return age
        age_range = age.strip()
        return int(age_range[0]) // 10
    
    df['Age'] = df['Age'].apply(age_to_numeric)

    # Replace NaN values in the Age, Height (cm), and Weight (kg) columns with their respective medians
    df['Age'].fillna(df['Age'].median(), inplace=True)
    df['Height (cm)'].fillna(df['Height (cm)'].median(), inplace=True)
    df['Weight (kg)'].fillna(df['Weight (kg)'].median(), inplace=True)
    df['Target INR'].fillna(df['Target INR'].median(), inplace=True)
    df['Ethnicity'] = df['Ethnicity'].apply(lambda x: 1 if x == 'Hispanic or Latino' else 0)
    df['Gender'] = df['Gender'].apply(lambda x: 1 if x == 'male' else 0)
    

    # Select relevant columns for the feature vector
    columns = [
        'Gender', 'Race', 'Ethnicity', 'Age', 'Height (cm)', 'Weight (kg)', 
        'Diabetes', 'Congestive Heart Failure and/or Cardiomyopathy', 
        'Valve Replacement', 'Aspirin', 'Acetaminophen or Paracetamol (Tylenol)', 'Simvastatin (Zocor)', 
        'Atorvastatin (Lipitor)', 'Fluvastatin (Lescol)', 'Pravastatin (Pravachol)', 'Rosuvastatin (Crestor)', 
        'Cerivastatin (Baycol)', 'Amiodarone (Cordarone)', 'Carbamazepine (Tegretol)', 'Phenytoin (Dilantin)', 
        'Rifampin or Rifampicin', 'Sulfonamide Antibiotics', 'Macrolide Antibiotics', 'Anti-fungal Azoles', 
        'Herbal Medications, Vitamins, Supplements', 'Target INR', 'Current Smoker'
    ]

    # Extract the relevant columns from the dataframe
    feature_df = df[columns]
    
    ### Update drug and comorbidity entries
    # Drug names and their corresponding column names
    drug_names = {
        'Simvastatin': 'Simvastatin (Zocor)',
        'Zocor': 'Simvastatin (Zocor)',
        'Atorvastatin': 'Atorvastatin (Lipitor)',
        'Lipitor': 'Atorvastatin (Lipitor)',
        'Fluvastatin': 'Fluvastatin (Lescol)',
        'Lescol': 'Fluvastatin (Lescol)',
        'Pravastatin': 'Pravastatin (Pravachol)',
        'Pravachol': 'Pravastatin (Pravachol)',
        'Rosuvastatin': 'Rosuvastatin (Crestor)',
        'Crestor': 'Rosuvastatin (Crestor)',
        'Cerivastatin': 'Cerivastatin (Baycol)',
        'Baycol': 'Cerivastatin (Baycol)',
        'Amiodarone': 'Amiodarone (Cordarone)',
        'Cordarone': 'Amiodarone (Cordarone)',
        'Carbamazepine': 'Carbamazepine (Tegretol)',
        'Tegretol': 'Carbamazepine (Tegretol)',
        'Phenytoin': 'Phenytoin (Dilantin)',
        'Dilantin': 'Phenytoin (Dilantin)',
        'Rifampin': 'Rifampin or Rifampicin',
        'Rifampicin': 'Rifampin or Rifampicin',
        'Septra': 'Sulfonamide Antibiotics',
        'Bactrim': 'Sulfonamide Antibiotics',
        'Cotrim': 'Sulfonamide Antibiotics',
        'Sulfatrim': 'Sulfonamide Antibiotics',
        'Erythromycin': 'Macrolide Antibiotics',
        'Azithromycin': 'Macrolide Antibiotics',
        'Clarithromycin': 'Macrolide Antibiotics',
    }

    # Update the corresponding drug columns based on the presence of drug names in 'Medications'
    for drug, column in drug_names.items():
        drug_pattern = re.compile(r'(?<!not\s)({}|{})'.format(drug, drug.lower()), re.IGNORECASE)
        df.loc[df['Medications'].apply(lambda x: bool(drug_pattern.search(str(x)))), column] = 1

    # Updating corresponding columns based on presence in 'Comorbidities' column
    conditions_to_check = {
        'Diabetes': 'Diabetes', 
        'Valve Replacement': 'Valve Replacement',
        'Congestive Heart Failure': 'Congestive Heart Failure and/or Cardiomyopathy',
        'Cardiomyopathy': 'Congestive Heart Failure and/or Cardiomyopathy'
    }
    
    for condition, column in conditions_to_check.items():
        condition_pattern = rf'(?i)(?<!not ){condition}'
        df.loc[df['Comorbidities'].str.contains(condition_pattern, na=False), column] = 1
    
    
    ### Produce and yield output vectors
    # One-hot encode categorical columns
    categorical_columns = [
        'Gender', 'Race', 'Ethnicity','Diabetes',
        'Congestive Heart Failure and/or Cardiomyopathy', 'Valve Replacement', 'Aspirin',
        'Acetaminophen or Paracetamol (Tylenol)', 'Simvastatin (Zocor)', 'Atorvastatin (Lipitor)',
        'Fluvastatin (Lescol)', 'Pravastatin (Pravachol)', 'Rosuvastatin (Crestor)', 'Cerivastatin (Baycol)',
        'Amiodarone (Cordarone)', 'Carbamazepine (Tegretol)', 'Phenytoin (Dilantin)', 'Rifampin or Rifampicin',
        'Sulfonamide Antibiotics', 'Macrolide Antibiotics', 'Anti-fungal Azoles', 'Herbal Medications, Vitamins, Supplements',
        'Current Smoker'
    ]
    
    # remove nan values
    def replace_nan_with_zero(value):
        return 0 if (isinstance(value, float) and np.isnan(value)) else value
    
    for column in categorical_columns:
        feature_df[column] = feature_df[column].apply(replace_nan_with_zero)
        df[column].fillna(0)

    df['Race'].replace(0, 'Unknown')
    categorical_columns_to_encode = ['Race']
    
    feature_df = pd.get_dummies(feature_df, columns=categorical_columns_to_encode)
    
    # Shuffle the preprocessed dataframe
    feature_df = feature_df.sample(frac=1).reset_index(drop=True)
    # Yield feature vectors one at a time
    for index, row in feature_df.iterrows():
        yield row.to_numpy()
        
        

i = 0
for row in preprocess_data('data/warfarin.csv'):
    print(row)
    print(row.size)
    i += 1
    if i == 10:
        break
    