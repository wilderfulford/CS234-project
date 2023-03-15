import pandas as pd
import numpy as np
import scipy.stats as stats
import re
import matplotlib.pyplot as plt
import random

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
        'Herbal Medications, Vitamins, Supplements', 'Target INR', 'Current Smoker', 'Therapeutic Dose of Warfarin'
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
    
    # Yield feature vectors one at a time
    for index, row in feature_df.iterrows():
        dose = row['Therapeutic Dose of Warfarin']
        row = row.drop('Therapeutic Dose of Warfarin')

        yield row.to_numpy(), dose
   
        
def patient_generator(file_path):
    preprocessor = preprocess_data(file_path)
    patients = list(preprocessor)
    random.shuffle(patients)
    for patient in patients:
        yield patient


class LinearUCB:
    def __init__(self, d, K, alpha):
        self.d = d
        self.K = K
        self.alpha = alpha
        self.A = [np.identity(d) for _ in range(K)]
        self.b = [np.zeros(d) for _ in range(K)]
        self.theta = [np.zeros(d) for _ in range(K)]

    def choose_action(self, x):
        ucb_values = []
        for i in range(self.K):
            self.theta[i] = np.linalg.inv(self.A[i]).dot(self.b[i])
            mean = x.T.dot(self.theta[i])
            uncertainty = self.alpha * np.sqrt(x.T.dot(np.linalg.inv(self.A[i])).dot(x))
            ucb_values.append(mean + uncertainty)
        return np.argmax(ucb_values)

    def update(self, arm, x, reward):
        self.A[arm] += x.dot(x.T)
        self.b[arm] += reward * x


def simulate_bandit(file_path, num_patients, feature_dim, num_actions, alpha):
    generator = patient_generator(file_path)
    bandit = LinearUCB(feature_dim, num_actions, alpha)
    regret = 0
    incorrect_dosing_decisions = 0
    total_reward = 0

    for t in range(num_patients):
        patient_features, therapeutic_dose = next(generator)
        action = bandit.choose_action(patient_features)
        
        if therapeutic_dose < 21:
            best_action = 0
        elif therapeutic_dose <= 49:
            best_action = 1
        else:
            best_action = 2

        reward = -1 if action != best_action else 0
        total_reward += reward
        incorrect_dosing_decisions += (reward == -1)
        regret -= total_reward
        fraction_incorrect = incorrect_dosing_decisions / (t + 1)
        bandit.update(action, patient_features, reward)

    return regret, fraction_incorrect


def run_bandit_multiple_times(file_path, num_patients, feature_dim, num_actions, alpha, num_runs=20):
    regrets = []
    fractions_incorrect = []

    for _ in range(num_runs):

        regret, fraction_incorrect = simulate_bandit(file_path, num_patients, feature_dim, num_actions, alpha)
        regrets.append(regret)
        fractions_incorrect.append(fraction_incorrect)

    # Calculate average performance
    avg_regret = np.mean(regrets)
    avg_fraction_incorrect = np.mean(fractions_incorrect)

    # Calculate 95% confidence intervals
    t_score = stats.t.ppf(1 - 0.025, num_runs - 1)  # T-distribution

    stderr_regret = stats.sem(regrets)
    ci_regret = t_score * stderr_regret

    stderr_fraction_incorrect = stats.sem(fractions_incorrect)
    ci_fraction_incorrect = t_score * stderr_fraction_incorrect

    return (avg_regret, ci_regret), (avg_fraction_incorrect, ci_fraction_incorrect)


def plot_performance(file_path, num_patients, feature_dim, num_actions, alpha, num_runs=20):
    (avg_regret, ci_regret), (avg_fraction_incorrect, ci_fraction_incorrect) = run_bandit_multiple_times(
        file_path, num_patients, feature_dim, num_actions, alpha, num_runs
    )

    fig, ax = plt.subplots(1, 2, figsize=(12, 6))

    # Plot Regret performance
    ax[0].bar(1, avg_regret, yerr=ci_regret, capsize=10)
    ax[0].set_title("Regret Performance")
    ax[0].set_ylabel("Average Regret")
    ax[0].set_xticks([])

    # Plot Fraction Incorrect performance
    ax[1].bar(1, avg_fraction_incorrect, yerr=ci_fraction_incorrect, capsize=10)
    ax[1].set_title("Fraction Incorrect Performance")
    ax[1].set_ylabel("Average Fraction Incorrect")
    ax[1].set_xticks([])

    plt.show()


if __name__ == '__main__':
    # Parameters
    num_patients = 5528
    feature_dim = 31
    num_actions = 3  # Low, medium, high
    alpha = 2
    file_path = "data/warfarin.csv"
    
    plot_performance(file_path, num_patients, feature_dim, num_actions, alpha)

    
