import pandas as pd

def fixed_dose(file_path):
    data = pd.read_csv(file_path)
    total_patients = len(data)
    incorrect_decisions = 0

    for index, row in data.iterrows():
        predicted_dose = 35
        therapeutic_dose = row['Therapeutic Dose of Warfarin']

        if therapeutic_dose < 21:
            bucketed_dose = 'low'
        elif therapeutic_dose <= 49:
            bucketed_dose = 'medium'
        else:
            bucketed_dose = 'high'
            
        if bucketed_dose != 'medium':
            incorrect_decisions += 1
            
    return incorrect_decisions / total_patients



def calculate_decades(age_range, median_age_decade):
    if type(age_range) == float:
        return median_age_decade
    lower_age = int(age_range.strip()[0])
    return lower_age // 10


def enzyme_inducer_present(medications, drug_name):
    try:
        drugs_list = medications.lower().split(';')
        for drug in drugs_list:
            drug_info = drug.strip()
            if drug_name in drug_info and 'not' not in drug_info:
                return True
        return False
    except: print(medications)



def wcda(file_path):
    data = pd.read_csv(file_path)
    total_patients = len(data)
    incorrect_decisions = 0

    for index, row in data.iterrows():
       
        age_decades = data['Age'].apply(lambda x: int(x.strip()[0]) // 10 if type(x) != float else None).dropna()
        median_age_decade = int(age_decades.median())
        age_decades = calculate_decades(row['Age'], median_age_decade)
        
        height_cm = row['Height (cm)']
        weight_kg = row['Weight (kg)']
      
        race = row['Race']
        asian_race = 1 if race == 'Asian' else 0
        black_race = 1 if race == 'Black or African American' else 0
        missing_mixed_race = 1 if race not in ['White', 'Asian', 'Black or African American'] else 0
        
        medications = row['Medications']
        if type(medications) == float: medications = 'NA'
       
        amiodarine_status = row['Amiodarone (Cordarone)']
        amiodarone_status = amiodarine_status == 1 or enzyme_inducer_present(medications, 'amiodarone') or enzyme_inducer_present(medications, 'cordarone')
        
        carbamazepine_status = row['Carbamazepine (Tegretol)']
        phenytoin_status = row['Phenytoin (Dilantin)']
        rifampin_status = row['Rifampin or Rifampicin']
        carbamazepine_present = carbamazepine_status == 1 or enzyme_inducer_present(medications, 'carbamazepine') or enzyme_inducer_present(medications, 'tegretol')
        phenytoin_present = phenytoin_status == 1 or enzyme_inducer_present(medications, 'phenytoin') or enzyme_inducer_present(medications, 'dilantin')
        rifampin_present = rifampin_status == 1 or enzyme_inducer_present(medications, 'rifampin') or enzyme_inducer_present(medications, 'rifampicin')
        enzyme_inducer_status = 1 if carbamazepine_present or phenytoin_present or rifampin_present else 0
        
        
        weekly_dose_sqrt = (
            4.0376 -
            0.2546 * age_decades +
            0.0118 * height_cm +
            0.0134 * weight_kg -
            0.6752 * asian_race +
            0.4060 * black_race +
            0.0443 * missing_mixed_race +
            1.2799 * enzyme_inducer_status -
            0.5695 * amiodarone_status
        )
        
        predicted_dose = weekly_dose_sqrt ** 2
        therapeutic_dose = row['Therapeutic Dose of Warfarin']

        if therapeutic_dose < 21:
            bucketed_dose = 'low'
        elif therapeutic_dose <= 49:
            bucketed_dose = 'medium'
        else:
            bucketed_dose = 'high'

        if (predicted_dose < 21 and bucketed_dose != 'low') or (21 <= predicted_dose <= 49 and bucketed_dose != 'medium') or (predicted_dose > 49 and bucketed_dose != 'high'):
            incorrect_decisions += 1

    performance = incorrect_decisions / total_patients
    return performance

if __name__ == '__main__':
    file_path = 'data/warfarin.csv'
    performance1 = fixed_dose(file_path)
    performance2 = wcda(file_path)
    print(f"Fixed: {performance1}, WCDA: {performance2}")
    