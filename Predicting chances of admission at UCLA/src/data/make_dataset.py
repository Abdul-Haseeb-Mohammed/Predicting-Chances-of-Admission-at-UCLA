import pandas as pd

def load_data(data_path):
    df = pd.read_csv(data_path)
    
    return df

def clean_dataset(data):
    
    data['Admit_Chance']=(data['Admit_Chance'] >=0.8).astype(int)
    # Dropping columns
    data = data.drop(['Serial_No'], axis=1)
    print("\n\nDisplaying preview of dataset after converting admit chance to categorical:\n",data.head())
    
    print("\n\nShape of dataset after dropping Serial No:",data.shape)
    
    print("\n\nCheck for missing values and datatypes",data.info())
    
    print("\n\nCheck the summary statistics of the data",data.describe().T)
    return data