import numpy as np
import pandas as pd


def replace_categorical_by_numerical(data):
    data = data.copy()
    data.loc[:, 'Levy'] = data['Levy'].replace({'-': 0})
    data.loc[:, 'Levy'] = pd.to_numeric(data['Levy'])
    
    data.loc[:, 'Engine volume'] = data['Engine volume'].str.replace('Turbo', '')
    data.loc[:, 'Engine volume'] = pd.to_numeric(data['Engine volume'])
    
    data.loc[:, 'Mileage'] = data['Mileage'].str.replace('km', '')
    data.loc[:, 'Mileage'] = pd.to_numeric(data['Mileage'])
    
    return data

def column_transformations(data):
    data['Mileage_log'] = np.log(data['Mileage']).replace(-np.inf, 1e-6)
    data['Levy_log'] = np.log(data['Levy']).replace(-np.inf, 1e-6)
    data['Engine_volume_log'] = np.log(data['Engine volume']).replace(-np.inf, 1e-6)
    
    return data

def clean_outliers(df, cols):
    for col in cols:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        
        df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
        
    return df

def engineer_features(df):
    current_year = pd.Timestamp.now().year
    df['Age'] = current_year - df['Prod. year']
    
    # Add more features
    
    return df

def preprocessing_pipeline(df: pd.DataFrame):
    print("Preprocessing started...")
    print(f"Initial shape: {df.shape}")

    df = df.drop_duplicates()
    print(f"After dropping duplicates: {df.shape}")

    print("Replacing categorical values...")
    df = replace_categorical_by_numerical(df)
    
    df = clean_outliers(df, ['Price', 'Levy', 'Engine volume', 'Mileage']) # clean outliers since we want to predict normal prices (we don't want the model to learn wrong prices)
    print(f"After cleaning outliers: {df.shape}")
    
    # print("Doing column transformations...")
    # df = column_transformations(df)
    
    print("Feature engineering...")
    df = engineer_features(df)
    
    print("Dropping columns...")
    df = df.drop(['ID', 'Doors', 'Prod. year'], axis=1)
    # df = df.drop(['ID', 'Doors', 'Prod. year', 'Levy', 'Mileage', 'Engine volume'], axis=1)
    
    print("Final shape:", df.shape)
    
    return df