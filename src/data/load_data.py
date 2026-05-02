import pandas as pd

def load_data(file_path:str) -> pd.DataFrame:
    df = pd.read_csv(file_path)
    df = df.ffill()
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    return df