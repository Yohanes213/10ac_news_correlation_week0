import pandas as pd

def load_csv_file(filepath):
  """Loads a CSV file into a pandas DataFrame"""
  return pd.read_csv(filepath)