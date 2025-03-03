import pandas as pd

class DataLoader:
    def __init__(self, file_path):
        self.file_path = file_path

    def load_data(self):
        """Load and clean bibliometric data from CSV file"""
        try:
            df = pd.read_csv(self.file_path)
            return self._clean_data(df)
        except Exception as e:
            raise Exception(f"Error loading data: {str(e)}")

    def _clean_data(self, df):
        """Clean and preprocess the data"""
        # Handle missing values
        df['Year'] = pd.to_numeric(df['Year'], errors='coerce')
        df['Cited by'] = pd.to_numeric(df['Cited by'], errors='coerce').fillna(0)

        # Remove rows with critical missing data
        df = df.dropna(subset=['Title', 'Year'])

        return df