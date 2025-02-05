import pandas as pd


def validate_and_load_file(file_path):
    try:
        df = pd.read_csv(file_path)

        # Strip spaces from column names
        df.columns = df.columns.str.strip()

        # Check if 'InvoiceDate' exists
        if 'InvoiceDate' not in df.columns:
            raise ValueError("InvoiceDate column is missing from the dataset.")

        # Ensure 'InvoiceDate' has valid datetime values
        df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'], errors='coerce')
        if df['InvoiceDate'].isnull().any():
            raise ValueError("Invalid or missing datetime values in InvoiceDate column.")

        return df
    except Exception as e:
        raise ValueError(f"Error during processing: {str(e)}")
