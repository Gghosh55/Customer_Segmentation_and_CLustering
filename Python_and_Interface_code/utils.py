import pandas as pd

def clean_data(df):
    # Strip column names to remove extra spaces
    df.columns = df.columns.str.strip()

    # Rename 'Customer ID' to 'CustomerID' for consistency
    if 'Customer ID' in df.columns:
        df = df.rename(columns={'Customer ID': 'CustomerID'})

    # Drop rows where CustomerID is missing
    df = df.dropna(subset=['CustomerID'])

    # Convert 'CustomerID' to integer (handle errors)
    df['CustomerID'] = pd.to_numeric(df['CustomerID'], errors='coerce').astype('Int64')

    # Drop unnecessary columns (non-numeric data)
    columns_to_remove = ['Invoice', 'StockCode', 'Description', 'Country']
    df = df.drop(columns=[col for col in columns_to_remove if col in df.columns])

    # Convert 'Quantity' and 'Price' to numeric
    df['Quantity'] = pd.to_numeric(df['Quantity'], errors='coerce')
    df['Price'] = pd.to_numeric(df['Price'], errors='coerce')

    # Remove rows with NaN values in 'Quantity' or 'Price'
    df = df.dropna(subset=['Quantity', 'Price'])

    # Remove rows with non-positive Quantity or Price
    df = df[(df['Quantity'] > 0) & (df['Price'] > 0)]

    # Convert 'InvoiceDate' to datetime format and handle missing values
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'], errors='coerce')

    # Drop rows where InvoiceDate is null
    df = df.dropna(subset=['InvoiceDate'])

    # Compute Recency (days since last purchase)
    max_date = df['InvoiceDate'].max()
    recency_df = df.groupby('CustomerID')['InvoiceDate'].max().reset_index()
    recency_df['Recency'] = (max_date - recency_df['InvoiceDate']).dt.days

    # Compute Frequency (total number of purchases)
    frequency_df = df.groupby('CustomerID')['InvoiceDate'].count().reset_index()
    frequency_df.columns = ['CustomerID', 'Frequency']

    # Compute MonetaryValue (total spending)
    df['MonetaryValue'] = df['Quantity'] * df['Price']
    monetary_df = df.groupby('CustomerID')['MonetaryValue'].sum().reset_index()

    # Merge Recency, Frequency, and MonetaryValue into one dataset
    rfm_df = recency_df.merge(frequency_df, on='CustomerID').merge(monetary_df, on='CustomerID')

    # Ensure all columns are numeric and drop rows with invalid values
    rfm_df['Recency'] = pd.to_numeric(rfm_df['Recency'], errors='coerce')
    rfm_df['Frequency'] = pd.to_numeric(rfm_df['Frequency'], errors='coerce')
    rfm_df['MonetaryValue'] = pd.to_numeric(rfm_df['MonetaryValue'], errors='coerce')

    # Drop rows with any NaN values that may result from the coercion
    rfm_df.dropna(subset=['Recency', 'Frequency', 'MonetaryValue'], inplace=True)

    # Remove columns with unique values
    for column in rfm_df.columns:
        if rfm_df[column].nunique() == 1:  # if the column has only one unique value
            rfm_df = rfm_df.drop(columns=[column])

    return rfm_df
