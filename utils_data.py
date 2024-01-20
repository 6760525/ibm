#!/usr/bin/env python
"""
Collection of Data Loading Functions

This script contains a set of functions designed to load, process, and manipulate data. These functions are tailored to handle various data formats and structures, ensuring efficient and effective data handling for further analysis or processing.

Functions included:
- load_json_files_to_dataframe(data_dir): Loads JSON formatted files from a specified directory into a single Pandas DataFrame.

Note:
Ensure that necessary libraries (like pandas, numpy) are installed before running this script.

Author: Alexey Tyurin
Date: 1/19/2024
"""

import time, os, re, shutil, random
import pandas as pd


TS_DIR = 'ts-data'
TRAIN_DATA_DIR = 'cs-train'
TOP_COUNTRIES = 10

def load_json_files_to_dataframe(data_dir):
    """
    Load all JSON formatted files from a specified directory into a single Pandas DataFrame.
    
    Parameters:
    data_dir (str): The directory containing JSON files.
    
    Returns:
    pd.DataFrame: A DataFrame containing the combined data from all JSON files.
    
    Raises:
    Exception: If the specified directory does not exist or contains no JSON files.
    """

    # Check if the specified directory exists
    if not os.path.isdir(data_dir):
        raise Exception(f"The specified directory '{data_dir}' does not exist.")

    # Filter for JSON files in the directory
    
    json_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.json')]

    if not json_files:
        raise Exception(f"No JSON files found in the directory '{data_dir}'.")

    # Correct column names
    correct_columns = ['country', 'customer_id', 'day', 'invoice_id', 'month', 'revenue', 'stream_id', 'times_viewed', 'year']
    rename_map = {'StreamID': 'stream_id', 'TimesViewed': 'times_viewed', 'total_price': 'revenue', 'invoice': 'invoice_id', 'price': 'revenue'}

    all_files = {}
    for file_name in json_files:
        df = pd.read_json(file_name)
        df.rename(columns=rename_map, inplace=True)
        if sorted(df.columns) != correct_columns:
            raise Exception(f"Columns in {file_name} do not match the required structure.")
        all_files[file_name] = df

    df = pd.concat(all_files.values(), sort=True)
    df['invoice_date'] = pd.to_datetime(df[['year', 'month', 'day']])
    df['customer_id'] = df['customer_id'].apply(lambda x: '' if pd.isna(x) else str(int(x)))

    df = df.astype({'customer_id': 'category',
                    'invoice_id': 'category',
                    'stream_id': 'category'})
    
    df = df[['country', 'invoice_date', 'customer_id', 'invoice_id', 'stream_id', 'times_viewed', 'revenue']] \
        .sort_values(by='invoice_date') \
        .reset_index(drop=True)

    return df


def _format_elapsed_time(start_time):
    """ Calculate and format elapsed time. """
    elapsed_time = time.time() - start_time
    hours, remainder = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    return "%d:%02d:%02d" % (hours, minutes, seconds)

def _clean_directory(directory):
    """ Remove all contents of a directory. """
    shutil.rmtree(directory)
    os.mkdir(directory)

def _load_processed_ts_data(ts_data_dir):
    """ Load processed time series data from CSV files. """
    return {re.sub(r"\.csv", "", filename)[3:]:
            pd.read_csv(os.path.join(ts_data_dir, filename))
            for filename in os.listdir(ts_data_dir)}

def _save_ts_data(dfs, ts_data_dir):
    """ Save time series data as CSV files. """
    for key, item in dfs.items():
        item.to_csv(os.path.join(ts_data_dir, f"ts-{key}.csv"), index=False)

def _convert_to_ts(df, country=None):
    """
    Convert the original clean dataframe to a time series
    by aggregating over each day for the given country.
    """
    
    # Group by necessary columns and aggregate
    group_cols = ['country', 'invoice_date'] if country else ['invoice_date']
    agg_funcs = {
        'purchases': ('invoice_id', 'size'),
        'unique_invoices': ('invoice_id', 'nunique'),
        'unique_streams': ('stream_id', 'nunique'),
        'total_views': ('times_viewed', 'sum'),
        'revenue': ('revenue', 'sum')
    }
    ts_data = df.groupby(group_cols).agg(**agg_funcs).reset_index()

    # Handle country-specific data
    if country:
        ts_data = ts_data[ts_data['country'] == country].drop(['country'], axis=1)

    date_range = pd.date_range(ts_data['invoice_date'].min(), ts_data['invoice_date'].max(), freq='1D')
    complete_df = pd.DataFrame(data=date_range, columns=['invoice_date'])
    ts_data = complete_df.merge(ts_data, on=['invoice_date'], how='left').fillna(0)

    if country == None:
        ts_data = ts_data.groupby('invoice_date').sum().reset_index()


    ts_data['invoice_date'] = pd.to_datetime(ts_data['invoice_date'])
    ts_data = ts_data.astype({'purchases': 'int',
                              'unique_invoices': 'int',
                              'unique_streams': 'int',
                              'total_views': 'int',
                              'revenue': 'float'})

    return ts_data

def _process_ts_data(df, top_n=10):
    """ Process and get time series data for the top N countries. """
    top_countries = (df.groupby('country')['revenue'].sum()
                     .nlargest(top_n)
                     .index.tolist())
    dfs = {'all': _convert_to_ts(df)}
    for country in top_countries:
        country_id = re.sub(r'\s+', '_', country.lower())
        dfs[country_id] = _convert_to_ts(df, country=country)
    return dfs

def fetch_ts(data_dir, clean=False):
    """
    Convenience function to read in new time series data.
    It loads data from existing CSV files or processes new data if CSV files are not present.
    
    Parameters:
    - data_dir (str): Directory containing the raw data.
    - clean (bool): If True, existing time series data will be re-created.

    Returns:
    - dict: A dictionary of DataFrames containing the time series data.
    """

    ts_data_dir = os.path.join(data_dir, TS_DIR)

    # Clean the directory if requested or create it if it doesn't exist
    if clean:
        print("... cleaning existing time series data")
        _clean_directory(ts_data_dir)
    elif not os.path.exists(ts_data_dir):
        print("... creating time series data directory")
        os.mkdir(ts_data_dir)

    # Load processed data if available
    if os.listdir(ts_data_dir):
        print("... loading time series data from files")
        return _load_processed_ts_data(ts_data_dir)

    # Process and save new data if no processed data is found
    print("... no processed data found, processing new data")
    df = load_json_files_to_dataframe(data_dir)
    dfs = _process_ts_data(df, top_n=TOP_COUNTRIES)
    _save_ts_data(dfs, ts_data_dir)
    return dfs

def engineer_rolling_features(X, shift_days, attributes, func='sum', engineer_target=False):
    """
    Engineer features and/or target based on a rolling window.

    Parameters:
    X (DataFrame): Input data frame.
    shift_days (int): Number of days for rolling window.
    attributes (list): List of column names to be transformed.
    func (str): Function to apply in rolling window ('sum' or 'mean').
    engineer_target (bool): If True, shift the result to engineer target.

    Returns:
    DataFrame: Transformed data frame.
    """
    
    X_indexed = X.set_index('invoice_date')
    freq = f'{shift_days}D'

    if func == 'sum':
        X_transformed = X_indexed[attributes].rolling(freq, closed='left').sum()
    else:
        X_transformed = X_indexed[attributes].rolling(freq, closed='left').mean()

    if engineer_target:
        X_transformed = X_transformed.shift(-shift_days)

    suffix = f'_p{freq}' if engineer_target else f'_m{freq}'
    X_transformed = X_indexed.merge(X_transformed, 
                                    left_index=True,
                                    right_index=True,
                                    how='left',
                                    suffixes=['', suffix]).fillna(0).reset_index()

    return X_transformed


if __name__ == '__main__':
    run_start = time.time()
    data_dir = os.path.join('.', TRAIN_DATA_DIR)
    print('Loading data...')

    df = fetch_ts(data_dir, clean=True)

    for country, ddf in df.items():
        print(f'\t{country}: {ddf.shape}')

    print('\t' + '-' * 20)
    print(f'Loading done. Data load time: {_format_elapsed_time(run_start)}\n')
    random_key = random.choice(list(df.keys()))
    ds = df[random_key]
    print(f'\nSample Data ({random_key}):')
    print(ds.head())
    print('\nDataFrame info:')
    print(ds.info())

    
    features = ds.columns.tolist()[2:-1]
    all_features = []

    # Engineer features
    for shift in [7, 14, 28, 35, 54]:
        ds = engineer_rolling_features(ds, shift, ['revenue'])
    ds = engineer_rolling_features(ds, 30, features, func='mean')

    # Engineer target
    ds = engineer_rolling_features(ds, 30, ['revenue'], engineer_target=True)

    # Drop original features and keep only engineered ones
    engineered_cols = [col for col in ds.columns if '_m' in col or '_p' in col]
    ds = ds[['invoice_date'] + engineered_cols]

    all_features.append(ds)

    ds = pd.concat(all_features)
    
    print('Engineered dataset:')
    print(ds.head())
