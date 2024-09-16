import pytest
import pandas as pd
import logging
import os
from unittest import mock
from unittest.mock import patch
import numpy as np

# Copying all relevant functions for the unit tests from 'Crime pipeline python.py'

def drop_na_rows(df, column):
    """
    Drops rows from a DataFrame that contain NaN values in the specified column.
    
    Parameters:
    df (pd.DataFrame): The DataFrame to process.
    column (str): The column to check for NaN values.
    
    Returns:
    pd.DataFrame: The DataFrame with rows containing NaN values in the specified column dropped.
    """
    try:
        df = df.dropna(subset=column)
    except Exception as e:
        logging.error(f'An error occured in {drop_na_rows.__name__}: {e}')
    return df

def replace_nulls(df, column, value):
    """
    Replaces null values in the specified column with 'No Data'.
    
    Parameters:
    df (pd.DataFrame): The DataFrame to modify.
    column (str): The column in which null values will be replaced.
    
    Returns:
    pd.DataFrame: The DataFrame with null values in the specified column replaced.
    """
    try:
        df[column] = df[column].fillna(value)
    except Exception as e:
        logging.error(f'An error occured in {replace_nulls.__name__}: {e}')
    return df

def round_values(df, col, decimals):
    """
    Change the datatype of the specified column to float and round the values to the specified number of decimal places.

    Parameters:
    df (pd.DataFrame): The DataFrame containing the column.
    col (str): The name of the column to process.
    decimals (int): The number of decimal places to round to.

    Returns:
    pd.DataFrame: The DataFrame with the rounded values in the specified column.
    """
    try:
        # Change the datatype to float
        df[col] = df[col].astype(float)
        # Round the values
        df[col] = df[col].round(decimals)
    except Exception as e:
        logging.error(f'An error occurred in {round_values.__name__}: {e}')
    return df

def merge_files_on(df_1, df_2, join_type, left_on, right_on):
    """
    Merges two DataFrames based on specified columns and join type.
    
    Parameters:
    df_1 (pd.DataFrame): The first DataFrame to merge.
    df_2 (pd.DataFrame): The second DataFrame to merge.
    join_type (str): The type of join to perform ('left', 'right', 'outer', 'inner').
    left_on (str): Column in the first DataFrame to join on.
    right_on (str): Column in the second DataFrame to join on.
    
    Returns:
    pd.DataFrame: The merged DataFrame.
    """
    try:
        df = pd.merge(left=df_1, right=df_2, how=join_type, left_on=left_on, right_on=right_on)
    except Exception as e:
        logging.error(f'An error occurred in {merge_files_on.__name__}: {e}')
        df = pd.DataFrame()  # Return an empty DataFrame in case of error
    return df

def output_to_csv(df, output_file):
    """
    Outputs a DataFrame to a CSV file.
    
    Parameters:
    df (pd.DataFrame): The DataFrame to write to a CSV file.
    output_file (str): The filename for the output CSV file.
    
    Returns:
    pd.DataFrame: The original DataFrame.
    """

    if isinstance(df, pd.DataFrame):  # Ensure that the object is a DataFrame
        try:
            df.to_csv(output_file, index=False)
            logging.info(f'Successfully output to csv as file name: {output_file}')
        except Exception as e:
            logging.error(f'An error occurred in {output_to_csv.__name__}: {e}')
    else:
        logging.error(f'Expected DataFrame but got {type(df).__name__}')
    return df

def replace_col_values(df, col, replace_dict):
    """
    Replaces values in the specified column based on a dictionary of replacement values.
    
    Parameters:
    df (pd.DataFrame): The DataFrame to modify.
    col (str): The column to apply the replacement.
    replace_dict (dict): A dictionary where keys are the values to replace, and values are the new values.
    
    Returns:
    pd.DataFrame: The DataFrame with values in the specified column replaced.
    """
    try:
        df[col] = df[col].replace(replace_dict)
    except Exception as e:
        logging.error(f'An error occured in {replace_col_values.__name__}: {e}')
    return df

def drop_col(df, columns):
    """
    Drops the specified columns from the DataFrame.
    
    Parameters:
    df (pd.DataFrame): The DataFrame to modify.
    columns (list): List of column names to drop.
    
    Returns:
    pd.DataFrame: The DataFrame with the specified columns removed.
    """
    try:
        # Drop columns that exist in the DataFrame
        columns_to_drop = [col for col in columns if col in df.columns]
        df = df.drop(columns=columns_to_drop, axis=1)
    except Exception as e:
        logging.error(f"An error occurred in {drop_col.__name__}: {e}")
    return df

def date_convert_dtype(df, columns):
    """
    Converts the specified columns to datetime type, coercing invalid parsing to NaT (Not a Time).
    
    Parameters:
    df (pd.DataFrame): The DataFrame to modify.
    columns (list): The list of column names to convert.
    
    Returns:
    pd.DataFrame: The DataFrame with the specified columns converted to datetime type.
    """     
    try:
        for column in columns:
            df[column] = pd.to_datetime(df[column], errors='coerce')
    except Exception as e:
        logging.error(f"Error converting dates: {e}")
    return df

def change_text_to_proper_case(df, cols_to_change):
    """
    Converts the text in specified columns to proper case (title case).
    
    Parameters:
    df (pd.DataFrame): The DataFrame to modify.
    
    Returns:
    pd.DataFrame: The DataFrame with text in specified columns converted to proper case.
    """
    try:
        for col in cols_to_change:
            if col in df.columns:
                df[col] = df[col].str.title()
            else:
                logging.warning(f'Column {col} not found in DataFrame.')
    except Exception as e:
        logging.error(f'An error occurred in {change_text_to_proper_case.__name__}: {e}')
    return df

def split_postcode(df, postcode_column, new_column):
    """
    Splits the values in the specified postcode column, extracting the first part of the postcode
    (i.e., the outward code) and adds it to a new column.

    Parameters:
    df (pd.DataFrame): The DataFrame to modify.
    postcode_column (str): The name of the column containing the full postcode.
    new_column (str): The name of the new column to store the extracted outward code.

    Returns:
    pd.DataFrame: The DataFrame with the new column containing the outward code of the postcode.
    """
    df[new_column] = df[postcode_column].str.split(' ').str[0].str.strip()
    return df


def check_and_create_folders(base_dir):
    """
    First checks if output folder 'crime_data_outputs' is in directory with nested folder 'aggregated_dataframes'
    If not, creates these folders

    Parameters:
    base_dir: variable of directory location

    """
    crime_data_outputs = os.path.join(base_dir, "crime_data_outputs")
    aggregated_dataframes = os.path.join(crime_data_outputs, "aggregated_dataframes")
    
    # Check if the "crime_data_outputs" folder exists, if not, create it
    if not os.path.exists(crime_data_outputs):
        os.makedirs(crime_data_outputs)
        logging.info(f"Created folder: {crime_data_outputs}")
    
    # Check if the "aggregated_dataframes" folder exists inside "crime_data_outputs", if not, create it
    if not os.path.exists(aggregated_dataframes):
        os.makedirs(aggregated_dataframes)
        logging.info(f"Created folder: {aggregated_dataframes}")




# Defining fixtures to pass into test functions 
@pytest.fixture
def street_sample_data():
    data = {
    'Falls within':['Merseyside Police', None, 'Merseyside Police', 'Nottinghamshire Police', None],
    'Longitude':[-2.743622, -2.857463, None, -2.475932, -2.675829],
    'Latitude':[None, 53.492847, 53.383772, 53.996882, -53.192333],
    'LSOA code':[None, None, 'E03034823', 'E38477746', 'E29943822'],
    'Floats':[13.13928, 45.22243, 6.92827, 6.66888, 9.11633],
    'Dates':['01/01/1998', '02/02/2001', '05/02/2013', '02/09/2017', '12/12/2012'],
    'Postcode':['HG1 74B', 'A91 84B', 'FD1 2RP', 'DR21 9TR', 'PT77 4BH']
    }
    return pd.DataFrame(data)

@pytest.fixture
def outcome_sample_data():
    data = {
    'Test':['A', 'C', 'F', 'Z', 'G'],
    'Floats':[13.13928, 45.22243, 6.92827, 6.66888, 9.11633]
    }
    return pd.DataFrame(data)




# Parameterized unit testing 
@pytest.mark.parametrize('column,expected_len', [
    ('Latitude', 4),
    ('Longitude', 4),
    ('Falls within', 3)
])
def test_drop_na_rows(street_sample_data, column, expected_len):
    result = drop_na_rows(street_sample_data, column)
    assert len(result) == expected_len


@pytest.mark.parametrize('column, expected_value', [
    ('Falls within', 'Unknown'),
    ('Longitude', 0.0),
    ('Latitude', 0.0),
    ('LSOA code', 'Unknown'),
    ('Floats', 0.0)
])
def test_replace_nulls(street_sample_data, column, expected_value):
    result = replace_nulls(street_sample_data, column, expected_value)
    assert result[column].isnull().sum() == 0


@pytest.mark.parametrize('column', ['Falls within'])
def test_change_text_to_proper_case(street_sample_data, column):
    result = change_text_to_proper_case(street_sample_data, column)
    assert result[column].str.istitle().sum() == result[column].notna().sum()




# Fixture based unit testing
def test_merge_files_on(street_sample_data, outcome_sample_data):
    result = merge_files_on(street_sample_data, outcome_sample_data, join_type='inner', left_on='Floats', right_on='Floats')
    expected_data = {
        'Falls within':['Merseyside Police', None, 'Merseyside Police', 'Nottinghamshire Police', None],
        'Longitude':[-2.743622, -2.857463, None, -2.475932, -2.675829],
        'Latitude':[None, 53.492847, 53.383772, 53.996882, -53.192333],
        'LSOA code':[None, None, 'E03034823', 'E38477746', 'E29943822'],
        'Floats':[13.13928, 45.22243, 6.92827, 6.66888, 9.11633],
        'Dates':['01/01/1998', '02/02/2001', '05/02/2013', '02/09/2017', '12/12/2012'],
        'Postcode':['HG1 74B', 'A91 84B', 'FD1 2RP', 'DR21 9TR', 'PT77 4BH'],
        'Test': ['A', 'C','F', 'Z', 'G']
    }
    expected_df = pd.DataFrame(expected_data)
    pd.testing.assert_frame_equal(result.reset_index(drop=True), expected_df.reset_index(drop=True))


def test_split_postcode(street_sample_data):
    result = split_postcode(street_sample_data, 'Postcode', 'Split postcode')
    assert 'Split postcode' in result 

def test_split_postcode_space(street_sample_data):
    result = split_postcode(street_sample_data, 'Postcode', 'Split postcode')
    assert not result['Split postcode'].str.contains(' ').any()


def test_merge_files_on_rowcheck(street_sample_data, outcome_sample_data):
    result = merge_files_on(street_sample_data, outcome_sample_data, join_type='inner', left_on='Floats', right_on='Floats')
    assert len(street_sample_data['Falls within']) == len(result['Falls within'])


def test_round_values(street_sample_data):
    result = round_values(street_sample_data, 'Floats', 3)
    expected_floats = [13.139, 45.222, 6.928, 6.669, 9.116]
    assert result['Floats'].tolist() == expected_floats


def test_output_to_csv(tmp_path, street_sample_data):
    output_file = tmp_path / "output.csv"
    output_to_csv(street_sample_data, output_file)
    assert output_file.exists()


def test_drop_col(street_sample_data):
    result = drop_col(street_sample_data, ['Floats'])
    assert 'Floats' not in result.columns

def test_date_convert_dtype(street_sample_data):
    result = date_convert_dtype(street_sample_data, ['Dates'])
    assert result['Dates'].dtype == np.dtype('datetime64[ns]')



# Unit tests including mocking
"""
Test function verifies that the output_to_csv function correctly writes a dataframe to csv
"""
def test_output_to_csv(street_sample_data):
    with mock.patch('pandas.DataFrame.to_csv') as mock_to_csv:
        result = output_to_csv(street_sample_data, 'dummy_file.csv')
        mock_to_csv.assert_called_once_with('dummy_file.csv', index=False)
        assert result.equals(street_sample_data)

"""
Test function verifies that the replace_col_values function handles exceptions correctly when replacing values in a DataFrame column
"""
def test_replace_col_values_exception(street_sample_data):
    replace_dict = {'Merseyside Police': 'Mer'}
    with mock.patch.object(street_sample_data['Falls within'], 'replace', side_effect=Exception('Mocked exception')) as mock_replace:
        with mock.patch('logging.error') as mock_log_error:
            result = replace_col_values(street_sample_data, 'Falls within', replace_dict)
            mock_replace.assert_called_once_with(replace_dict)
            mock_log_error.assert_called_once_with('An error occured in replace_col_values: Mocked exception')
            expected_data = {
                'Falls within': ['Merseyside Police', None, 'Merseyside Police', 'Nottinghamshire Police', None],
                'Longitude': [-2.743622, -2.857463, None, -2.475932, -2.675829],
                'Latitude': [None, 53.492847, 53.383772, 53.996882, -53.192333],
                'LSOA code': [None, None, 'E03034823', 'E38477746', 'E29943822'],
                'Floats': [13.13928, 45.22243, 6.92827, 6.66888, 9.11633],
                'Dates':['01/01/1998', '02/02/2001', '05/02/2013', '02/09/2017', '12/12/2012'],
                'Postcode':['HG1 74B', 'A91 84B', 'FD1 2RP', 'DR21 9TR', 'PT77 4BH']
            }
            expected_df = pd.DataFrame(expected_data)
            pd.testing.assert_frame_equal(result, expected_df)


"""
Test function check_and_create_folders to see if folder is created for 
"""
@patch('os.makedirs')  
@patch('os.path.exists')  
def test_check_and_create_folders_crime_data_outputs(mock_path_exists, mock_makedirs):
    base_dir = '/test/directory'
    crime_data_outputs = os.path.join(base_dir, 'crime_data_outputs')
    aggregated_dataframes = os.path.join(crime_data_outputs, 'aggregated_dataframes')
    
    # Mock behavior for os.path.exists
    # Return False for crime_data_outputs and True for aggregated_dataframes
    mock_path_exists.side_effect = lambda path: {
        crime_data_outputs: False, 
        aggregated_dataframes: True  
    }.get(path, False)

    check_and_create_folders(base_dir)

    # Assert os.makedirs is called only for the crime_data_outputs folder
    mock_makedirs.assert_called_once_with(crime_data_outputs)