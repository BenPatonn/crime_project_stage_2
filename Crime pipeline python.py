import os
import pandas as pd
import logging

LOG_FILE = 'LOG_FILE.txt'
logging.basicConfig(
    filename=LOG_FILE,
    filemode='a',
    format='%(asctime)s %(levelname)s:%(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.INFO
)

directory_path = 'crime_datasets/'

def load_csvs_to_list(directory_path, keyword):
    """
    Loads all CSV files in the given directory and its subdirectories into a list of DataFrames,
    filtering for files with the specified keyword in the filename.
    
    Parameters:
    directory_path (str): The path to the main folder containing subfolders with CSV files.
    keyword (str): The keyword to filter filenames (e.g., 'street', 'outcomes').
    
    Returns:
    list: A list of DataFrames for files containing the keyword in the filename.
    """
    
    df_list = []
    
    try:
        for root, dirs, files in os.walk(directory_path):
            for file in files:
                if file.endswith('.csv'):
                    file_path = os.path.join(root, file)
                    try:
                        df = pd.read_csv(file_path, index_col=None, header=0)
                        if isinstance(df, pd.DataFrame):
                            if keyword in file.lower():  # Check for 'street' in filename (case insensitive)
                                df_list.append(df)
                        else:
                            logging.error(f"File {file_path} could not be read as DataFrame.")
                    except Exception as e:
                        logging.error(f"Error reading {file_path}: {e}")
    except Exception as e:
        logging.error(f'An error occurred in {load_csvs_to_list.__name__}: {e}')
    
    return df_list


def load_csvs_with_area(directory_path, keyword):
    """
    Loads CSV files from a directory, adds an 'Area' column based on the filename, 
    and returns a single concatenated DataFrame containing the specified keyword.
    
    Parameters:
    directory_path (str): Path to the directory containing subfolders with CSV files.
    keyword (str): The keyword to filter filenames.
    
    Returns:
    pd.DataFrame: A concatenated DataFrame with the 'Area' column based on the filename.
    """
    df_list = []
    area_mapping = {
        'merseyside': 'merseyside',
        'nottinghamshire': 'nottinghamshire'
    }
    
    try:
        # Walk through the directory and its subdirectories
        for root, dirs, files in os.walk(directory_path):
            for file in files:
                if file.endswith('.csv') and keyword in file.lower():
                    # Construct full file path
                    file_path = os.path.join(root, file)
                    
                    # Read CSV into DataFrame
                    df = pd.read_csv(file_path, encoding='latin1', index_col=None, header=0)
                    
                    # Check the filename for any of the area keywords
                    for area_keyword, area in area_mapping.items():
                        if area_keyword in file.lower():
                            # Add 'Area' column with the corresponding area value
                            df['Area'] = area
                            df_list.append(df)
                            break
    
    except Exception as e:
        logging.error(f'An error occurred in {load_csvs_with_area.__name__}: {e}')
    
    # Concatenate all DataFrames in the list into a single DataFrame
    if df_list:
        df = pd.concat(df_list, ignore_index=True)
    else:
        df = pd.DataFrame()  # Return an empty DataFrame if the list is empty
    return df


def concat_df(df_list):
    """
    Concatenates a list of DataFrames into a single DataFrame.
    
    Parameters:
    df_list (list of pd.DataFrame): A list of DataFrames to concatenate.
    
    Returns:
    pd.DataFrame: A single concatenated DataFrame.
    """
    try:
        df = pd.concat(df_list, axis=0, ignore_index = True)
    except Exception as e:
        logging.error(f'An error occured in {concat_df.__name__}: {e}')
    return df

def count_nulls_in_columns(df):
    """
    Counts the number of null values in each column of a DataFrame.
    
    Parameters:
    df (pd.DataFrame): The DataFrame to analyze.
    
    Returns:
    pd.Series: A series showing the count of null values for each column.
    """
    null_counts = df.isnull().sum()
    print("Null values in each column:\n")
    print(null_counts)
    return null_counts

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

col_to_fill = ['Month_x','Reported by_x', 'Falls within_x', 'Longitude_x',
                                  'Latitude_x', 'Location_x', 'LSOA code_x', 'LSOA_name']

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

def replace_nulls(df, column):
    """
    Replaces null values in the specified column with 'No Data'.
    
    Parameters:
    df (pd.DataFrame): The DataFrame to modify.
    column (str): The column in which null values will be replaced.
    
    Returns:
    pd.DataFrame: The DataFrame with null values in the specified column replaced.
    """
    try:
        df[column] = df[column].fillna('No Data')
    except Exception as e:
        logging.error(f'An error occured in {replace_nulls.__name__}: {e}')
    return df

def trim_strings_col(df, column):
    """
    Trims leading and trailing whitespaces from string values in the specified column.
    
    Parameters:
    df (pd.DataFrame): The DataFrame to modify.
    column (str): The column to apply the string trimming.
    
    Returns:
    pd.DataFrame: The DataFrame with the specified column's string values trimmed.
    """
    try:
        column in df.columns
        df[column] = df[column].str.strip()
    except Exception as e: 
        logging.error(f'An error occured in {trim_strings_col.__name__}: {e}')
    return df

def filter_crime_types(df, crime_types):
    """
    Filters the DataFrame to include only rows where 'Crime type' matches the specified crime types.
    
    Parameters:
    df (pd.DataFrame): The DataFrame to filter.
    crime_types (list): List of crime types to filter by.
    
    Returns:
    pd.DataFrame: The filtered DataFrame.
    """
    try:
        df = df[df['Crime type'].isin(crime_types)]
    except Exception as e:
        logging.error(f'An error occured in {filter_crime_types.__name__}: {e}')
    return df

def categorize_outcome(df):
    """
    Categorizes the 'Outcome type' column into broad categories and adds a new column 'Broad Outcome Category'.
    
    Parameters:
    df (pd.DataFrame): The DataFrame to modify.
    
    Returns:
    pd.DataFrame: The DataFrame with the 'Broad Outcome Category' column added.
    """
    def categorize(outcome):
        if outcome in ['Unable to prosecute suspect', 'Investigation complete; no suspect identified', 'Status update unavailable']:
            return 'No Further Action'
        elif outcome in ['Local resolution', 'Offender given a caution', 'Action to be taken by another organisation']:
            return 'Non-criminal Outcome'
        elif outcome in ['Further investigation is not in the public interest', 'Further action is not in the public interest', 'Formal action is not in the public interest']:
            return 'Public Interest Consideration'
        elif outcome in ['Suspect charged', 'Offender given a drugs possession warning', 'Suspect charged as part of another case', 'Offender given penalty notice']:
            return 'Criminal action taken'
        else:
            return 'No data'
            
    df['Broad Outcome Category'] = df['Outcome type'].apply(categorize)
    
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

def add_area_col(df, area_name):
    """
    Adds an 'Area' column to the DataFrame with a specified area name.
    
    Parameters:
    df (pd.DataFrame): The DataFrame to modify.
    area_name (str): The value to set for the 'Area' column.
    
    Returns:
    pd.DataFrame: The DataFrame with the 'Area' column added.
    """
    try:
        df['Area'] = area_name
    except Exception as e:
        logging.error(f'An error occured in {add_area_col.__name__}: {e}')
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
            df[column] = pd.to_datetime(df[column], errors='coerce')  # Coerce invalid parsing to NaT
    except Exception as e:
        logging.error(f"Error converting dates: {e}")
    return df

def change_col_names(df_list):
    """
    Changes the column names of DataFrames in the list to standard column names.
    
    Parameters:
    df_list (list of pd.DataFrame): A list of DataFrames to modify.
    
    Returns:
    list: A list of DataFrames with updated column names.
    """
    new_column_names = [
        'Transaction unique identifier', 'Price', 'Date of transfer', 'Postcode', 'Property type', 
        'Old/new', 'Duration', 'PAON', 'SAON', 'Street', 'Locality', 'Town/city', 'District', 
        'County', 'PPD category type', 'Record status'
    ]

    processed_dfs = []
    for df in df_list:
        if isinstance(df, pd.DataFrame):
            try:
                if len(df.columns) == len(new_column_names):
                    df.columns = new_column_names
                else:
                    logging.warning(f'Column count mismatch: Expected {len(new_column_names)}, but got {len(df.columns)}')
                processed_dfs.append(df)
            except Exception as e:
                logging.error(f'An error occurred in change_col_names: {e}')
                processed_dfs.append(df)
        else:
            logging.error(f'Expected DataFrame but got {type(df).__name__}')
            processed_dfs.append(None)

    return processed_dfs

cols_to_change = ['Street', 'Locality', 'Town/city', 'District', 'County']
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

def rename_cols(df, col_dict):
    """
    Renames the columns of the DataFrame based on a dictionary of new column names.
    
    Parameters:
    df (pd.DataFrame): The DataFrame to modify.
    col_dict (dict): A dictionary mapping old column names to new column names.
    
    Returns:
    pd.DataFrame: The DataFrame with renamed columns.
    """
    try:
        df = df.rename(columns=col_dict)
    except Exception as e:
        logging.error(f'An error occured in {rename_cols.__name__}: {e}')
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

def drop_duplicate_rows(df, col_a, col_b):
    """
    Drop rows from the DataFrame where both col_a and col_b have duplicated values.

    Parameters:
    df (pd.DataFrame): The DataFrame to process.
    col_a (str): The name of the first column to check for duplicates.
    col_b (str): The name of the second column to check for duplicates.

    Returns:
    pd.DataFrame: The DataFrame with the duplicated rows removed.
    """
    try:
        mask = df.duplicated(subset=[col_a, col_b], keep=False)
        df = df[~mask]
    except Exception as e:
        logging.error(f'An error occured in {drop_duplicate_rows.__name__}: {e}')
    return df

def merge_on_two_columns(df_1, df_2, col_1_df1, col_2_df1, col_1_df2, col_2_df2, join_type='inner'):
    """
    Merge two DataFrames based on matches in two columns from each DataFrame.

    Parameters:
    df_1 (pd.DataFrame): The first DataFrame to merge.
    df_2 (pd.DataFrame): The second DataFrame to merge.
    col_1_df1 (str): The name of the first column in the first DataFrame to join on.
    col_2_df1 (str): The name of the second column in the first DataFrame to join on.
    col_1_df2 (str): The name of the first column in the second DataFrame to join on.
    col_2_df2 (str): The name of the second column in the second DataFrame to join on.
    join_type (str): The type of join to perform ('left', 'right', 'outer', 'inner'). Default is 'inner'.

    Returns:
    pd.DataFrame: The merged DataFrame.
    """
    try:
        df = pd.merge(df_1, df_2, how=join_type, left_on=[col_1_df1, col_2_df1], right_on=[col_1_df2, col_2_df2])
        logging.info(f'Merged DataFrame shape: {df.shape}')
    except Exception as e:
        logging.error(f'An error occurred in {merge_on_two_columns.__name__}: {e}')
        df = pd.DataFrame()  # Return an empty DataFrame in case of error
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


# for staging
directory_path = 'all_crime_data/crime_datasets/'
directory_path_prop = 'all_crime_data/property_datasets/'
directory_path_lsoa = 'all_crime_data/lsoa_datasets/uk_postcodes.csv'
directory_path_pop_area = 'all_crime_data/population_datasets/population_area.csv'
directory_path_dep_area = 'all_crime_data/deprivation_datasets/deprivation.csv'

base_dir = '.'

# for primary
crime_types = [
    'Violence and sexual offences', 'Public order', 'Anti-social behaviour', 
    'Criminal damage and arson', 'Burglary', 'Robbery', 
    'Theft from the person', 'Bicycle theft'
]


def staging():
    """
    Loads, processes, and outputs data from various CSV datasets, performing the necessary staging 
    transformations for each dataset. This includes concatenating, renaming columns, and performing
    other transformations like converting dates and formatting text.

    Logging is used to track the progress and any errors that occur during the execution of the staging 
    pipeline. The datasets staged are:
    
    1. Street dataset: Load, concatenate, and output the street-level crime data.
    2. Outcome dataset: Load, concatenate, and output the outcome-level crime data.
    3. Stop and search (sas) dataset: Load, convert dates, and output the stop-and-search data.
    4. Property dataset: Load, rename columns, concatenate, and convert data types for property transactions.
    5. LSOA dataset: Load and rename columns for the LSOA dataset.
    6. Population area dataset: Load and output population-related data.
    7. Deprivation dataset: Load, rename columns, and output deprivation data.

    Each operation outputs the processed data to its respective CSV file, logging success or failure.
    """
    logging.info('Starting staging layer...')
    
    # Street dataset
    try:
        check_and_create_folders(base_dir)
        df_1 = load_csvs_to_list(directory_path, 'street')
        if df_1:  # Ensure it's not an empty list
            df_1 = concat_df(df_1)
            if isinstance(df_1, pd.DataFrame):
                output_to_csv(df_1, 'crime_data_outputs/street_staging_output.csv')
            else:
                logging.error("Concatenation of street failed.")
    except Exception as e:
        logging.error(f'An error occurred in street staging: {e}')  

    # outcome dataset
    try:
        df_2 = load_csvs_to_list(directory_path, 'outcomes')
        if df_2:  # Ensure it's not an empty list
            df_2 = concat_df(df_2)
            if isinstance(df_2, pd.DataFrame):
                output_to_csv(df_2, 'crime_data_outputs/outcomes_staging_output.csv')
            else:
                logging.error("Concatenation of outcomes failed.")
    except Exception as e:
        logging.error(f'An error occurred in outcomes staging: {e}') 
    
    # Stop and search datasets 
    try:
        df_3 = load_csvs_with_area(directory_path, 'search')
        if not df_3.empty:  # Ensure it's not an empty list
            df_3 = date_convert_dtype(df_3, ['Date'])
            if isinstance(df_3, pd.DataFrame):
                output_to_csv(df_3, 'crime_data_outputs/sas_staging_output.csv')
            else:
                logging.error("Concatenation of stop and search datasets failed.")
    except Exception as e:
        logging.error(f'An error occurred in sas staging: {e}')    
    
    # Property datasets
    try:
        df_4 = load_csvs_to_list(directory_path_prop, 'property')
        if df_4:  # Ensure it's not an empty list
            df_4 = change_col_names(df_4)
            df_4 = concat_df(df_4)
            df_4 = change_text_to_proper_case(df_4, cols_to_change)
            df_4 = date_convert_dtype(df_4, ['Date of transfer'])
            if isinstance(df_4, pd.DataFrame):
                output_to_csv(df_4, 'crime_data_outputs/property_staging_output.csv')
            else:
                logging.error("Concatenation of property datasets failed.")
    except Exception as e:
        logging.error(f'An error occurred in property staging: {e}')
    
    # LSOA datasets
    try:
        df_5 = pd.read_csv(directory_path_lsoa)
        lsoa_headers = {
            'longitude':'Longitude',
            'latitude':'Latitude'
        }
        df_5 = rename_cols(df_5, lsoa_headers)
        output_to_csv(df_5, 'crime_data_outputs/lsoa_staging_output.csv')
    except Exception as e: 
        logging.error(f'An error occurred in lsoa staging: {e}')

    # Population area dataset
    try:
        df_6 = pd.read_csv(directory_path_pop_area)
        output_to_csv(df_6, 'crime_data_outputs/population_area_staging_output.csv')
    except Exception as e: 
        logging.error(f'An error occurred in population_area staging: {e}')    

    # deprivation dataset
    try:
        df_7 = pd.read_csv(directory_path_dep_area)
        col_headers={
            'Lower tier local authorities Code':'LTLA code',
            'Household deprived in the education dimension (3 categories)':'Household deprived in education',
            'Household deprived in the health and disability dimension (3 categories)':'Household deprived in health and disability',
            'Household deprived in the employment dimension (3 categories)':'Household deprived in employment',
            'Household deprived in the housing dimension (3 categories)':'Household deprived in housing'
        }
        df_7 = rename_cols(df_7, col_headers)
        output_to_csv(df_7, 'crime_data_outputs/deprivation_staging_output.csv')
    except Exception as e:
        logging.error(f'An error occurred in deprivation staging: {e}')

    logging.info('Finished staging layer')


def primary():
    """
    Merges, filters, and further processes staged data, preparing it for reporting. Operations include:
    
    1. Street and outcome datasets: Merge and clean street and outcome crime data, filter crime types, 
       and categorize outcomes.
    2. Stop and search (sas) dataset: Output the primary stop-and-search dataset as-is.
    3. Property dataset: Clean and process property data, replacing specific column values and handling 
       missing postcodes.
    4. LSOA dataset: Merge LSOA data with crime data, clean and round coordinate data, and split postcodes.
    5. Population dataset: Output the primary population area dataset as-is.
    6. Deprivation dataset: Clean deprivation data, replace specific column values, and drop unnecessary columns.

    The processed datasets are then saved to CSV files, and logging is used to monitor errors and progress.
    """
    logging.info('Starting primary layer...')
    
    df_1 = None
    # Street and outcome datasets
    try:
        df_1a = pd.read_csv('crime_data_outputs/street_staging_output.csv')
        df_1b = pd.read_csv('crime_data_outputs/outcomes_staging_output.csv')
        df_1 = merge_files_on(df_1a, df_1b, 'outer', 'Crime ID', 'Crime ID')
        col_drop = {
            'Context', 'Month_y', 'Reported by_y', 'Falls within_y', 'Longitude_y', 'Latitude_y',
            'Location_y', 'LSOA code_y', 'LSOA name_y' 
        }
        df_1 = drop_col(df_1, col_drop)
        col_dict = {
            'Month_x':'Month',
            'Reported by_x':'Reported by',
            'Falls within_x':'Falls within',
            'Longitude_x':'Longitude',
            'Latitude_x':'Latitude',
            'Location_x':'Location',
            'LSOA code_x':'LSOA code',
            'LSOA name_x':'LSOA name'
        }
        df_1 = rename_cols(df_1, col_dict)
        df_1 = drop_na_rows(df_1, 'Longitude')
        df_1 = drop_col(df_1, 'Context')
        df_1 = round_values(df_1, 'Latitude', 3)
        df_1 = round_values(df_1, 'Longitude', 3)
        df_1 = replace_nulls(df_1, 'Crime type')
        df_1 = filter_crime_types(df_1, crime_types)
        df_1 = categorize_outcome(df_1)
    except Exception as e:
        logging.error(f'An error occurred in street_outcomes primary: {e}')

    # Stop and search datasets 
    try:
        df_2 = pd.read_csv('crime_data_outputs/sas_staging_output.csv')
        output_to_csv(df_2, 'crime_data_outputs/sas_primary_output.csv')
    except Exception as e:
        logging.error(f'An error occurred in sas primary: {e}')    
    
    # Property datasets
    try:
        df_3 = pd.read_csv('crime_data_outputs/property_staging_output.csv')
        df_3 = drop_na_rows(df_3, 'Postcode')
        replace_list = {
            'Y':'New property',
            'N': 'Established property'
        }
        df_3 = replace_col_values(df_3, ['Old/new'], replace_list)
        output_to_csv(df_3, 'crime_data_outputs/property_primary_output.csv')
    except Exception as e:
        logging.error(f'An error occurred in property primary: {e}')
    
    # LSOA datasets
    try:
        df_4 = pd.read_csv('crime_data_outputs/lsoa_staging_output.csv')
        df_4 = round_values(df_4, 'Latitude', 3)
        df_4 = round_values(df_4, 'Longitude', 3)
        df_4 = drop_duplicate_rows(df_4, 'Latitude', 'Longitude')
        df_4 = pd.merge(df_1, df_4, on=['Longitude','Latitude'])
        df_4 = split_postcode(df_4, 'postcode','First half of postcode')
        output_to_csv(df_4, 'crime_data_outputs/lsoa_primary_output.csv')
    except Exception as e:
        logging.error(f'An error occurred in lsoa primary: {e}')

    # population datasets
    try:
        df_5 = pd.read_csv('crime_data_outputs/population_area_staging_output.csv')
        output_to_csv(df_5, 'crime_data_outputs/population_area_primary_output.csv')
    except Exception as e:
        logging.error(f'An error occurred in population_area primary: {e}')
    
    # deprived datasets
    try:
        df_6 = pd.read_csv('crime_data_outputs/deprivation_staging_output.csv')
        cols_drop = {
            'Household deprived in the education dimension (3 categories) Code',
            'Household deprived in the health and disability dimension (3 categories) Code',
            'Household deprived in the employment dimension (3 categories) Code',
            'Household deprived in the housing dimension (3 categories) Code'
        }
        df_6 = drop_col(df_6, cols_drop)
        ed_val_replace = {
            'Household is not deprived in the education dimension':'Not education deprived',
            'Household is deprived in the education dimension':'Education deprived'
        }
        emp_val_replace = {
            'Household is not deprived in the employment dimension':'Not employment deprived',
            'Household is deprived in the employment dimension':'Employment deprived'
        }
        hd_val_replace = {
            'Household is not deprived in the health and disability dimension':'Not health and disability deprived',
            'Household is deprived in the health and disability dimension':'Health and disability deprived'
        }
        hou_val_replace = {
            'Household is not deprived in the housing dimension':'Not housing deprived',
            'Household is deprived in the housing dimension':'Household deprived'
        }
        df_6 = replace_col_values(df_6, ['Household deprived in education'], ed_val_replace)
        df_6 = replace_col_values(df_6, ['Household deprived in employment'], emp_val_replace)
        df_6 = replace_col_values(df_6, ['Household deprived in health and disability'], hd_val_replace)
        df_6 = replace_col_values(df_6, ['Household deprived in housing'], hou_val_replace)
        output_to_csv(df_6, 'crime_data_outputs/deprivation_primary_output.csv')
    except Exception as e:
        logging.error(f'An error occurred in deprivation primary: {e}')
    
    logging.info('Finished primary layer')


def reporting():
    """
    Aggregates and summarizes processed datasets for reporting purposes, generating key insights like crime 
    counts per area, outcomes, and other indicators.

    Reporting operations include:
    
    1. LSOA and crime type aggregation: Group the LSOA data by 'Falls within' and 'Crime type', counting crimes.
    2. LSOA and postcode aggregation: Group the LSOA data by 'Falls within' and 'First half of postcode', counting crimes.
    3. LSOA and broad outcome category aggregation: Group the LSOA data by 'Falls within' and 'Broad Outcome Category', counting crimes.
    4. Stop and search (sas) aggregation: Group SAS data by 'Area' and 'Age range' and by 'Area' and 'Outcome'.
    5. Deprivation data aggregation: Sum the observations of crime in deprivation datasets by 'Lower tier local authorities'.

    Outputs the aggregated datasets to respective CSV files, logging errors and reporting success.
    """
    logging.info('Starting reporting layer...')
    
    try:
        df_1 = pd.read_csv('crime_data_outputs/lsoa_primary_output.csv', low_memory = False)
        df_1a = df_1.groupby(['Falls within', 'Crime type'])['Falls within'].count().reset_index(name='Crime count')
        df_1a = df_1a[['Falls within', 'Crime type', 'Crime count']]
        df_1a = df_1a.sort_values(by=['Falls within', 'Crime count'], ascending=[True, False])
        output_to_csv(df_1a, 'crime_data_outputs/aggregated_dataframes/lsoa_crimetype_agg_output.csv')
        logging.info('LSOA_code aggregated dataframe #1 completed!')
    except Exception as e:
        logging.error('An error occured in creating the #1 aggregated dataframe for crime type')

    try:
        df_1b = df_1.groupby(['Falls within', 'First half of postcode'])['Falls within'].count().reset_index(name='Crime count')
        df_1b = df_1b[['Falls within', 'First half of postcode', 'Crime count']]
        df_1b = df_1b.sort_values(by=['Falls within', 'Crime count'], ascending=[True, False])
        output_to_csv(df_1b, 'crime_data_outputs/aggregated_dataframes/lsoa_postcode_agg_output.csv')
        logging.info('LSOA_code aggregated dataframe #2 completed!')
    except Exception as e:
        logging.error('An error occured in creating the #2 aggregated dataframe for postcode')
        
    try:
        df_1c = df_1.groupby(['Falls within', 'Broad Outcome Category'])['Falls within'].count().reset_index(name='Crime count')
        df_1c = df_1c[['Falls within', 'Broad Outcome Category', 'Crime count']]
        df_1c = df_1c.sort_values(by=['Falls within', 'Crime count'], ascending=[True, False])
        output_to_csv(df_1c, 'crime_data_outputs/aggregated_dataframes/lsoa_broadoutcome_agg_output.csv')
        logging.info('LSOA_code aggregated dataframe #3 completed!')
    except Exception as e:
        logging.error('An error occured in creating the #3 aggregated dataframe for broad outcome category')

    try:
        df_2 = pd.read_csv('crime_data_outputs/sas_primary_output.csv')
        df_2a = df_2.groupby(['Area', 'Age range'])['Area'].count().reset_index(name='Crime count')
        df_2a = df_2a[['Area', 'Age range', 'Crime count']]
        df_2a = df_2a.sort_values(by=['Area', 'Crime count'], ascending=[True, False])
        output_to_csv(df_2a, 'crime_data_outputs/aggregated_dataframes/sas_agerange_agg_output.csv')
        logging.info('SAS aggregated dataframe #1 completed!')
    except Exception as e:
        logging.error('An error occured in creating the #1 aggregated dataframe for SAS age range')

    try:
        df_2b = df_2.groupby(['Area', 'Outcome'])['Area'].count().reset_index(name='Crime count')
        df_2b = df_2b[['Area', 'Outcome', 'Crime count']]
        df_2b = df_2b.sort_values(by=['Area', 'Crime count'], ascending=[True, False])
        output_to_csv(df_2b, 'crime_data_outputs/aggregated_dataframes/sas_outcome_agg_output.csv')
        logging.info('SAS aggregated dataframe #2 completed!')
    except Exception as e:
        logging.error('An error occured in creating the #2 aggregated dataframe for SAS outcomes')

    try:
        df_3 = pd.read_csv('crime_data_outputs/deprivation_primary_output.csv')
        df_3a = df_3.groupby('Lower tier local authorities')['Observation'].sum().reset_index(name='Sum of crime')
        df_3a = df_3a.sort_values(by=['Lower tier local authorities', 'Sum of crime'], ascending=[True, False])
        output_to_csv(df_3a, 'crime_data_outputs/aggregated_dataframes/deprivation_LTLA_agg_output.csv')
        logging.info('Deprivation aggregated dataframe #1 completed!')
    except Exception as e:
        logging.error('An error occured in creating the #1 aggregated dataframe for deprivation LTLA')


def main(pipeline='all'):
    """
    Executes the pipeline for processing crime and other related datasets. Depending on the input parameter, 
    the function can run the entire pipeline or individual stages (staging, primary, or reporting).

    Parameters:
    pipeline (str): Indicates which part of the pipeline to run. Options are:
        - 'all': Run the entire pipeline (staging, primary, and reporting).
        - 'staging': Run only the staging layer.
        - 'primary': Run only the primary layer.
        - 'reporting': Run only the reporting layer.

    The function logs the progress and handles any exceptions that may occur during pipeline execution.
    """
    logging.info('Pipeline execution started...')
    print('Pipeline execution started...')
    try:
        if pipeline in ['all', 'staging', 'primary', 'reporting']:
            staging()
            logging.info("Staging execution completed successfully")
            if pipeline == 'staging':
                # If only staging is requested, print success and return
                logging.info("Pipeline run complete")
                return
            # Process the staged data
            primary()
            logging.info("Primary execution completed successfully")
            if pipeline == 'primary':
                # If only primary is requested, print success and return 
                logging.info("Pipeline run complete")
                return
            # Generate reports based on processed data
            reporting()
            logging.info("Reporting execution completed successfully")
            if pipeline == 'reporting':
                logging.info("Pipeline run complete")
                return
            logging.info("Full pipeline run complete")
            print('Pipeline finished!')
        else:
            # Inform the user about an invalid pipeline stage input
            logging.critical("Invalid pipeline stage specified. Please choose 'staging', 'primary', 'reporting', or 'all'.")
    except Exception as e:
        # Catch and print any exceptions occurred during pipeline execution
        logging.error(f"Pipeline execution failed: {e}")

if __name__ == '__main__':
    main()            