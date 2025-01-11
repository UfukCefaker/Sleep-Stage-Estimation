"""
This file is provided as a pipeline file for the patient's night-long datas.

Example : subprocess.run(["python", "pipeline.py", "Alice"])
"""

import pandas as pd
import sys
import numpy as np
import polars as pl
import os
import re
from tqdm import tqdm

def shrink_data(df: pd.DataFrame,shrink_factor):
    """
    This function shrinks the dataframe of a patient by the provided factor, while clearing the nan values with appropriate fillings.

    :param df: The dataframe containing the patient data.
    :type df: pandas.DataFrame
    :param shrink_factor: The factor by which to shrink or adjust the data.
    :type shrink_factor: int
    :return: The modified dataframe after applying the shrink factor.
    :rtype: pandas.DataFrame
    """
    shrinked = df
    numeric_col_list = shrinked.select_dtypes("float64").columns#["TIMESTAMP","BVP","ACC_X","ACC_Y","ACC_Z","TEMP","EDA","HR","IBI"]
    numeric_columns = shrinked.loc[:,numeric_col_list]
    sleep_stage = shrinked.loc[:,["Sleep_Stage"]]
    #diseases = shrinked.loc[:,[col for col in shrinked.columns if col not in numeric_col_list and col!= "Sleep_Stage"]]
    numeric_columns = numeric_columns.groupby(shrinked.index // shrink_factor).mean()
    sleep_stage = sleep_stage.groupby(shrinked.index // shrink_factor).first()
    #diseases = diseases.groupby(df.index // shrink_factor).any().astype(int)
    shrinked = pd.concat([numeric_columns,sleep_stage],axis = 1)
    return shrinked


def pipeline(file_name):
    """
    This is the pipeline function that processes the patient data.

    :param file_name: The name of the file containing patient data.
    :type file_name: str
    :return: The processed DataFrame.
    :rtype: pandas.DataFrame
    """
    patient_df = pd.read_csv(file_name)
    numeric_columns = [x for x in patient_df.columns if pd.api.types.is_numeric_dtype(patient_df[x])]
   
    patient_df.loc[:,["TIMESTAMP","BVP","ACC_X","ACC_Y","ACC_Z","TEMP","EDA","HR","IBI","Sleep_Stage"]].bfill(inplace=True) # Fill numeric columns with backfill
    patient_df.loc[:,["Obstructive_Apnea","Central_Apnea","Hypopnea","Multiple_Events"]] = patient_df.loc[:,["Obstructive_Apnea","Central_Apnea","Hypopnea","Multiple_Events"]].isna().astype(int)
    
    patient_df["IBI_Moving_Average_640"] = patient_df["IBI"].rolling(64*10).mean() #Every 64 rows contain a second, which means our window contains a 10 second average
    patient_df["IBI_Moving_STD_640"] = patient_df["IBI"].rolling(64*10).std() 

    patient_df["Abnormal_IBI_Moving_3"] =  (abs(patient_df["IBI_Moving_Average_640"] - patient_df["IBI"]) > 3*patient_df["IBI_Moving_STD_640"]).astype(int)
    patient_df["Abnormal_IBI_Moving_4"] =  (abs(patient_df["IBI_Moving_Average_640"] - patient_df["IBI"]) > 4*patient_df["IBI_Moving_STD_640"]).astype(int)
    
    patient_df.loc[:,["Abnormal_IBI_Moving_4","Abnormal_IBI_Moving_3","IBI_Moving_STD_640","IBI_Moving_Average_640","IBI"]] =     patient_df.loc[:,["Abnormal_IBI_Moving_4","Abnormal_IBI_Moving_3","IBI_Moving_STD_640","IBI_Moving_Average_640","IBI"]].bfill()

    patient_df["ACC"] = np.sqrt(patient_df["ACC_X"]**2 + patient_df["ACC_Y"]**2 + patient_df["ACC_Z"]**2)
    
    return patient_df

def pipeline_efficient(file_name):
    patient_df = pl.read_csv(file_name)

    # Fill numeric columns with backfill
    columns_to_backfill = ["TIMESTAMP", "BVP", "ACC_X", "ACC_Y", "ACC_Z", "TEMP", "EDA", "HR", "IBI"]
    patient_df = patient_df.with_columns([
        patient_df[col].fill_null(strategy="backward") for col in columns_to_backfill
    ])


    # Mark NaN in specific columns as 1 (events), others as 0
    """event_columns = ["Obstructive_Apnea", "Central_Apnea", "Hypopnea", "Multiple_Events"]
    patient_df = patient_df.with_columns([
    (patient_df[col].is_not_null().cast(pl.Int8)).alias(col) for col in event_columns
    ])"""

    # Compute moving average and standard deviation for IBI
    window_size = 64 * 10
    patient_df = patient_df.with_columns([
        patient_df["IBI"].rolling_mean(window_size).alias("IBI_Moving_Average_640"),
        patient_df["IBI"].rolling_std(window_size).alias("IBI_Moving_STD_640"),
    ])
    patient_df = patient_df.with_columns([
        patient_df["IBI"].cast(pl.Float64).fill_null(strategy="backward"),
        patient_df["IBI_Moving_Average_640"].cast(pl.Float64).fill_null(strategy="backward"),
        patient_df["IBI_Moving_STD_640"].cast(pl.Float64).fill_null(strategy="backward"),
    ])
    #Replace Missing with null
    patient_df = patient_df.with_columns(
    pl.col("Sleep_Stage").replace("Missing", None)
    )

    # Backfill null values in the DataFrame
    patient_df = patient_df.fill_null(strategy="backward")
    # Identify abnormal IBI based on threshold
    patient_df = patient_df.with_columns([
        ((patient_df["IBI"] - patient_df["IBI_Moving_Average_640"]).abs() > 3 * patient_df["IBI_Moving_STD_640"]).cast(pl.Int8).alias("Abnormal_IBI_Moving_3"),
        ((patient_df["IBI"] - patient_df["IBI_Moving_Average_640"]).abs() > 4 * patient_df["IBI_Moving_STD_640"]).cast(pl.Int8).alias("Abnormal_IBI_Moving_4"),
    ])

    # Backfill specific columns
    backfill_columns = ["Abnormal_IBI_Moving_4", "Abnormal_IBI_Moving_3", "IBI_Moving_STD_640", "IBI_Moving_Average_640", "IBI"]
    patient_df = patient_df.with_columns([
        patient_df[col].fill_null(strategy="backward") for col in backfill_columns
    ])

    # Calculate the magnitude of accelerometer data
    patient_df = patient_df.with_columns(
        (np.sqrt(
            patient_df["ACC_X"]**2 + 
            patient_df["ACC_Y"]**2 + 
            patient_df["ACC_Z"]**2
        )).alias("ACC")
    )

    # Convert to pandas DataFrame
    pandas_df = patient_df.to_pandas()

    return pandas_df

def pipeline_efficient_not_processed(file_name):
    return pd.read_csv(file_name)


def gather_dict(parent_folder,process = True):
    csv_files = [file for file in os.listdir(parent_folder) if file.endswith(".csv")]
    patients_records={}
    for patient in tqdm(csv_files):
        patient_id = extract_sxxx(patient)
        if not patient_id:
            return
        if process:
            patients_records[patient_id] = pipeline_efficient(os.path.join(parent_folder, patient))
        else:
            patients_records[patient_id] = pipeline_efficient_not_processed(os.path.join(parent_folder, patient))

    return patients_records

def extract_sxxx(filename):
    # Define the pattern to match 'Sxxx' where x is a digit (1 to 3 digits)
    pattern = r"(S\d{1,3})"
    match = re.search(pattern, filename)
    if match:
        return match.group(1)  # Return the matched string
    return None  #