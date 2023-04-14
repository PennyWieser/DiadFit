import numpy as np
import pandas as pd
import math
import os

try:
    import docx
except ImportError:
    ImportError('Not installed')

import datetime
import warnings

encode="ISO-8859-1"

## Function for getting file names

def get_files(*,path,filetype):
    """
    Returns a list of files with specific file type(s) in the specified directory
    Parameters
    --------------
    path: str
        Path of the directory where the files are located.
    filetype: str or list of str
        Filetype(s) of the files to be included in the output list.

    Returns
    -------------
    file_ls: list
        A list of files with the specified file type(s) in the directory
    """
    file_ls=[]
    for file in os.listdir(path):
        if file.endswith(tuple(filetype)):
            file_ls.append(file)
    return file_ls

## Function for extracting information from the docx reports
def report_info (*,path,report):
    """
    Reads a word document report (exported from ESI-TEC software), extracts and returns the start date and time of the pressure recording and the serial number of the sensor.
    Parameters
    --------------
    path: str
        Path of the directory where the word document is located
    report: str
        The name of the word document

    Returns
    -------------
    start_time: datetime object
        Start time of the analysis in the report
    sn_str: str
        Serial number of the sensor
    """

    # Open the Word document
    document = docx.Document(path+'/'+report)

    # Iterate over all paragraphs in the document
    for para in document.paragraphs:
        # Check if the paragraph contains the text "Test Date:"
        if "Test Date:" in para.text:
            # Extract the date and time from the paragraph text
            date_time_str = ":".join(para.text.split(":")[1:]).strip()
            start_time = datetime.datetime.strptime(date_time_str, "%d/%m/%Y %H:%M:%S")
        if "Serial No:" in para.text:
        # Extract the date and time from the paragraph text
            sn_str = ":".join(para.text.split(":")[1:]).strip()

    print(start_time)
    print('Serial No. '+ sn_str)
    return start_time, sn_str

## Function for reading in data

def read_pfiles(*,path,file,start_time,sn_name='0132212'): #UCB '0132212', cornell '0830903'
    """
    Reads a csv or xlsx file of pressure data exported from ESI-TEC software and returns a dataframe with two extra columns "Date and Time" (datetime object) and "unix_timestamp" (timestamp expressed as UNIX time, or time in seconds since the epoch time Jan 1st, 1970 00:00:00 UTC) based on the start_time of the pressure recording and time since start in the file. It also renames the time column to Time_sincestart.
    Parameters
    --------------
    path: str
        Path of the directory where the file is located
    file: str
        The name of the file to be read
    start_time: str
        The starting time of the recording in the format 'yyyy-mm-dd hh:mm:ss', this can be obtained from the docx report by using the report_info function
    sn_name: str
        The serial number of the sensor. Default is '0132212', this can be obtained from the docx report by using the report_info function.

    Returns
    -------------
    data: pd.DataFrame
        DataFrame containing the data from the file along with "Date and Time" and "unix_timestamp" columns.
    """

    _,filetype=os.path.splitext(path+'/'+file)

    if filetype=='.csv':
        data=pd.read_csv(path+'/'+file,skiprows=[0])
        data['Date and Time'] = pd.to_timedelta(data['Time/s'],unit='s')
        data['Date and Time'] = start_time + data['Date and Time']
        data['unix_timestamp'] = data['Date and Time'].apply(lambda x: x.timestamp())
        data= data.rename(columns={'Time/s': 'Time_sincestart/s'})
    if filetype=='.xlsx':
        data=pd.read_excel(path+'/'+file,sheet_name='Sensor '+sn_name)
        data['Date and Time'] = pd.to_timedelta(data['Time'])
        data['Date and Time'] = start_time + data['Date and Time']
        data['unix_timestamp'] = data['Date and Time'].apply(lambda x: x.timestamp())
        data= data.rename(columns={'Time': 'Time_sincestart'})

    return data

## Function for calculating datetime and duration from metadata file

def add_datetime_and_duration_cols(*,df,raman_cpu_offset='none',offset_hms=[0,0,0]):
    """
    Takes a DataFrame and adds columns for "Date and Time", "unix_timestamp" and "duration_s". The input frame should be either the complete DataFrame with spectra metadata and fits output by DiadFit or just the spectral metadata. "Date and Time" contains datetime objects; 'unix_timestamp' contains the numeric (float) timestamp for the date and time in standard UNIX time or seconds since epoch time (Jan 1st 1970 00:00:00 UTC), this is plottable; and "duration_s" is the duration of the analysis in seconds.

    Parameters
    --------------
    df: pd.DataFrame
        The input DataFrame

    Returns
    -------------
    df:pd.DataFrame
        The input DataFrame with additional columns for date and time, duration, and unix timestamp.
    """

    def duration_to_timedelta(time_string):
        """
        Converts the duration time string to a timedelta object.

        Parameters
        --------------
        time_string: str
            The time string in the format of 'hh:mm:ss' or '[hh:mm:ss]'

        Returns
        -------------
        time: timedelta object
            Duration as a timedelta object
        """
        time_string = time_string.replace("'","").replace("[","").replace("]","")
        hours, minutes, seconds = [int(val.rstrip('hms')) for val in time_string.split(',')]
        time = datetime.timedelta(hours=hours, minutes=minutes, seconds=seconds)
        return time

    for i in df.index:
        df.loc[i,'date']=df['date'][i].strip()
        df.loc[i,'24hr_time']=df['24hr_time'][i].strip()

    df['Date and Time'] = df['date'] + ' ' + df['24hr_time']
    df['Date and Time'] = df['Date and Time'].apply(lambda x: datetime.datetime.strptime(x, '%B %d, %Y %I:%M:%S %p'))

    if raman_cpu_offset=='none':
        df['unix_timestamp'] = df['Date and Time'].apply(lambda x: x.timestamp())
    elif raman_cpu_offset=='behind':
        df['Date and Time - offset']=df['Date and Time']+datetime.timedelta(hours=offset_hms[0],minutes=offset_hms[1],seconds=offset_hms[2])
        df['unix_timestamp'] = df['Date and Time - offset'].apply(lambda x: x.timestamp())
    elif raman_cpu_offset=='ahead':
        df['Date and Time - offset']=df['Date and Time']-datetime.timedelta(hours=offset_hms[0],minutes=offset_hms[1],seconds=offset_hms[2])
        df['unix_timestamp'] = df['Date and Time - offset'].apply(lambda x: x.timestamp())
    else:
        warnings.warn("Invalid value for raman_cpu_offset, please use 'behind', 'ahead' or 'none'")
        return
    dur_days = df['duration'].apply(duration_to_timedelta)
    df['duration_s']=dur_days.dt.total_seconds()
    return df

## Function for calculating the pressure median for each analysis

def get_p_medians(*,pdata,sdata,export_all=False):
    """
    Takes two DataFrames and returns a new DataFrame containing the median and median absolute deviation of the pressure values for each Raman analysis. It finds the closest matching rows in the two DataFrames based on timestamp and filters the pressure data between the matched timestamps. It then calculates the median and mean absolute deviation of the filtered pressure data. If export_all==True, it also includes the start time, end time, duration, and filename in the output DataFrame.

    Parameters
    --------------
    pdata: pd.DataFrame
        The pressure DataFrame (output by read_pfiles)
    sdata: pd.DataFrame
        The spectral DataFrame (loaded in by user)
    export_all: bool (Optional)
        Indicates whether to include additional information in the output DataFrame.

    """
    df1=pdata.copy()
    df2=sdata.copy()

    df1['unix_timestamp'] = pd.to_datetime(df1['unix_timestamp'], unit='s')
    df2['unix_timestamp'] = pd.to_datetime(df2['unix_timestamp'], unit='s')

    idx = []
    new_data = pd.DataFrame([])

    # iterate over the rows in the second DataFrame
    for i, row in df2.iterrows():
        # find the closest matching row in the first DataFrame
        closest_index = (df1['unix_timestamp'] - row['unix_timestamp']).abs().idxmin()
        # check if the difference between the timestamps is less than one second
        if abs((df1.loc[closest_index,'unix_timestamp'] - row['unix_timestamp']).total_seconds()) <= 2:
            idx.append((closest_index, i))  # <-- append a tuple containing the indices

    # Iterate through the list of tuples in 'idx'
    for idx_df1, idx_df2 in idx:
        # Extract the relevant information from 'df1' and 'df2'
        filename = df2.loc[idx_df2, 'filename_x']
        start_time_P = df1.loc[idx_df1, 'unix_timestamp']
        start_time_S = df2.loc[idx_df2,'unix_timestamp']
        duration = df2.loc[idx_df2, 'duration_s']
        end_time_P = start_time_P + pd.Timedelta(seconds=float(duration))
        # filter pressure data between start_time_P and end_time
        pressure_data = df1[(df1['unix_timestamp'] >= start_time_P) & (df1['unix_timestamp'] <= end_time_P)]
        median_pressure = pressure_data['Pressure / MPa'].median()
        mad_pressure = (pressure_data['Pressure / MPa'] - pressure_data['Pressure / MPa'].mean()).abs().mean()
        median_temp = pressure_data['Temperature / °C'].median()
        mad_temp = (pressure_data['Temperature / °C'] - pressure_data['Temperature / °C'].mean()).abs().mean()

        # Append a new row to the new dataframe
        if export_all==True:
            new_row = pd.DataFrame({'filename_x': filename, 'start_time_S':start_time_S,'start_time_P': start_time_P, 'duration': duration,'end_time_P':end_time_P, 'median_pressure': median_pressure, 'mad_pressure': mad_pressure,'median_temp': median_temp, 'mad_temp': mad_temp},index=[0])
            new_data=pd.concat([new_data,new_row],ignore_index=True)
        else:
            new_row = pd.DataFrame({'filename_x': filename, 'start_time_P': start_time_P, 'end_time_P':end_time_P, 'median_pressure': median_pressure, 'mad_pressure': mad_pressure,'median_temp': median_temp, 'mad_temp': mad_temp},index=[0])
            new_data=pd.concat([new_data,new_row],ignore_index=True)
    return new_data