import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches
import lmfit
from lmfit.models import GaussianModel, VoigtModel, LinearModel, ConstantModel
from scipy.signal import find_peaks
import os
import re
from os import listdir
from os.path import isfile, join
from tqdm import tqdm
import datetime
import calendar

encode="ISO-8859-1"

## Functions for getting file names


def check_for_duplicates(spectra_path, prefix=True, prefix_str=' ', exception=True):
    """    This function checks for duplicate filenames in a specified directory and prints the duplicates if found.
    
    Parameters:
    spectra_path (str): 
        The path of the directory containing the files to be checked for duplicates.
    prefix (bool):
        If True, the function will remove the specified prefix string from the filenames before checking for duplicates. Default is True.
    prefix_str (str: 
        The prefix string to be removed from filenames if 'prefix' is set to True. Default is a single space ' '.
    
    Returns:
    file_m (numpy.ndarray): A numpy array containing the modified filenames after removing the prefix (if specified).
    
    
    """

    All_files_spectra= [f for f in listdir(spectra_path) if isfile(join(spectra_path, f))]

    file_m=np.empty(len(All_files_spectra), dtype=object)
    for i in range(0, len(All_files_spectra)):
        name=All_files_spectra[i]
        # If no prefix or suffix to remove, simple
        if prefix is False:
            name2=name
        else:
            name2=name.split(prefix_str, maxsplit=1)[1:]
        file_m[i]=name2[0]

    if len(file_m)!=len(pd.Series(file_m).unique()):
        file_m_s=pd.Series(file_m)
        print('duplicates')
        print(file_m_s[file_m_s.duplicated()])
        print('OOPS. at least one of your file name is duplicated go back to your spectra, you named a file twice, this will confuse the stitching ')
        #raise Exception('Duplicate file')
        if exception is True:
            raise TypeError('SORT OUT YOUR DUPLICATES BEFORE PROCEEDING!')

    return file_m



def get_files(path, ID_str=None, file_ext='txt', exclude_str=None, exclude_type=None, sort=True):
    """ This function takes a user path, and extracts all files which contain the ID_str

    Parameters
    -----------

    path: str
        Folder user wishes to read data from
    sort: bool
        If true, sorts files alphabetically
    ID_str: list
        Finds all files containing this string (e.g. ['Ne', 'NE']
    exclude_str: str
        Excludes files with this string in the name
    file_ext: str
        Gets all files of this format only (e.g. txt)


    Returns
    -----------
    list: file names as a list.

    """

    Allfiles = [f for f in listdir(path) if isfile(join(path, f))]

    # Take only files with the right file extension
    if ID_str is not None:
        Allfiles_type=[item for item in Allfiles if file_ext in item and ID_str in item]
    else:
        Allfiles_type=[item for item in Allfiles if file_ext in item]


    if exclude_str is None:
        Ne_files=Allfiles_type
    else:
        Ne_files=[x for x in Allfiles_type if not any(e in x for e in exclude_str)]


    # Allfiles = [f for f in listdir(path) if isfile(join(path, f))]
    # Ne_files=[item for item in Allfiles if ID_str in item and file_ext in item and exclude_str not in item]

    if sort is True:
        Ne_files=sorted(Ne_files)
    return Ne_files







def get_all_txt_files(path):
    """ This function takes a user path, and gets all the .txt. files in that path.

    Parameters
    -----------
    path: str
        Folder user wishes to read data from
    """

    Allfiles_all = [f for f in listdir(path) if isfile(join(path, f))]
    # Use only txt files
    type(Allfiles_all)
    All_files=[]
    for file in Allfiles_all:
        if '.txt' in file and 'pandas' not in file:
            All_files.append(format(file))
    return All_files


## Functions to just simply get data to plot up
def get_data(*, path=None, filename=None, Diad_files=None, filetype='Witec_ASCII'):
    """
    Extracts data as a np.array from user file of differen types

    Parameters
    ---------------
    path: str
        path with spectra in
    filename: str
        Filename of specific spectra
    filetype: str
        choose from 'Witec_ASCII', 'headless_txt', 'headless_csv', 'head_csv', 'Witec_ASCII',
        'HORIBA_txt', 'Renishaw_txt'
    Diad_Files: 
        Name of file, if you dont want to have to specify a path
        
    """
    if Diad_files is None:
        if filetype == 'headless_txt':
            df=pd.read_csv(path+'/'+filename, sep="\t", header=None )

        if filetype=='Witec_ASCII':
            df=read_witec_to_df(path=path, filename=filename)

        if filetype=='Renishaw_txt':
            df_long=pd.read_csv(path+'/'+filename, sep="\t" )
            df=df_long.iloc[:, 0:2]

        if filetype=='HORIBA_txt':
            df=read_HORIBA_to_df(path=path, filename=filename)

        if filetype=='headless_csv':
            df=pd.read_csv(path+str('/')+filename, header=None)
        if filetype=='head_csv':
            df=pd.read_csv(path+str('/')+filename)

    if Diad_files is not None:
        if filetype == 'headless_txt':
            df=pd.read_csv(Diad_files, sep="\t", header=None )

        if filetype=='Witec_ASCII':
            df=read_witec_to_df(Diad_files)

        if filetype=='Renishaw_txt':
            df_long=pd.read_csv(Diad_files, sep="\t" )
            df=df_long.iloc[:, 0:2]

        if filetype=='HORIBA_txt':
            df=read_HORIBA_to_df(Diad_files)

        if filetype=='headless_csv':
            df=pd.read_csv(Diad_files, header=None)
        if filetype=='head_csv':
            df=pd.read_csv(Diad_files)

    df_in=np.array(df)

    if (df_in[0, 0]-df_in[1, 0])>0:
        df_in=np.flipud(df_in)
    # Check if the intrument has inverted it. The new horiba instruments do this.

    return df_in

## Reading different file formats
def read_HORIBA_to_df(*,  path=None, filename):
    """ This function takes in a HORIBA .txt. file with headers with #, and looks down to the row where Data starts (no #),
    and saves this to a new csv file called pandas_.... old file. It exports the data as a pandas dataframe

    Parameters
    -----------

    path: str
        Folder user wishes to read data from

    filename: str
        Specific file being read

    Returns
    ------------
    pd.DataFrame:  
        Dataframe of x-y data


    """
    path2=path+'/'+ 'Peak_fits_txt'
    if os.path.exists(path2):
        a='path exists'
    else:
        os.makedirs(path+'/'+ 'Peak_fits_txt', exist_ok=False)
        print('Ive made a new folder to store your intermediate txt files in')

    if path is None:
        fr = open(filename, 'r', encoding=encode)

        fw=open('pandas2_'+filename, 'w')
    else:
        fr = open(path+'/'+filename, 'r', encoding=encode)
        fw= open(path+'/'+'Peak_fits_txt'+'/'+'pandas2_'+filename, 'w')

    if fr.readline().startswith('#Acq. time'):
        out='HORIBA txt file recognised'
    else:
        raise TypeError('Not a HORIBA txt file with headers')
    while True:
        l=fr.readline()
        if not l.startswith('#'):

            break

    for line in fr:
        fw.write(line)

    fw.close()
    fr.close()
    if path is None:
        #print(filename)
        df=pd.read_csv('pandas2_'+filename, sep="\t", header=None)
    else:
        #print(filename)
        df=pd.read_csv(path+'/'+'Peak_fits_txt'+'/'+'pandas2_'+filename, sep="\t", header=None)

        return df

def read_witec_to_df(*,  path=None, filename):
    """ This function takes in a WITec ASCII.txt. file with metadata mixed with data, and looks down to the row where Data starts,
    and saves this to a new file called pandas_.... old file. It exports the data as a pandas dataframe

    Parameters
    -----------

    path: str
        Folder user wishes to read data from

    filename: str
        Specific file being read


        Returns
    ------------
    pd.DataFrame:  
        Dataframe of x-y data

    """
    if path is None:
        path=os.getcwd()

    path2=path+'/'+ 'Peak_fits_txt'
    if os.path.exists(path2):
        a='path exists'
    else:
        os.makedirs(path+'/'+ 'Peak_fits_txt', exist_ok=False)
        print('Ive made a new folder to store your intermediate txt files in')

    if path is None:
        fr = open(filename, 'r', encoding=encode)
        fw=open('pandas2_'+filename, 'w')
    else:
        fr = open(path+'/'+filename, 'r', encoding=encode)
        fw= open(path+'/'+'Peak_fits_txt'+'/'+'pandas2_'+filename, 'w')

    if fr.readline().startswith('//Exported ASCII'):
        out='ASCI file recognised'
    else:
        raise TypeError('file not an ASCI file')

    while True:
        l=fr.readline()

        if l.startswith('[Data]'):

            break

    for line in fr:
        fw.write(line)

    fw.close()
    fr.close()
    if path is None:

        df=pd.read_csv('pandas2_'+filename, sep="\t")
    else:
        #print(filename)
        df=pd.read_csv(path+'/'+'Peak_fits_txt'+'/'+'pandas2_'+filename, sep="\t")
    array=np.array(df)
    if np.median(array[:, 1])==0:
        raise TypeError(filename+': The median y value is 0, is it possible you stopped the acq before you got any counts? Please delete this file so it doesnt break the loops')

    return df



## Function to extract metadata based on creation or modification of file


def convert_datastamp_to_metadata(path, filename, creation=True, modification=False):
    """ Gets file modification or creation time, outputs as metadata in the same format as for WITEC

    Parameters
    -------------
    
    path: str
        Path where spectra files are stored
    filename: str
        Specific filename
    creation: bool
        If True, gets timestamp based on creation date of file
    modification: bool
        If True, gets timestamp based on modification date of file

    Returns
    ----------
    df of timestamp, and other columns to have the same format as the WITEC metadata output


    """
    if creation is True and modification is True:
        raise Exception('select either Creation=True or modification=True, not both')
    if creation is False and modification is False:
        raise Exception('select one of Creation=True or modification=True')

    path2=path+'\\'+filename
    m_time=os.path.getmtime(path2)
    dt_m = datetime.datetime.fromtimestamp(m_time)
    # Creation time
    c_time = os.path.getctime(path2)
    dt_c = datetime.datetime.fromtimestamp(c_time)
    if creation is True:
        df_time=dt_c
    if modification is True:
        df_time=dt_m
    #date

    month=calendar.month_name[df_time.month]
    Day=df_time.day
    Year=df_time.year
    date_str=month+' ' + str(Day)+', '+str(Year)
    time_str=str(df_time.hour) + ':' + str(df_time.minute) + ':' +str(df_time.second)
    time=df_time.hour*60*60+df_time.minute*60+df_time.second

    Time_Df=pd.DataFrame(data={'filename': filename,
                               'date': date_str,
                               'Month': month,
                               'Day': Day,
                               'power (mW)': np.nan,
                               'Int_time (s)': np.nan,
                               'accumulations': np.nan,
                               'Mag (X)': np.nan,
                               'duration': np.nan,
                   '24hr_time': time_str,
    'sec since midnight': time,
    'Spectral Center': np.nan
                              }, index=[0])
    return Time_Df

def loop_convert_datastamp_to_metadata(path, files, creation=True, modification=False):
    """ Loops over multiple files to get timestamp the file was created or modified
    using the convert_datastamp_to_metadata function.

    path: str
        Path where spectra files are stored
    files: list
        list of filenames
    creation: bool
        If True, gets timestamp based on creation date of file
    modification: bool
        If True, gets timestamp based on modification date of file

    Returns
    ----------
    df of timestamp, and other columns to have the same format as the WITEC metadata output

   
    
    """
    df_meta=pd.DataFrame([])
    for file in files:
        df_loop=convert_datastamp_to_metadata(path=path, filename=file,
creation=creation, modification=modification)
        df_meta=pd.concat([df_meta, df_loop], axis=0)
        df_meta=df_meta.sort_values(by='24hr_time')
    return df_meta


## Functions to extract things for HORIBA

## HORIBA acquisition time
encode="ISO-8859-1"
def extract_duration_horiba(*, path, filename):
    """ This function extracts the duration from a HORIBA file by finding the line starting with #Acq. """
    fr = open(path+'/'+filename,  'r', encoding=encode)

    while True:
        l=fr.readline()
        if l.startswith('#Acq.'):
            line=l
            break
    return line

def extract_accumulations_horiba(*, path, filename):
    """ This function extracts the accumulations from a HORIBA file by finding the line starting with #Accumu. """
    fr = open(path+'/'+filename,  'r', encoding=encode)

    while True:
        l=fr.readline()
        if l.startswith('#Accumu'):
            line=l
            break
    return line

def extract_objective_horiba(*, path, filename):
    """ This function extracts the objective used from a HORIBA file by finding the line starting with #Object. """
    fr = open(path+'/'+filename,  'r', encoding=encode)

    while True:
        l=fr.readline()
        if l.startswith('#Object'):
            line=l
            break
    return line

def extract_date_horiba(*, path, filename):
    """ This function extracts the date used from a HORIBA file by finding the line starting with #Date. """
    fr = open(path+'/'+filename,  'r', encoding=encode)

    while True:
        l=fr.readline()
        if l.startswith('#Date'):
            line=l
            break
    return line

def extract_spectral_center_horiba(*, path, filename):
    """ This function extracts the spectral center used from a HORIBA file by finding the line starting with #Spectro (cm-¹). """
    fr = open(path+'/'+filename,  'r', encoding=encode)

    while True:
        l=fr.readline()
        if l.startswith('#Spectro (cm-¹)'):
            line=l
            break
    return line

def extract_24hr_time_horiba(*, path, filename):
    """ This function extracts the 24 hr time from a HORIBA file by finding the line starting with #Acquired. """
    fr = open(path+'/'+filename,  'r', encoding=encode)

    while True:
        l=fr.readline()
        if l.startswith('#Acquired'):
            line=l
            break
    return line

def extract_spectraname_horiba(*, path, filename):
    """
    This function extracts the spectral name from HORIBA files
    """
    fr = open(path+'/'+filename,  'r', encoding=encode)

    while True:
        l=fr.readline()
        if l.startswith('#Title'):
            line=l
            break
    return line


def extract_acq_params_horiba(path, filename):
    """ Extracts all relevant acquisition parameters from a HORIBA file, returns as a dataframe. 
    """
    from datetime import datetime
    # Integration time in seconds
    Int_str=extract_duration_horiba(path=path, filename=filename)
    integ=float(Int_str.split()[3])

    #Extracting accumulations
    accums_str=extract_accumulations_horiba(path=path, filename=filename)
    accums=float(accums_str.split("\t")[1].split('\n')[0])

    # Doesnt seem to have, can calculate
    Dur=integ*accums

    # Objective used
    Obj_str=extract_objective_horiba(path=path, filename=filename)
    Obj=Obj_str.split("\t")[1].split('\n')[0]


    date_str=extract_date_horiba(path=path, filename=filename)
    date=date_str.split('\t')[1].split( )[0]
    day=int(date.split('.')[0])
    month=int(date.split('.')[1])
    year=int(date.split('.')[2])
    month_name=calendar.month_name[month]
    Day=datetime.strptime(date, "%d.%m.%Y")
    spec_cen=extract_spectral_center_horiba(path=path, filename=filename)
    spec=float(spec_cen.split('=')[1])

    spec_str=extract_spectral_center_horiba(path=path,
    filename=filename)

    spec=spec_str.split('\t')[1].split('\n')[0]


    time_str=extract_24hr_time_horiba(path=path, filename=filename)
    time=time_str.split(' ')[1].split('\n')[0]

    hour=int(time.split(':')[0])
    minute=int(time.split(':')[1])
    sec=int(time.split(':')[2])

    sec_since_midnight=hour*60*60 + minute*60 + sec

    tes=extract_spectraname_horiba(path=path, filename=filename)
    spec_name=tes.split('\t')[1].split('\n')[0]

    df=pd.DataFrame(data={'filename': filename,
                          'spectral_name': spec_name,
                          'date': date,
                          'Month': month_name,
                          'Day': Day,
                          'power (mw)' : 'no data',
                          'Int_time (s)': integ,
                          'accumulations': accums,
                          'Mag (X)': Obj,
                          'duration': Dur,
                          '24hr_time': time,
                          'sec since midnight': sec_since_midnight,
                          'Spectral_Center': spec}, index=[0])


    return df




def stitch_metadata_in_loop_horiba(AllFiles, path=None):

    """ Stitches acquisition parameters together from the function extract_acq_params_horiba for multiple files
    Parameters
    -------------
    AllFiles: list
        List of all file names

    path: str
        Path where files are found

    Returns
    -------------
    df of aquisitoin parameters
    """
    if path is None:
        path=os.getcwd()

    df=pd.DataFrame([])
    for i in tqdm(range(0, len(AllFiles))):
        file=AllFiles[i]
        one_file=extract_acq_params_horiba(path=path, filename=file)
        df=pd.concat([df, one_file], axis=0)
    df_out=df.reset_index(drop=True)
    return df_out



## Functions to extract metadata from WITEC files (v instrument specific)

def extract_time_stamp_witec(*, path, filename):
    """ Extracts time stamp from a WITEC file
    """

    fr = open(path+'/'+filename,  'r', encoding=encode)


    while True:
        l=fr.readline()
        if l.startswith('Start'):
            line=l
            break
    return line

def extract_laser_power_witec(*, path, filename):
    """ Extracts laser power from a WITEC file
    """
    fr = open(path+'/'+filename,  'r', encoding=encode)


    while True:
        l=fr.readline()
        if l.startswith('Laser'):
            line=l
            break
    return line

def extract_accumulations_witec(*, path, filename):
    """ Extracts accumulations from a WITEC file
    """
    fr = open(path+'/'+filename,  'r', encoding=encode)


    while True:
        l=fr.readline()
        if l.startswith('Number'):
            line=l
            break
    return line


def extract_integration_time_witec(*, path, filename):
    """ Extracts Integration time from a WITEC file
    """

    fr = open(path+'/'+filename,  'r', encoding=encode)

    while True:
        l=fr.readline()
        if l.startswith('Integration'):
            line=l
            break
    return line

def extract_spectral_center_witec(*, path, filename):
    """ Extracts Spectral Center from a WITEC file
    """

    fr = open(path+'/'+filename,  'r', encoding=encode)

    while True:
        l=fr.readline()
        if l.startswith('Spectral'):
            line=l
            break
    return line

def extract_objective_witec(*, path, filename):
    """ Extracts objective magnification from a WITEC file
    """

    fr = open(path+'/'+filename,  'r', encoding=encode)


    while True:
        l=fr.readline()
        if "Magnification" in l:
            line=l
            break
    return line

def extract_duration_witec(*, path, filename):
    """ Extracts analysis duration from a WITEC file
    """

    fr = open(path+'/'+filename,  'r', encoding=encode)


    while True:
        l=fr.readline()
        if l.startswith('Duration'):
            line=l
            break
    return line

def extract_date_witec(*, path, filename):
    """ Extracts date from a WITEC file"""

    fr = open(path+'/'+filename,  'r', encoding=encode)


    while True:
        l=fr.readline()
        if l.startswith('Start Date'):
            line=l
            break

    return line

def checks_if_video_witec(*, path, filename):
    """ Checks if a WITEC file is an image (as doesnt have all metadata)
    """
    fr = open(path+'/'+filename,  'r', encoding=encode)
    l1=fr.readline()
    #print(l1)
    if 'Video' in l1:
        return 'Video'
    else:

        return 'not Video'

def checks_if_imagescan_witec(*, path, filename):
    """ Checks if a WITEC file is an imagescan (as doesnt have all metadata)
    """
    fr = open(path+'/'+filename,  'r', encoding=encode)
    l1=fr.readline()
    #print(l1)
    if 'Scan' in l1:
        return 'Scan'
    else:

        return 'not Scan'

def checks_if_general_witec(*, path, filename):
    """ Checks if a WITEC file is a spectra file with all the right metadata
    """
    fr = open(path+'/'+filename,  'r', encoding=encode)
    l1=fr.readline()
    #print(l1)
    if 'General' in l1:
        return 'General'
    else:
        return 'not General'

## Functions for extracting the metadata from WITEC files

def extract_acq_params_witec(*, path, filename, trupower=False):
    """ This function checks what type of file you have, and if its a spectra file,
    uses the functions above to extract various bits of metadata.

    Parameters
    --------------
    path: str
        Folder where spectra are stored
    filename: str
        Specific filename

    Truepower: bool
        True if your WITEC system has Trupower, else false, as no power in the metadata file

    Returns
    -------------
    power, accums, integ, Obj, Dur, dat, spec
    Values for each acquisition parameters. 
    """
    # Prints what it is, e.g. general if general, video if video
    if path is None:
        path=os.getcwd()

    line_general=checks_if_general_witec(path=path, filename=filename)
    line_video_check=checks_if_video_witec(path=path, filename=filename)
    line_scan=checks_if_imagescan_witec(path=path, filename=filename)

    # If not a
    if line_video_check == "Video":
        power=np.nan
        accums=np.nan
        integ=np.nan
        Obj=np.nan
        Dur=np.nan
        dat=np.nan
        spec=np.nan
    if line_scan == "Scan":
        power=np.nan
        accums=np.nan
        integ=np.nan
        Obj=np.nan
        Dur=np.nan
        dat=np.nan
        spec=np.nan
    if line_general == 'General':
        power=np.nan
        accums=np.nan
        integ=np.nan
        Obj=np.nan
        Dur=np.nan
        dat=np.nan
        spec=np.nan

    # If a real spectra file
    if line_video_check == 'not Video'  and line_scan == "not Scan": #Removed general for berkeley. as witec removed "spectrum from top of file"
        if trupower is True:
            power_str=extract_laser_power_witec(path=path, filename=filename)
            power=float(power_str.split()[3])
        else:
            power=np.nan

        accums_str=extract_accumulations_witec(path=path, filename=filename)
        accums=float(accums_str.split()[3])

        integ_str=extract_integration_time_witec(path=path, filename=filename)
        integ=float(integ_str.split()[3])

        Obj_str=extract_objective_witec(path=path, filename=filename)
        Obj=float(Obj_str.split()[2])

        Dur_str=extract_duration_witec(path=path, filename=filename)
        Dur=Dur_str.split()[1:]

        dat_str=extract_date_witec(path=path, filename=filename)

        dat=dat_str.split(':')[1].split(',',1)[1].lstrip( )

        spec=extract_spectral_center_witec(path=path, filename=filename)
        spec=float(spec.split()[1:][3])

    return power, accums, integ, Obj, Dur, dat, spec




def calculates_time_witec(*, path, filename):
    """ calculates time as seconds after midnight for non video files for WITEC files
    
    """



    # Need to throw out video and peak fit files "general"
    line_general=checks_if_general_witec(path=path, filename=filename)
    line_video_check=checks_if_video_witec(path=path, filename=filename)
    line_scan=checks_if_imagescan_witec(path=path, filename=filename)

    # If not a
    if line_video_check == "Video":
        line3_sec_int=np.nan
        line2=np.nan
    if line_general == 'General':
        line3_sec_int=np.nan
        line2=np.nan
    if line_scan== "Scan":
        line3_sec_int=np.nan
        line2=np.nan
    # If a real spectra file
    if line_video_check == 'not Video' and line_scan == "not Scan": # Had to remove general for berkeley
        line=extract_time_stamp_witec(path=path, filename=filename)


        line2=line.strip('Start Time:\t')
        if 'PM' in line2:
            line3=line2.strip(' PM\n')
            line3_hr=line3.split(':')[0]
            line3_min=re.search(':(.*):', line3).group(1)
            line3_sec=re.search(':(.*)', line2).group(1)[3:5]


        if 'AM' in line2:
            line3=line2.strip(' AM\n')
            line3_hr=line3.split(':')[0]
            line3_min=re.search(':(.*):', line3).group(1)
            line3_sec=re.search(':(.*)', line2).group(1)[3:5]




        # If its any pm after 12, you add 12 hours to the time
        if line3_hr != '12' and 'PM' in line2:
            line3_sec_int=12*60*60+float(line3_hr)*60*60+float(line3_min)*60+float(line3_sec)
        elif line3_hr=='12' and 'AM' in line2:
            line3_sec_int=float(line3_hr)*60*60+float(line3_min)*60+float(line3_sec)-12*60*60
        # If its 12 pm,  then you can just do the maths as normal
        else:
            line3_sec_int=float(line3_hr)*60*60+float(line3_min)*60+float(line3_sec)


    return line3_sec_int, line2

def stitch_metadata_in_loop_witec(*, Allfiles, path, prefix=True, trupower=False):
    """ Stitches together WITEC metadata for all files in a loop using the function
    extract_acq_params_witec and calculates_time_witec, exports as a dataframe

    Parameters
    -----------------
    Allfiles:list
        List of files to fit
    path: str
        Name of folder with files in
    prefix: bool
        If True, removes any characters in the name before the space ' '
    trupower: bool
        Can only be True if you have Trupower on your Witec Raman

    Returns
    -----------
    DataFrame of metadata parameters with a row for each file. 
    """
    if path is None:
        path=os.getcwd()
    # string values
    time_str=[]
    hour_str=[]
    filename_str=[]
    duration_str=[]
    date_str=[]
    month_str=[]
    # Numerical values
    Int_time=np.empty(len(Allfiles), dtype=float)
    objec=np.empty(len(Allfiles), dtype=float)
    time=np.empty(len(Allfiles), dtype=float)

    Day=np.empty(len(Allfiles), dtype=float)
    power=np.empty(len(Allfiles), dtype=float)
    accumulations=np.empty(len(Allfiles), dtype=float)
    spectral_cent=np.empty(len(Allfiles), dtype=float)

    for i in tqdm(range(0, len(Allfiles))):
        filename1=Allfiles[i] #.rsplit('.',1)[0]
        if prefix is True:
            filename=filename1.split(' ')[1:][0]

        else:
            filename=filename1

        #print('working on file' + str(filename1))
        time_num, t_str=calculates_time_witec(path=path, filename=filename1)

        powr, accums, integ, Obj, Dur, dat, spec=extract_acq_params_witec(path=path,
                                                       filename=filename1, trupower=trupower)

        if type(dat)==float:
            if np.isnan(dat):
                date2=dat
        else:
            date2=dat.split(',')[0]
        if type(date2)==float:
            if np.isnan(date2):
                m_str=date2
                Day[i]=date2
        else:
            m_str=date2.split(' ')[0]
            Day[i]=date2.split(' ')[1]
        Int_time[i]=integ
        objec[i]=Obj
        power[i]=powr
        accumulations[i]=accums
        spectral_cent[i]=spec


        time[i]=time_num


        month_str.append(format(m_str))
        time_str.append(format(t_str))
        filename_str.append(format(filename))
        duration_str.append(format(Dur))
        date_str.append(format(dat))







    Time_Df=pd.DataFrame(data={'filename': filename_str,
                               'date': date_str,
                               'Month': month_str,
                               'Day': Day,
                               'power (mW)': power,
                               'Int_time (s)': Int_time,
                               'accumulations': accumulations,
                               'Mag (X)': objec,
                               'duration': duration_str,
                   '24hr_time': time_str,
    'sec since midnight': time,
    'Spectral Center': spectral_cent
                              })

    Time_Df_2=Time_Df[Time_Df['sec since midnight'].notna()].reset_index(drop=True)


    Time_Df_2=Time_Df_2.sort_values('sec since midnight', axis=0, ascending=True)
    print('Done')

    # Check if the person worked after midnight (lame)
    dates_unique=Time_Df_2['date'].unique()
    month_unique=Time_Df_2['Month'].unique()
    if len(dates_unique)>1:
        print('Oof, try not to work after midnight!')

    if len(dates_unique)>1 and len(month_unique)==1:
        min_date=np.min(Time_Df_2['date'])
        max_date=np.max(Time_Df_2['date'])

        Time_Df_2.loc[Time_Df_2['date']==max_date, 'sec since midnight' ]= Time_Df_2['sec since midnight']+24*60*60

    # If youve crossed a month boundary, the minimum date is the one you did afterwards.
    if len(dates_unique)>1 and len(month_unique)>1:
        min_date=np.min(Time_Df_2['date'])
        max_date=np.max(Time_Df_2['date'])
        Time_Df_2.loc[Time_Df_2['date']==min_date, 'sec since midnight' ]= Time_Df_2['sec since midnight']+24*60*60



    return Time_Df_2

## Getting nice names from any file types

def extracting_filenames_generic(*, names, prefix=False,
    str_prefix=None, suffix=False, CRR_filter=True,
    str_suffix=None,
   file_type=None):
    """
    Takes filenames from a panda series (e.g., a column of a dataframe of metadata), outputs a numpy array that is consistent with the same function for 
    spectra, to allow stitching of spectra and metadata. 


    Parameters
    -----------
    names: Pandas.Series 
        Series of sample names, e.g., from 'filename' column of metadata output

    prefix: bool
        if True, has a number before the file name

    str_prefix: str
        The string separating the prefix from the file name (e.g. if file is 01 test, str_prefix=" ")

    suffix: bool
        if True, has a number or name after the filename

    str_suffix: str
        The string separating the filename from the suffix

    file_type: str
        The file extension, e.g., '.csv'

    Returns
    -----------------
    np.array of names, with prefix, suffix and filetype stripped away

    """
    

    if isinstance(names, list):
        names_df=pd.DataFrame(data={'name': names})
        names=names_df['name']

    if CRR_filter is True:
        names = names.str.replace('_CRR_DiadFit', '')

    # if prefix is True:
    #     names=names.str.split(str_prefix).str[1]

    # if suffix is True:
    #     names=names.str.split(str_suffix).str[1]

    # if file_type is not None:
    #     names=names.str.replace(file_type, '')

    file_m=list(names)

    file_m=np.empty(len(names), dtype=object)
    for i in range(0, len(names)):
        name=names.iloc[i]
        # If no prefix or suffix to remove, simple
        if prefix is False and suffix is False:
            file_m[i]=name
            #print(file_m)

        else:
            if prefix is True:
                str_nof_name=name.split(str_prefix, maxsplit=1)[1:]
                # print(str_nof_name)
                # print(type(str_nof_name))
            if prefix is False:
                str_nof_name=name

            if suffix is True:
                file_m[i]=str_nof_name.split(str_suffix, maxsplit=1)[0]
            if suffix is False:

                file_m[i]=str_nof_name[0]

        if file_type in file_m[i]:
            file_m[i]=file_m[i].replace(file_type, '')



    if len(file_m)!=len(pd.Series(file_m).unique()):
        file_m_s=pd.Series(file_m)
        print('OOPS. at least one of your file name is duplicated go back to your spectra, you named a file twice, this will confuse the stitching ')
        print(file_m_s[file_m_s.duplicated()])
    else:
        print('good job, no duplicate file names')
        #raise Exception('Duplicate file')

    return file_m

# These are largely redundant.  
def extract_temp_Aranet(df):
    """ Extracts temperature data from the aranet
    """
    TD=str(Temp['Time(dd/mm/yyyy)'])
    hour=np.empty(len(Temp), dtype=object)
    date=np.empty(len(Temp), dtype=object)
    time=np.empty(len(Temp), dtype=object)
    minutes=np.empty(len(Temp), dtype=object)
    seconds=np.empty(len(Temp), dtype=object)
    secs_sm=np.empty(len(Temp), dtype=object)
    for i in range(0, len(Temp)):
        TD=str(Temp['Time(dd/mm/yyyy)'].iloc[i])
        date[i]=TD.split(' ')[0]
        time[i]=TD.split(' ')[1]
        hour[i]=time[i].split(':')[0]
        minutes[i]=time[i].split(':')[1]
        seconds[i]=time[i].split(':')[2]
        secs_sm[i]=float(hour[i])*60*60+float(minutes[i])*60+float(seconds[i])

    return secs_sm


## Stitching together looped and individually fitted spectra


def get_ind_saved_files(*, path, ID_str='ind_fit_', sort=True, file_ext='.csv'):

    Allfiles = [f for f in listdir(path) if isfile(join(path, f))]
    ind_files=[item for item in Allfiles if ID_str in item and file_ext in item]

    if sort is True:
        ind_files=sorted(ind_files)
    return ind_files



def stitch_loop_individual_fits(*, fit_individually=True,
    saved_spectra_path, looped_df,
  ID_str='ind_fit_', sort=True,  file_ext='.csv'):

    df_Dense=looped_df.copy()

    ind_files=get_ind_saved_files(path=saved_spectra_path,
        sort=sort, ID_str=ID_str, file_ext=file_ext)

    if fit_individually:
        df_Dense2 = pd.DataFrame([])
        for file in ind_files:
            data=pd.read_csv(file)
            df_Dense2 = pd.concat([df_Dense2, data], axis=0)

        df_Dense_loop=df_Dense.reset_index(drop=True)
        cols=list(df_Dense_loop.columns)
        for file in df_Dense_loop['filename'].unique():
            if file in df_Dense2['filename'].unique():
                df_Dense2_fill=df_Dense2.loc[df_Dense2['filename']==file]
                df_Dense_loop.loc[df_Dense_loop['filename']==file, cols]= df_Dense2_fill[cols].values

            else:
                df_Dense_Fill=df_Dense_loop.loc[df_Dense_loop['filename']==file]
                df_Dense_loop.loc[df_Dense_loop['filename']==file, cols]=df_Dense_Fill[cols]
                #df_Dense_loop.loc[df_Dense_loop['filename']==file, 'filename']= file + str(' ind_fit')

        df_Dense_Combo=df_Dense_loop.copy()
    else:
        df_Dense_Combo=df_Dense

    return df_Dense_Combo
