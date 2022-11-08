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

encode="ISO-8859-1"

## Functions for getting file names


def get_Ne_files(path, ID_str='Ne', file_ext='txt', exclude_str=None, sort=True):
    """ This function takes a user path, and extracts all files which contain the ID_str

    Parameters
    -----------

    path: str
        Folder user wishes to read data from
    sort: bool
        If true, sorts files alphabetically
    ID_str: str
        Finds all files containing this string (e.g. Ne, NE)
    exclude_str: str
        Excludes files with this string in the name
    file_ext: str
        Gets all files of this format only (e.g. txt)


    Returns
    -----------
    Returns file names as a list.

    """
    Allfiles = [f for f in listdir(path) if isfile(join(path, f))]
    Ne_files=[item for item in Allfiles if ID_str in item and file_ext in item and exclude_str not in item]

    if sort is True:
        Ne_files=sorted(Ne_files)
    return Ne_files


def get_diad_files(path, sort=True, file_ext='txt', exclude_str='Ne', exclude_str_2='ne', exclude_str_3='Si_wafer', exclude_type='.png'):
    """ This function takes a user path, and extracts all files which dont contain the excluded string and type

    Parameters
    -----------

    path: str
        Folder user wishes to read data from
    sort: bool
        If true, sorts files alphabetically
    file_ext: str
        File format. Default txt, could also enter csv etc.
    exclude_str: str
        Excludes files with this string in their name. E.g. if exclude_str='Ne' it will exclude Ne lines
    exclude_type: str
        Excludes files of this type, e.g. exclude_type='png' gets rid of image files.

    Returns
    -----------
    Returns file names as a list.

    """


    Allfiles = [f for f in listdir(path) if isfile(join(path, f))]


    Diad_files=[item for item in Allfiles if exclude_str not in item and file_ext in item and exclude_str_2 not in item and exclude_str_3 not in item and exclude_type not in item]

    if sort is True:

        Diad_files2=sorted(Diad_files)
    else:
        Diad_files2=Diad_files
    return Diad_files2



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

    return df_in

## Reading different file formats
def read_HORIBA_to_df(*,  path=None, filename):
    """ This function takes in a HORIBA .txt. file with headers with #, and looks down to the row where Data starts (no #),
    and saves this to a new file called pandas_.... old file. It exports the data as a pandas dataframe

    Parameters
    -----------

    path: str
        Folder user wishes to read data from

    filename: str
        Specific file being read


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
        print(filename)
        df=pd.read_csv('pandas2_'+filename, sep="\t", header=None)
    else:
        print(filename)
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
        print(filename)
        df=pd.read_csv('pandas2_'+filename, sep="\t")
    else:
        print(filename)
        df=pd.read_csv(path+'/'+'Peak_fits_txt'+'/'+'pandas2_'+filename, sep="\t")

    return df



## Instrument specific metadata things
## Functions to extract metadata from WITEC files (v instrument specific)

def extract_time_stamp_witec(*, path, filename):
    """ Extracts time stamps
    """

    fr = open(path+'/'+filename,  'r', encoding=encode)


    while True:
        l=fr.readline()
        if l.startswith('Start'):
            line=l
            break
    return line

def extract_laser_power_witec(*, path, filename):
    """ Extracts laser power
    """
    fr = open(path+'/'+filename,  'r', encoding=encode)


    while True:
        l=fr.readline()
        if l.startswith('Laser'):
            line=l
            break
    return line

def extract_accumulations(*, path, filename):
    """ Extracts accumulations
    """
    fr = open(path+'/'+filename,  'r', encoding=encode)


    while True:
        l=fr.readline()
        if l.startswith('Number'):
            line=l
            break
    return line


def extract_Integration_Time(*, path, filename):
    """ Extracts Integration time
    """

    fr = open(path+'/'+filename,  'r', encoding=encode)

    while True:
        l=fr.readline()
        if l.startswith('Integration'):
            line=l
            break
    return line

def extract_Spectral_Center(*, path, filename):
    """ Extracts Spectral Center
    """

    fr = open(path+'/'+filename,  'r', encoding=encode)

    while True:
        l=fr.readline()
        if l.startswith('Spectral'):
            line=l
            break
    return line

def extract_objective(*, path, filename):
    """ Extracts objective magnification
    """

    fr = open(path+'/'+filename,  'r', encoding=encode)


    while True:
        l=fr.readline()
        if "Magnification" in l:
            line=l
            break
    return line

def extract_duration(*, path, filename):
    """ Extracts analysis duration
    """

    fr = open(path+'/'+filename,  'r', encoding=encode)


    while True:
        l=fr.readline()
        if l.startswith('Duration'):
            line=l
            break
    return line

def extract_date(*, path, filename):
    """ Extracts date"""

    fr = open(path+'/'+filename,  'r', encoding=encode)


    while True:
        l=fr.readline()
        if l.startswith('Start Date'):
            line=l
            break
    return line

def checks_if_video(*, path, filename):
    """ Checks if file is an image (as doesnt have all metadata)
    """
    fr = open(path+'/'+filename,  'r', encoding=encode)
    l1=fr.readline()
    #print(l1)
    if 'Video' in l1:
        return 'Video'
    else:

        return 'not Video'

def checks_if_imagescan(*, path, filename):
    """ Checks if file is an imagescan (as doesnt have all metadata)
    """
    fr = open(path+'/'+filename,  'r', encoding=encode)
    l1=fr.readline()
    #print(l1)
    if 'Scan' in l1:
        return 'Scan'
    else:

        return 'not Scan'

def checks_if_general(*, path, filename):
    """ Checks if file is a spectra file with all the right metadata
    """
    fr = open(path+'/'+filename,  'r', encoding=encode)
    l1=fr.readline()
    #print(l1)
    if 'General' in l1:
        return 'General'
    else:
        return 'not General'

## Functions for extracting the metadata from WITEC files

def extract_acq_params(*, path, filename, trupower=False):
    """ This function checks what type of file you have, and if its a spectra file,
    uses the functions above to extract various bits of metadata
    """
    # Prints what it is, e.g. general if general, video if video
    if path is None:
        path=os.getcwd()

    line_general=checks_if_general(path=path, filename=filename)
    line_video_check=checks_if_video(path=path, filename=filename)
    line_scan=checks_if_imagescan(path=path, filename=filename)

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

        accums_str=extract_accumulations(path=path, filename=filename)
        accums=float(accums_str.split()[3])

        integ_str=extract_Integration_Time(path=path, filename=filename)
        integ=float(integ_str.split()[3])

        Obj_str=extract_objective(path=path, filename=filename)
        Obj=float(Obj_str.split()[2])

        Dur_str=extract_duration(path=path, filename=filename)
        Dur=Dur_str.split()[1:]

        dat_str=extract_date(path=path, filename=filename)
        dat=dat_str.split(':')[1].split(',',1)[1].lstrip( )

        spec=extract_Spectral_Center(path=path, filename=filename)
        spec=float(spec.split()[1:][3])

    return power, accums, integ, Obj, Dur, dat, spec




def calculates_time(*, path, filename):
    """ calculates time for non video files for WITEC files"""



    # Need to throw out video and peak fit files "general"
    line_general=checks_if_general(path=path, filename=filename)
    line_video_check=checks_if_video(path=path, filename=filename)
    line_scan=checks_if_imagescan(path=path, filename=filename)

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



        if line3_hr != '12' and 'PM' in line2:
            line3_sec_int=12*60*60+float(line3_hr)*60*60+float(line3_min)*60+float(line3_sec)
        else:
            line3_sec_int=float(line3_hr)*60*60+float(line3_min)*60+float(line3_sec)


    return line3_sec_int, line2

def stitch_metadata_in_loop(*, Allfiles=None, path=None, prefix=True, trupower=False):
    """ Stitches together WITEC metadata for all files in a loop
    """
    if path is None:
        path=os.getcwd()
    # string values
    time_str=[]
    filename_str=[]
    duration_str=[]
    date_str=[]
    # Numerical values
    Int_time=np.empty(len(Allfiles), dtype=float)
    objec=np.empty(len(Allfiles), dtype=float)
    time=np.empty(len(Allfiles), dtype=float)
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
        time_num, t_str=calculates_time(path=path, filename=filename1)

        powr, accums, integ, Obj, Dur, dat, spec=extract_acq_params(path=path,
                                                       filename=filename1, trupower=trupower)



        Int_time[i]=integ
        objec[i]=Obj
        power[i]=powr
        accumulations[i]=accums
        spectral_cent[i]=spec


        time[i]=time_num
        time_str.append(format(t_str))
        filename_str.append(format(filename))
        duration_str.append(format(Dur))
        date_str.append(format(dat))




    Time_Df=pd.DataFrame(data={'filename': filename_str,
                               'date': date_str,
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

    return Time_Df_2

## Getting nice names from any file types

def extracting_filenames_generic(*, names, prefix=False,
    str_prefix=None, suffix=False,
    str_suffix=None,
   file_type=None):
    """
    Takes filenames from metadata, and makes something consistent with spectra

    Parameters
    -----------
    names: Pandas.Series of sample names from 'filename' column of metadata output

    prefix: bool
        if True, has a number before the file name

    str_prefix: str
        The string separating the prefix from the file name

    suffix: bool
        if True, has a number or name after the filename

    str_suffix: str
        The string separating the filename from the suffix

    file_type: str
        The file extension, e.g., '.csv'

    """


    file_m=np.empty(len(names), dtype=object)
    for i in range(0, len(names)):
        name=names.iloc[i]
        # If no prefix or suffix to remove, simple
        if prefix is False and suffix is False:
            file_m[i]=name

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
        print('duplicates')
        print(file_m_s[file_m_s.duplicated()])
        print('OOPS. at least one of your file name is duplicated go back to your spectra, you named a file twice, this will confuse the stitching ')
        raise Exception('Duplicate file')

    return file_m

def extract_temp_Aranet(df):
    """ Extracts temperature data from the aranet
    """
    TD=df['Time(dd/mm/yyyy)']
    hour=np.empty(len(Temp), dtype=object)
    date=np.empty(len(Temp), dtype=object)
    time=np.empty(len(Temp), dtype=object)
    minutes=np.empty(len(Temp), dtype=object)
    seconds=np.empty(len(Temp), dtype=object)
    secs_sm=np.empty(len(Temp), dtype=object)
    for i in range(0, len(Temp)):
        date[i]=TD.iloc[i].split(' ')[0]
        time[i]=TD.iloc[i].split(' ')[1]
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
