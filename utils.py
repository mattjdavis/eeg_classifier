import os
import mne
import numpy as np
import pandas as pd
import glob
import matplotlib.pyplot as plt
from pathlib import Path

import scipy.stats as stats
import scipy.signal as signal



def info_to_df(info_path,return_channels = False):
    """Loads summary text file from single subject from CHB-MIT dataset
    
    Splits text files into chunks (text separated by blank lines in summary file). Then parses
    each chunk to extract file/siezure information. Code could be modified to return channel
    info.
    
    Parameters
    ----------
    info_path : str
       Path to summary file (.txt)
        
    Returns
    -------
    df : pd.DataFrame
        Summary info in a dataframe
    """
    with open(info_path, 'r') as infile:
            info_string = infile.read()

    info_array = info_string.split('\n')

    chunked_list = []
    chunk=[]

    for line in info_array:
        if line:
            chunk.append(line)
        elif chunk: # this prevents empty chunks
            chunked_list.append(chunk)
            chunk=[]
    
    file_list = [x for x in chunked_list if 'File Name' in x[0]] # filter only files
    channel_chunk = [x for x in chunked_list if 'Channels in EDF Files' in x[0]]

    ds = []

    channels=[]
    for line in channel_chunk[0][2:]:
        string = line.split(': ')
        channels.append(string[1])


    for file in file_list:
        d={}
        for f in file[:4]:
            string = f.split(': ')
            d.update({string[0].replace(' ','_').lower(): string[1]})
            
            
        d.update({'subject':d['file_name'].split('_')[0]})
        
        # read seizures to dict (can do any number of seizures)
        times = []
        for f in file[4:]:
            string = f.split(': ')
            sub = string[1].split(' ')
            times.append(sub[0])
        if times:
            times = np.reshape(times,(int(len(times)/2),2))

        d.update({'seizure_times':times})

        
        ds.append(d)
    
    if return_channels:
        return pd.DataFrame(ds),channels
    else:
        return pd.DataFrame(ds)



def get_summaries(data_dir):
    """Loads all subjects summary file contents into a dataframe. Includes seizures info.
    
    Parameters
    ----------
    data_dir : str
       Path to data CHB-MIT (str or Path object)
        
    Returns
    -------
    df_info : pd.DataFrame
        Summary info dataframe
    """
    
    files = glob.glob(str(data_dir) + "*/*/*.txt")
    dfs = []
    for f in files:
        
        if 'chb24' not in f: # skip odd summary for now
            df = info_to_df(f)

            df["folder"] = Path(f).parent
            
            #df["file_path"] = f
            dfs.append(df)

    df_info = pd.concat(dfs)
    df_info.reset_index(drop=True,inplace=True)
    return df_info



def make_stats_df(X,stat_label,ch_names):
    """ turn stats array in DataFrame with stats_label column names
    
    Args:
        X (ndarray): data matrix (segments x time)
        stat_label (string): name of the stat to append to col label
    Returns:
        df (pd.DataFrame)
    """
    df = pd.DataFrame(X, columns=[stat_label + '_' + x for x in ch_names])

    return df

def calc_segment_stats(raw,win_length=5,channels=None):
    """Calculates various statistics for EEG signals.

    Args:
        raw (nme) TODO
        win_lenth (int): length of window for each segment (seconds)
        Channels (list): list of strings, which channels to keep, if None keep all

    Returns:
        df_out (pd.DataFrame): segments x (channels * number of stats)

    """
    # get ndarray from raw object
    if channels:
        data, times = raw.get_data(return_times=True,picks=channels)
        ch_names=channels
    else:
        data, times = raw.get_data(return_times=True)
        ch_names=raw.info['ch_names']

    # get info
    fs = raw.info['sfreq']
    n_segments = int(data.shape[1]/(win_length*fs))

    cut = int(n_segments*win_length*fs)
    data=data[:,:cut] # remove last chunk of samples
    D = np.reshape(data,(data.shape[0],n_segments,-1)) # channel x segment x time


    # calc stats for segments
    m = np.mean(D,axis=2).T
    v = np.var(D,axis=2).T
    s = stats.skew(D,axis=2).T
    k = stats.kurtosis(D,axis=2).T
    sd = np.std(D,axis = 2).T

    zerox = (np.diff(np.sign(D)) != 0).sum(axis=2).T # count sign changes/zero crossings
    max_min = (np.max(D,axis=2) - np.min(D,axis=2)).T # peak-to-peak voltage

    # make data frames
    stats_list = [m,v,s,k,sd,zerox,max_min]
    labels = ["mean","variance","skew","kurtosis","std","zerox","max-min"]
    df_out = pd.concat([make_stats_df(x,y,ch_names) for x,y in zip(stats_list,labels)],axis=1)

    return df_out

def process_raw_chb_data(data_dir, path, win_length=5, include_ictal=True, preictal_window=None,channels=[]):
    """Convert raw CHB-MIT data into processed pd.DataFrames with ictal labels
    
    Processing means extracting statistics of raw signals across time windows, defined by win_length


    Args:
        data_dir (str): directory where all CHB data live
        path (pathlib.Path): path to CHB-MIT file # TODO: dont assume path file
        win_length (int): (seconds)
        include_ictal (bool):
        prectical_window (int): If None, do not incluse preictal labels, else window length in (seconds) 
    Returns:
        df_stats (pf.DataFrame):
        
    """
    # t8-p8 is duplicated, - and . are dummy channel names
    raw = mne.io.read_raw_edf(path, exclude=['-', 'T8-P8', '.'], verbose=False, preload=True)
    
    df_stats = calc_segment_stats(raw, win_length,channels=channels)


    fs = raw.info['sfreq']


    df_info = get_summaries(data_dir) # TODO: can just load this df once somewhere else and save

    
    ## siezure labels
    if include_ictal:

        # init labels vector
        y = np.zeros(len(raw))

        fn = path.name
        row = df_info.query("file_name== @fn")
        

        if int(row.number_of_seizures_in_file) > 0:

            # for loop allows multiple siezures in file
            for seizures in row.seizure_times.values[0]:
                s, e = (seizures.astype('int') * fs).astype(int)
                y[s:e] = 1

        
        n_segments = int(len(raw)/(win_length*fs)) # dup from calc_semgment_stats()
        cut = int(n_segments*win_length*fs)
        y=y[:cut] # remove last chunk of samples

        y = np.reshape(y,(n_segments,-1))
        y = np.where(np.sum(y,axis=1)>0,1,0) # could change the 1st zero to threshold how many samples to count as siezure

        df_stats["ictal"] = y

    # TODO
    # preictal labels

    df_stats['subject'] = path.parts[-2]
    df_stats.astype('category')
    

    return df_stats
