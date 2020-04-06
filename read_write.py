import mne
import numpy as np
import scipy as sp
import pandas as pd
import pickle
from itcfpy.spatial import make_bip_coords
from itcfpy.read_write import loadmat
import os
import glob
import scipy.io as sio
import sys
from pathlib import Path
import numpy as np

def get_dataset_info(dataset_dir, filename='*', n_levels=0, dir_pattern=None):
    if not Path(dataset_dir).is_dir():
        raise FileNotFoundError(dataset_dir)

    data_info = []
    data_path = []

    if dir_pattern is None:
        dir_pattern = n_levels * '**/'
    pattern = os.path.join(dataset_dir,dir_pattern,filename)
        
    files = glob.glob(pattern)

    for file in files:
        s = file.split('/')
        parents = s[-(n_levels+1):-1]
        filename = s[-1].split('.')[:-1]
        data_info.append(parents + filename)
        data_path.append(file)

    print(f'{len(data_info)} files found matching {pattern}.')
    return np.array(data_path), np.array(data_info)

def save_pickle(fname, variable):
    fname = str(fname)
    if not fname.endswith('.pkl'):
        fname += '.pkl'
    output = open(fname, 'wb')
    pickle.dump(variable, output)
    output.close()


def load_pickle(fname):
    fname = str(fname)
    if not fname.endswith('.pkl'):
        fname += '.pkl'
    pkl_file = open(fname, 'rb')
    variable = pickle.load(pkl_file)
    pkl_file.close()
    return variable

def loadmat(fname):
    """
    this function should be called instead of direct spio.loadmat
    as it cures the problem of not properly recovering python dictionaries
    from mat files. It calls the function check keys to cure all entries
    which are still mat-objects
    """
    fname = str(fname)
    data = sio.loadmat(fname, struct_as_record=False, squeeze_me=True)
    return _check_keys(data)


def _check_keys(dict_in):
    """
    checks if entries in dictionary are mat-objects. If yes
    todict is called to change them to nested dictionaries
    """
    for key in dict_in:
        if isinstance(dict_in[key], sio.matlab.mio5_params.mat_struct):
            dict_in[key] = _todict(dict_in[key])
    return dict_in


def _todict(matobj):
    """
    A recursive function which constructs from matobjects nested dictionaries
    """
    dict = {}
    for strg in matobj._fieldnames:
        elem = matobj.__dict__[strg]
        if isinstance(elem, sio.matlab.mio5_params.mat_struct):
            dict[strg] = _todict(elem)
        else:
            dict[strg] = elem
    return dict
