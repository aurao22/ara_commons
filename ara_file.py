import pandas as pd
from os import listdir, remove
from os.path import isfile, join, isdir, exists, getsize
from pathlib import Path  
from glob import glob

from util_file import *

from tensorflow import compat

def del_corrupt_img(dir_path, include_sub_dir=0, verbose=0):
    
    removed_files = []
    fichiers = get_dir_files(dir_path=dir_path, include_sub_dir=include_sub_dir, verbose=verbose-1)
        
    for fname in fichiers:
        is_jfif = False
        fpath = join(dir_path, fname)
        try:
            fobj = open(fpath, "rb")
            is_jfif = compat.as_bytes("JFIF") in fobj.peek(10)
        finally:
            fobj.close()

        if not is_jfif:
            removed_files.append(fpath)
            # Delete corrupted image
            remove_file(fpath)

    if verbose: print(f"Deleted {len(removed_files)} images.")
    return removed_files


def list_dir_files(dir_path, endwith=None, verbose=0):
    end = "*"
    if endwith is not None:
        end = endwith.replace(".", "")

    files = glob.glob(f"{dir_path}/*.{end}")
    return files

def get_file_list_df(file_df_path, force_reloading=False, verbose=0):

    file_df = None
    # On vérifie que le fichier existe
    if exists(file_df_path) and getsize(file_df_path) > 0 and not force_reloading:
        if verbose > 0: print(file_df_path, "Exist")
        # Chargement de la DF fichier
        file_df = pd.read_csv(file_df_path, sep=",")
    else:
        if verbose > 0: print(file_df_path, "NOT exist")
        file_df = pd.DataFrame({'FileNames' : []})
    if verbose > 0: print("Ever loaded files:", file_df.shape)
    return file_df

def get_files_to_load(file_df, source_data_path=None, suffix=".csv", new_files=None, verbose=0):

    if new_files is None:
        new_files = [f for f in listdir(source_data_path) if isfile(join(source_data_path, f)) and f.endswith(suffix)]
    
    if verbose > 1: 
        print("Listed files:", new_files)
        print("Files ever loaded:", file_df)

    new_files_df = pd.DataFrame({'FileNames' : new_files})
    new_files_df = new_files_df[~new_files_df['FileNames'].isin(file_df['FileNames'])]

    if verbose > 1: print("Files to loaded:")
    if verbose > 0: print(new_files_df)
    return new_files_df


def add_loaded_files(file_df, loaded_files, verbose=0):
    file_df = pd.concat([file_df, loaded_files])
    if verbose:
        print("Proceeded files:", file_df.shape)
        print(file_df)
    return file_df


def save_files(file_df, loaded_files, verbose=0):
    file_df = pd.concat([file_df, loaded_files])
    if verbose:
        print("Proceeded files:", file_df.shape)
        print(file_df)
    return file_df

def remove_file(file_path):
    try:
        if path.exists(file_path):
            return remove(file_path)
    except OSError as e:
        print(e)


import requests

def wikipedia_page(title):
    '''
    This function returns the raw text of a wikipedia page 
    given a wikipedia page title
    '''
    params = { 
        'action': 'query', 
        'format': 'json', # request json formatted content
        'titles': title, # title of the wikipedia page
        'prop': 'extracts', 
        'explaintext': True
    }
    # send a request to the wikipedia api 
    response = requests.get(
         'https://en.wikipedia.org/w/api.php',
         params= params
     ).json()

    # Parse the result
    page = next(iter(response['query']['pages'].values()))
    # return the page content 
    if 'extract' in page.keys():
        return page['extract']
    else:
        return "Page not found"

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#                                              TESTS
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

if __name__ == "__main__":
    root = r"C:/Users/User\WORK\workspace-ia/PROJETS/projet_cat_or_dog\dataset/"

    for p in ['training_set', 'validation_set']:
        path1=join(root, p)
        print(path1)
        sub_d = get_sub_dir(path1, verbose=0)
        for s in sub_d:
            print(s)

        for cat in ['cat', 'dog']:   
            path=join(root, p, cat)
            corrupted = corrupted_img(path=path, verbose=0)
            for s in corrupted:
                print(s)