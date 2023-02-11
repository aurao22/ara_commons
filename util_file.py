
# %% import
from os.path import join, exists, isfile
from os import remove, rename, listdir
from pathlib import Path
import shutil
import glob
import pandas as pd
from tqdm import tqdm
import json

import sys
from os import getcwd
from os.path import join

from util_print import *

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# DIR GENERIC FUNCTIONS
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# %% remove_file_if_exist
def remove_file_if_exist(file_path, backup_file=False, verbose=0):
    """
    Remove file

    Args:
        file_path (str): the file path (inlude file name)
        backup_file (bool, optional): if True save the previous file with .backup. Defaults to False.
        verbose (int, optional): Log level. Defaults to 0.
    """
    if (exists(file_path)):
        if backup_file:
            if (exists(str(file_path)+".backup")):
                remove(str(file_path)+".backup")
            rename(str(file_path), str(file_path)+".backup")
        else:
            remove(file_path)

def remove_dir_if_exist(folder_path, empty_only=True, verbose=0):
    
    if exists(folder_path):
        all_files = glob.glob(join(folder_path , "*.*"))
        if not empty_only or all_files is None or len(all_files)==0:
            try:
                remove(folder_path)
            except Exception as error:
                if verbose>0:
                    warn('remove_dir_if_exist', f"impossible to remove dir {folder_path} => {error}")

def create_parent_dir(file_path, verbose=0):
    """Création du répertoire parent s'il n'existe pas

    Args:
        file_path (_type_): _description_
        verbose (int, optional): _description_. Defaults to 0.
    """
    file_name = _file_name(file_path)
    parent_dir = file_path[:-(len(file_name))]
    if parent_dir.endswith('\\') or  parent_dir.endswith('/'):
        parent_dir = parent_dir[:-1]
    creat_dir(dest_path=parent_dir, verbose=verbose)
        

def creat_dir(dest_path, verbose=0):
    # Création du répertoire s'il n'existe pas
    if dest_path is None or len(dest_path.strip()) > 0:   
        base = Path(dest_path)
        base.mkdir(exist_ok=True)

def get_dir_files(dir_path, endwith=None, verbose=0):
    """
    List all file in the directory

    Args:
        dir_path (str): _description_
        endwith (str, optional): the end of the file name. Defaults to None.
        verbose (int, optional): log level. Defaults to 0.

    Returns:
        list: list of file path
    """
    fichiers = None
    if endwith is not None:
        fichiers = [join(dir_path,f) for f in listdir(dir_path) if isfile(join(dir_path, f)) and f.endswith(endwith)]
    else:
        fichiers = [join(dir_path,f) for f in listdir(dir_path) if isfile(join(dir_path, f))]
    
    return  sorted(fichiers)

def csv_file(dir, verbose=0):
    res = get_dir_files(dir, endwith=".csv", verbose=verbose)    
    if res is not None and len(res)==1:
        return res[0]
    else:
        return res
    
def pickle_file(dir, verbose=0):
    res = get_dir_files(dir, endwith=".pickle", verbose=verbose)    
    if res is not None and len(res)==1:
        return res[0]
    else:
        return res

def get_dir_name(dir_path, endwith=None, verbose=0):
    """Liste les noms de répertoires contenu dans le répertoire reçu

    Args:
        dir_path (str): path du répertoire à scanner
        endwith (str, optional): chaine recherchée en fin de nom. Defaults to None.
        verbose (int, optional): log level. Defaults to 0.

    Returns:
        list(str): _description_
    """
    dirs = None
    if endwith is not None:
        dirs = [f for f in listdir(dir_path) if isdir(join(dir_path, f)) and f.endswith(endwith)]
    else:
        dirs = [f for f in listdir(dir_path) if isdir(join(dir_path, f))]
    return dirs

def get_sub_dir(dir_path, verbose=0):
    from glob import glob
    return glob(dir_path+ "/*/", recursive = True)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~µ
# READ FILES GENERIC FUNCTIONS
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def file_name(file_path):
    return _file_name(file_path=file_path)


def _file_name(file_path):
    file_name=file_path.split("/")[-1]
    file_name=file_name.split("\\")[-1]
    return file_name

def get_file_name_without_path_and_ext(file):
    """Supprime le chemin et l'extension pour ne retourner que le nom du fichier

    Args:
        file (str or list(str)): fichier ou liste de fichiers

    Returns:
        str or list(str): nom du fichier
    """
    res = None
    if isinstance(file, str):
        res = Path(file).stem
    elif isinstance(file, list):
        res = [Path(text).stem for text in file]
    return res

def read_json_file(file_path, verbose=0):
    short_name = "read_json_file"
    res = None
    if file_path is not None:
        if (exists(file_path) and isfile(file_path)):
            with open(file_path) as json_file:
                res = json.load(json_file)
        elif verbose > 0:
            info(short_name,f"{file_path} doesn't exist")
    else:
        msg = f"file_path attribute is missing"
        if verbose > 0:
            warn(short_name,msg)
        raise AttributeError(msg)
    return res

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# WRITE FILES GENERIC FUNCTIONS
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def write_json_file(file_path, datas, remove_if_exist=True, verbose=0):
    short_name = "write_json_file"
    file_name = _file_name(file_path)
    
    # Suppression si existe
    if remove_if_exist:
        remove_file_if_exist(file_path=file_path, backup_file=verbose)
    
    with open(file_path, 'w') as fp:
        json.dump(datas, fp, indent=4)
        if verbose > 0:
            info(short_name,f"{file_name}  --> {surligne_text('SAVED')}")
    return file_path
        
# %% write_file
def write_file(dest_path, lines, file_name="",remove_if_exist=True, verbose=0):
    """
    Write the file

    Args:
        dest_path (str): destination path (directory)
        file_name (str): json file name
        lines (iterable) : list
        remove_if_exist (bool, optional): if True check if the file ever exist and remove it. Defaults to True.
        verbose (int, optional): log level. Defaults to 0.
    """
    short_name = "write_file"
    if lines is not None and len(lines)>0:
        # Création du répertoire s'il n'existe pas
        if file_name is not None and len(file_name.strip()) > 0:   
            creat_dir(dest_path=dest_path, verbose=verbose)
            dest_file_path = join(dest_path,file_name)           
        else:
            create_parent_dir(file_path=dest_path)
            dest_file_path = dest_path
            file_name = _file_name(dest_path)
        
        if not exists(dest_file_path) or isfile(dest_file_path):
            # Suppression si existe
            if remove_if_exist:
                remove_file_if_exist(file_path=dest_file_path, backup_file=verbose)
        
            # On convertit ce qu'on reçoit en chaine pour éviter des erreurs si ce qui est reçu n'est pas une chaine
            str_lines = []
            for item in lines:
                str_lines.append(str(item))
        
            end_line_need = '\n' if '\n' not in str_lines[0] else '' 
            
            with open(dest_file_path, 'w') as outfile:
                outfile.write(end_line_need.join(str_lines))
            
            if verbose>0:
                info(short_name, f"{file_name} file {surligne_text('SAVED')}")
            return dest_file_path
        else:
            raise AttributeError("This is a directory path and not a file path")
    elif verbose>0:
        info(short_name, "no content to write.")
        

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# MOVE FILES GENERIC FUNCTIONS
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# %% move_files
def move_files(path, dest_path, phrases_ids, verbose=0):
    short_name = 'move_files'
    
    path_annoted = "annoted"
    path_annoted_token = "annoted_token"
    path_txt = "txt"
    path_txt_src = "txt_src"
    
    fail = []
    
     # Création du répertoire s'il n'existe pas
    if (not exists(dest_path)) and dest_path is None or len(dest_path.strip()) > 0:   
        base = Path(dest_path)
        base.mkdir(exist_ok=True)
        
    for phraseID in phrases_ids:
        # catching pour le cas où les fichiers auraient déjà été déplacés
        try:
            shutil.move(join(path, path_annoted,phraseID+".csv"), join(dest_path, phraseID+"_"+path_annoted+".csv"))
        except:
            fail.append(join(path, path_annoted,phraseID+".csv"))
        try:
            shutil.move(join(path, path_annoted_token,phraseID+"_token.csv"), join(dest_path, phraseID+"_"+path_annoted+"_token.csv"))
        except:
            fail.append(join(path, path_annoted_token,phraseID+"_token.csv"))
        try:
            shutil.move(join(path, path_txt,phraseID+".txt"), join(dest_path, phraseID+"_"+path_txt+".txt"))
        except:
            fail.append(join(path, path_txt,phraseID+".txt"))
        try:
            shutil.move(join(path, path_txt_src,phraseID+".txt"), join(dest_path, phraseID+"_"+path_txt_src+".txt"))
        except:
            fail.append(join(path, path_txt_src,phraseID+".txt"))
    if verbose > 0:
        info(short_name, f'{len(fail)} files in error')
    return fail


def move_files_from_file_name_to_file_name(source_path, dest_path, begin_name=None, end_name=None, end_with=None, verbose=0):
    short_name = 'move_files_from_to'
    fail = []
    nb_moved = 0
    if source_path is not None and dest_path is not None:
        # charger tous les fichiers csv dans une df unique pour exploration avant apprentissage du modèle.
        all_files = get_dir_files(dir_path=source_path, endwith=end_with, verbose=verbose-1)
                
        has_begin = begin_name is None or len(begin_name) == 0
        has_ended = not(end_name is None or len(end_name) == 0)
        
        for file_name in all_files:
            
            if not has_begin and begin_name is not None and begin_name in file_name:
                has_begin = True
            
            if not has_ended and end_name is not None and end_name in file_name:
                end_name = True
                break
            
            if has_begin and (end_with is None or file_name.endswith(end_with)):
                src = file_name
                dest = file_name.replace(source_path, dest_path)
                # catching pour le cas où les fichiers auraient déjà été déplacés
                try:
                    shutil.move(src, dest)
                    nb_moved += 1
                except:
                    fail.append(src)
        
    else:
        if verbose>0:
            info(short_name, f'one parameter missing :')
            info(short_name, f'- source_path = {source_path}')
            info(short_name, f'- dest_path = {dest_path}')
            
    if verbose > 0:
        info(short_name, f'{nb_moved} files moved')
        warn(short_name, f'{len(fail)} files in error')
    return fail

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# LOAD FILES GENERIC FUNCTIONS
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# %% load_all_csv_files
def load_all_csv_files(path_csv, sort_on='phraseID', extension="*.csv", verbose=0):
    short_name = 'load_all_csv_files'
    # charger tous les fichiers csv dans une df unique pour exploration avant apprentissage du modèle.
    all_files = glob.glob(join(path_csv , extension))
    all_files = sorted(all_files)
    
    if verbose > 0:
        info(short_name, f'{len(all_files)} files found')
    li = []

    for filename in all_files:
        df = pd.read_csv(filename, index_col=None, header=0)
        li.append(df)

    frame = pd.concat(li, axis=0, ignore_index=True)
    
    if sort_on is not None:
        frame = frame.sort_values(by=[sort_on])
        frame = frame.reset_index()
        try:
            frame = frame.drop(columns=['index'])
        except Exception:
            pass
    try:
        frame = frame.drop(columns=['Unnamed: 0'])
    except Exception:
        pass
    if verbose > 0:
        info(short_name, f'{frame.shape} loaded')
    return frame

# %% replace_last_occurrence
def replace_last_occurrence(input_str, strToReplace, replacementStr,verbose=0):
    # Reverse the substring that need to be replaced
    strToReplaceReversed   = strToReplace[::-1]
    # Reverse the replacement substring
    replacementStrReversed = replacementStr[::-1]
    # Replace last occurrences of substring 'is' in string with 'XX'
    strValue = input_str[::-1].replace(strToReplaceReversed, replacementStrReversed, 1)[::-1]
    
    return strValue




# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# %%                                              TEST
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def _test_replace_last_occurrence(verbose=1):
    to_test = {
        "This is the last rain of Season and Jack is here.":
            ("is", "xx", "This is the last rain of Season and Jack xx here."),
            
            
        "3.7 milliards":
            ("3.7", str(3.7*1000), "3700.0 milliards"),
    }
    for input_str, (strToReplace,replacementStr,expect)  in tqdm(to_test.items(), desc=f"[TEST > replace_last_occurrence]"):
        r = replace_last_occurrence(input_str, strToReplace, replacementStr,verbose=verbose)
        assert expect == r, f"FAIL {expect} expected for {input_str} get {r}"

        
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# %%                                              MAIN
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
if __name__ == ('__main__'):
    short_name = "util_file"
    info(short_name, "---------------- TESTS ------------------ START")
    _test_replace_last_occurrence()
    info(short_name, "---------------- TESTS ------------------ END")
    

        