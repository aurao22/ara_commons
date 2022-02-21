
from collections import defaultdict
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder


def get_numeric_columns_names(df, verbose=False):
    """Retourne les noms des colonnes numériques
    Args:
        df (DataFrame): Données
        verbose (bool, optional): Mode debug. Defaults to False.

    Returns:
        List(String): liste des noms de colonne
    """
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    newdf = df.select_dtypes(include=numerics)
    return list(newdf.columns)


def remove_empty_numeric_columns(df, verbose=True, inplace=True, min_nunique_val=3):
    """Supprime les colonnes qui ont un pourcentage de NA supérieur au max_na

    Args:
        df (DataFrame): Données à nettoyer
        verbose (bool, optional): Mode debug. Defaults to False.
        inplace (bool, optional): Pour mettre à jour la dataframe reçue directement. Defaults to True.

    Returns:
        [DataFrame]: DataFrame avec les données mises à jour
    """
    if not inplace:
        df = df.copy()
        
    to_remove = set()
    dict_col = defaultdict(list)

    cols = get_numeric_columns_names(df, verbose)
    # Constitution de la list des colonnes à supprimer
    
    for col in cols:
        
        nb = df[col].nunique()
        dict_col[nb].append(col)

        if (df[col].max() == 0 and df[col].mean() == 0)  or (df[col].max() == -1 and df[col].mean() == -1):
            to_remove.add(col)
        elif (df[col].max() == df[col].mean())  and (df[col].max() == df[col].min()):
            to_remove.add(col)
        elif min_nunique_val is not None and nb < min_nunique_val:
            to_remove.add(col)
        else:
            df.loc[df[col] == -1, col] = np.nan

    
    shape_start = df.shape
    # Suppression des colonnes
    df.drop(to_remove, axis=1, inplace=True)
    shape_end = df.shape

    if verbose:
        keys = list(dict_col.keys())
        keys = sorted(keys)
        for k in keys:
            print(k, "=>", dict_col[k])
        print("Removes columns :",to_remove)
    print("remove_empty_columns, shape start: ",shape_start,"=>",shape_end," ............................................... END")        
    return df  


def remove_na_columns(df, max_na=73, verbose=True, inplace=True):
    """Supprime les colonnes qui ont un pourcentage de NA supérieur au max_na

    Args:
        df (DataFrame): Données à nettoyer
        max_na (int) : pourcentage de NA maximum accepté (qui sera conserver)
        verbose (bool, optional): Mode debug. Defaults to False.
        inplace (bool, optional): Pour mettre à jour la dataframe reçue directement. Defaults to True.

    Returns:
        [DataFrame]: DataFrame avec les données mises à jour
    """
    if not inplace:
        df = df.copy()
        
    to_remove = set()
    dict_col = {}

    # Constitution de la list des colonnes à supprimer
    for col in df.columns:
        pourcent = int((df[col].isna().sum()*100)/df.shape[0])
        list = dict_col.get(pourcent, [])
        list.append(col)
        dict_col[pourcent] = list
        if pourcent > max_na:
            to_remove.add(col)
    
    if verbose:
        for k in range(101):
            if len(dict_col.get(k, [])) > 0:
                print(k, "=>", len(dict_col.get(k, [])), dict_col.get(k, []))
    
    shape_start = df.shape
    # Suppression des colonnes
    df.drop(to_remove, axis=1, inplace=True)
    shape_end = df.shape
    
    print("remove_na_columns, shape start: ",shape_start,"=>",shape_end,"s............................................... END")        
    return df  


def process_one_hot(df, col="description", verbose=0):
    encoder = OneHotEncoder(sparse=False)
    transformed = encoder.fit_transform(df[[col]])
    if verbose:
        print(transformed)
    #Create a Pandas DataFrame of the hot encoded column
    ohe_df = pd.DataFrame(transformed, columns=encoder.get_feature_names_out())
    if verbose:
        print("ohe_df:", ohe_df.shape, "data:", df.shape)

    #concat with original data
    df_completed = df.copy()
    df_completed = pd.concat([df_completed, ohe_df], axis=1)
    if verbose:
        print("ohe_df:", ohe_df.shape, "data:", df.shape, "data_encode:", df_completed.shape)
    return df_completed
