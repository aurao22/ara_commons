import pandas as pd
from tqdm import tqdm
import json
from collections import defaultdict
from datetime import datetime
from country_constants import *


def get_country_df(verbose=0, json_file_path = r"C:\Users\User\WORK\workspace-ia\PROJETS\projet_bonheur_bed\dataset\world-countries.json"):

    f = open(json_file_path, "r")
    json_data = f.read()

    info = json.loads(json_data)
    print(type(info))

    features = info.get('features', None)
    res = defaultdict(list)

    ever_proceed = ["Northern Cyprus".lower(), 'Somaliland'.lower(), 'Argentine'.lower(), 'European Union'.lower(), 
                    'The Republic of Cyprus'.lower(), 'Eastern Uruguay'.lower(), 'Federative Brazil'.lower(), 'Gabonese'.lower(),
                    'Hellenic'.lower()
                    ]
    
    if features is not None:
        for country_feature in tqdm(features):
            alpha3 = country_feature.get("id", None)
            if alpha3 is not None:
                props = country_feature.get("properties")
                if props is not None:
                    country_name = props.get("name", None)
                    if country_name is not None and country_name.lower() not in ever_proceed:
                        country_code, continent_code, latitude, longitude, a3, official_name, country_id = get_country_data(country_name, alpha3_param=alpha3, verbose=verbose)
                        res["id"].append(country_id)
                        try:
                            if alpha3 is None or len(str(alpha3)) == 0 or '-99' in str(alpha3) or np.isnan(alpha3):
                                res["alpha3"].append(a3)
                            else:
                                res["alpha3"].append(alpha3)
                        except Exception as error:
                            if verbose:
                                print(f"alpha3 = {alpha3}=>{error}")
                            res["alpha3"].append(alpha3)
                        
                        res["alpha2"].append(country_code)
                        res["country"].append(country_name)
                        res["country_official"].append(official_name)
                        res["continent_code"].append(continent_code)
                        res["latitude"].append(latitude)
                        res["longitude"].append(longitude)
                        ever_proceed.append(official_name.lower())
                        ever_proceed.append(country_name.lower())
    if verbose:
        print(f"{len(ever_proceed)} pays ajouté à partir fichier json")

    # Nettoyage de la mémoire
    alpha3 = None
    props = None
    json_file_path = None
    json_data = None
    info = None
    country_feature = None

    # traitement des pays qui ne seraient pas dans le fichier json
    nb_c = 0
    for country_name in tqdm(countries_possibilities.keys()):
        if country_name.lower() not in ever_proceed:
            country_code, continent_code, latitude, longitude, a3, official_name, country_id = get_country_data(country_name, verbose=verbose)
            if official_name is None:
                official_name = country_name

            if official_name.lower() not in ever_proceed:
                res["id"].append(country_id)
                res["alpha3"].append(a3)
                res["alpha2"].append(country_code)
                res["country"].append(country_name)
                res["country_official"].append(official_name)
                res["continent_code"].append(continent_code)
                res["latitude"].append(latitude)
                res["longitude"].append(longitude)
                ever_proceed.append(official_name)
                nb_c += 1
    if verbose:
        print(f"{nb_c} pays ajouté en plus du fichier json")

    df = pd.DataFrame.from_dict(res)
    df = df.sort_values(by="country_official")
    return df

import csv

def save_df_in_file(df_to_save, file_path):
    now = datetime.now() # current date and time
    date_time = now.strftime("%Y-%m-%d-%H_%M_%S")
    df_to_save.to_csv(file_path+"_" + date_time + '.csv', quoting=csv.QUOTE_NONNUMERIC, sep=',', index=False)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#                                              TESTS
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
from os import getcwd

if __name__ == "__main__":
    verbose = 0
    execution_path = getcwd() + "\\"
    if "PERSO" not in execution_path:
        execution_path = execution_path + "PERSO\\ara_commons\\countries\\"
    df = get_country_df(verbose=verbose, json_file_path = execution_path+"world-countries.json")
    save_df_in_file(df, execution_path+r"data_set_countries")
    print(df)