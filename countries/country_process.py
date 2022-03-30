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

    if features is not None:
        for country_feature in tqdm(features):
            alpha3 = country_feature.get("id", None)
            if alpha3 is not None:
                props = country_feature.get("properties")
                if props is not None:
                    country_name = props.get("name", None)
                    if country_name is not None:
                        country_code, continent_code, latitude, longitude, a3, official_name, country_id = get_country_data(country_name, alpha3_param=alpha3, verbose=verbose)
                        res["id"].append(country_id)
                        if alpha3 is None:
                            res["alpha3"].append(a3)
                        else:
                            res["alpha3"].append(alpha3)
                        res["alpha2"].append(country_code)
                        res["country"].append(country_name)
                        res["country_official"].append(official_name)
                        res["continent_code"].append(continent_code)
                        res["latitude"].append(latitude)
                        res["longitude"].append(longitude)

    df = pd.DataFrame.from_dict(res)
    return df


def save_df_in_file(df_to_save, file_path):
    now = datetime.now() # current date and time
    date_time = now.strftime("%Y-%m-%d-%H_%M_%S")
    df_to_save.to_csv(file_path+"_" + date_time + '.csv', sep=',', index=False)

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