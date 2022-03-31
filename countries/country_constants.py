from collections import defaultdict
import numpy as np
from tqdm import tqdm
from pycountry_convert import country_alpha2_to_continent_code, country_name_to_country_alpha2, country_name_to_country_alpha3, convert_country_alpha2_to_continent_code
from geopy.geocoders import Nominatim
from time import time
import pycountry

# ---------------------------------------------------------------------------------------------
#                               TRAITEMENT DES PAYS
# ---------------------------------------------------------------------------------------------
countries_possibilities = {
                            "Albania":['albanie'],
                            "Algeria":['algerie', 'algérie'],
                            "Argentina":['argentine'],
                            "Armenia":['arménie'],
                            "Australia":['null-australia', 'australie'],
                            "Austria":['autriche'],
                            "Azerbaijan":['azerbaïdjan'],
                            "Bahamas":['the-bahamas'],
                            "Bahrain":['bahreïn'],
                            "Belarus":['biélorussie'],
                            "Belgium":['belgica', 'belgie', 'belgien', 'belgio', 'belgique'],
                            "Benin":['bénin'],
                            "Bhutan":['bhoutan'],
                            "Bolivia":['bolivie'],
                            "Bosnia And Herzegovina":['bosnia i hercegovina bosnian', 'bosnia-i-hercegovina-bosnian', 'bosnie-herzégovine', 'dominican republicvbosnia and herzegovina'],
                            "Brazil":['brésil', 'brazil,pt'],
                            "Bulgaria":['bulgarien', 'bulagria', 'bulgarie'],
                            "Cambodia":['cambodge'],
                            "Cameroon":['cameroon', 'cameroun'],
                            "Central African Republic":['centrafrique'],
                            "Chad":['tchad'],
                            "Chile":['chile', 'chili'],
                            "China":['chine'],
                            "Colombia":['colombie'],
                            "Comoros":['comores'],
                            "Congo":['congo (rdc)', 'republic-of-the-congo', 'congo (kinshasa)', 'congo (brazzaville)'],
                            "Croatia":['croatia', 'croacia', 'croatie'],
                            "Cyprus":['κύπρο', 'chypre'],
                            "Czech Republic":['česko', 'czech-republic', 'czech-repblik', 'czechy', 'czech-republi', 'république tchèque'],
                            "Democratic People's Republic of Korea":['korea'],
                            "Democratic Republic Of The Congo":['democratic-democratic-republic-of-the-congo', 'democratic-republic-of-the-congo'],
                            "Denmark":['danemark', 'dinamarca'],
                            "Dominican Republic":['republica-dominicana-espanol', 'république dominicaine'],
                            "Ecuador":['équateur'],
                            "Egypt":['égypte'],
                            "El Salvador":['salvador'],
                            "Estonia":['estonie'],
                            "Ethiopia":['éthiopie'],
                            "European Union":['europa', 'europe'],
                            "Finland":['suomi', 'finlande'],
                            "France":['france', 'dom-tom', 'franca', 'francia', 'francja', 'frankreich', 'frankrijk', 'franța', 'paris', 'ranska'],
                            "French Polynesia":['polinesia-francesa', 'polynesie-francaise'],
                            "Gambia":['gambie'],
                            "Georgia":['géorgie'],
                            "Germany":['germany', 'alemania', 'germania', 'niemcy', 'deutschland', 'east-germany', 'allemagne'],
                            "Greece":['grèce'],
                            "Guadeloupe":['guadalupe'],
                            "Guatemala":['guatemaltecos'],
                            "Guinea":['guinee', 'guinée'],
                            "Haiti":['haïti'],
                            "Holy See":['vatican-city'],
                            "Hong Kong":['hong kong (chine)'],
                            "Hungary":['hungria', 'hungaria', 'hongrie'],
                            "Iceland":['islande'],
                            "India":['inda', 'indian-subcontinent', 'inde'],
                            "Indonesia":['indonésie'],
                            "Ireland":['irland-en-de', 'irlanda', 'irland', 'irlande'],
                            "Israel":['ישראל', 'israël'],
                            "Italy":['italia', 'italien', 'italy', 'andria', 'italie'],
                            "Ivory Coast":['cote-d-ivoire'],
                            "Jamaica":['jamaïque'],
                            "Japan":['japon'],
                            "Jordan":['الأردن', 'لأردن', 'jordanie'],
                            "Korea":['korea-한국어'],
                            "Kosovo":['kosovo', 'kosovo'],
                            "Kurdistan irakien":['iraqi-kurdistan'],
                            "Kuwait":['koweït'],
                            "Kyrgyzstan":['kirghizistan'],
                            "Latin America":['latinoamerica'],
                            "Latvia":['lettonie'],
                            "Lebanon":['liban'],
                            "Libya":['libye'],
                            "Lithuania":['lituanie'],
                            "Luxembourg":['luxemburgo'],
                            "Malaysia":['malay', 'malaisie'],
                            "Malta":['malte'],
                            "Martinique":['martinica'],
                            "Mauritania":['mauritanie'],
                            "Mauritius":['maurice'],
                            "Mexico":['mexique', 'mexixco', 'tijuana-baja-california'],
                            "Moldova":['moldavie'],
                            "Mongolia":['mongolia', 'mongolie'],
                            "Montenegro":['monténégro'],
                            "Morocco":['maroc', 'marruecos', 'marokko', 'moroccov', 'المغرب', 'morocco'],
                            "Myanmar":['myanmar (birmanie)'],
                            "Namibia":['namibie'],
                            "Nepal":['népal'],
                            "Netherlands":['niederlande', 'paises bajos', 'nederland', 'pays-bas'],
                            "New Caledonia":['nouvelle-caledonie'],
                            "New Zealand":['new-zealand-english', 'new zealand', 'nouvelle-zélande'],
                            "North Macedonia":['republic-of-macedonia', 'north macedonia'],
                            "Northern Cyprus":['chypre du nord'],
                            "Norway":['oslo', 'norvège'],
                            "Peru":['pérou'],
                            "Poland":['poland', 'polonia', 'pologne', 'polska', 'polen', 'poland-polski', 'poland-romania'],
                            "Portugal":['portugal'],
                            "Reunion":['la reunion', 'la-reunion'],
                            "Romania":['romanina', 'romaniaă', 'roumanie'],
                            "Russian Federation":['soviet-union', 'russia-русский', 'rusia', 'russie'],
                            "Saudi Arabia":['arabie saoudite'],
                            "Senegal":['sénégal'],
                            "Serbia":['serbie'],
                            "Singapore":['singapour'],
                            "Slovakia":['slowakai', 'slovaquie', 'slovénie'],
                            "Slovenia":['slowenien', 'eslovenia'],
                            "Somalia":['somalie'],
                            "South Africa":['afrique du sud'],
                            "South Korea":['corée du sud'],
                            "South Sudan":['soudan du sud'],
                            "Spain":['espa�a', 'espagne', 'spagna', 'españa', 'spain', 'espanha', 'spanien'],
                            "State of Palestine":['palestinian territories', 'palestinian territories', 'palestinian-territories', 'فلسطين'],
                            "Sudan":['soudan'],
                            "Sweden":['sverige', 'schweden', 'suecia', 'swaziland', 'suède'],
                            "Switzerland":['svizzera', 'suisse', 'szwajcaria', 'svizzera', 'schweiz', 'suiza', 'swiss'],
                            "Syria":['syrie'],
                            "Taiwan":['taïwan'],
                            "Tajikistan":['tadjikistan'],
                            "Tanzania":['tanzanie'],
                            "Thailand":['thailande', 'thaïlande'],
                            "Trinidad And Tobago":['trinidad-tobagot-english', 'trinité-et-tobago', 'trinidad & tobago'],
                            "Tunisia":['tunisia', 'tunisie', 'تونس'],
                            "Turkey":['birleşik-krallık-en-turkey', 'turquie', 'turkiye'],
                            "Turkmenistan":['turkménistan'],
                            "US Minor Outlying Islands":['u-s-minor-outlying-islands'],
                            "Uganda":['ouganda'],
                            "United Arab Emirates":['émirats arabes unis'],
                            "United Kingdom":['royaume-uni', 'reino-unido', 'inglaterra', 'england', 'wales'],
                            "United Republic of Tanzania":['tanzania'],
                            "United States":['estados unidos', 'etats-unis', 'etats-unis', 'vereinigte-staaten-von-amerika', 'porto rico (états-unis)', 'états-unis', 'estados-unidos', 'vereinigte staaten von amerika'],
                            "Uzbekistan":['ouzbékistan'],
                            "Vietnam":['viêt nam'],
                            "Yemen":['yémen'],
                            "Yugoslavia":['yugoslavia'],
                            "Zambia":['zambie']
                            }
_correspondance_with_official_name_lower_keys = defaultdict(list)
                            

# ---------------------------------------------------------------------------------------------
# Correction des pays qui sont en erreur dans la librairie
lat_long = {('IT', 'EU'): (41.871940, 12.567380),  # Italie
            ('JP', 'AS'): (34.886306, 134.379711),  # ('JP', 'AS') Japan nan
            ('CZ', 'EU'): (49.817492, 15.472962),  # ('CZ', 'EU') Czech Republic nan
            ('VE', 'SA'): (6.423750, -66.589730),  # ('VE', 'SA') Venezuela nan
            ('NP', 'AS'): (28.394857, 84.124008),  # ('NP', 'AS') Nepal nan
            ('SY', 'AS'): (34.802075, 38.996815),  # ('SY', 'AS') Syria nan
            ('IE', 'EU'): (53.412910, -8.243890),  # ('IE', 'EU') Ireland nan
            ('UY', 'SA'): (-32.522779, -55.765835),  # ('UY', 'SA') Uruguay nan
            ('KY', 'NA'): (19.313300, -81.254600),  # ('KY', 'NA') Cayman Islands nan
            ('JO', 'AS'): (30.585164, 36.238414),  # ('JO', 'AS') Jordan nan
            ('ZW', 'AF'): (-19.015438, 29.154857),  # ('ZW', 'AF') Zimbabwe nan
            ('FI', 'EU'): (61.924110, 25.748151),  # ('FI', 'EU') Finland nan
            ('MW', 'AF'): (-13.254308, 34.301525),  # ('MW', 'AF') Malawi nan
            ('PY', 'SA'): (-23.442503, -58.443832),  # ('PY', 'SA') Paraguay nan
            ('UA', 'EU'): (44.874119, 33.151245),  # ('UA', 'EU') Ukraine nan
            ('EC', 'SA'): (-1.831239, -78.183406),  # ('EC', 'SA') Ecuador nan
            ('AM', 'AS'): (40.069099, 45.038189),  # ('AM', 'AS') Armenia nan
            ('LK', 'AS'): (7.873592, 80.773137),  # ('LK', 'AS') Sri Lanka nan
            ('PR', 'NA'): (18.220833, -66.590149),  # Puerto Rico
            ('GB', 'EU'): (52.3555177, -1.1743197),  # United Kingdom
            ('UG', 'AF'): (1.373333, 32.290275),  # ('UG', 'AF') Uganda nan
            ('GF', 'SA'): (3.921724136000023, -53.23312207499998),  # ('GF', 'SA') French Guiana nan
            ('PF', 'OC'): (-17.67739793399994, -149.40097329699998),  # French Polynesia nan
            ('GD', 'NA'): (12.151965053000026, -61.659644958999934),  # Grenada
            ('GY', 'SA'): (4.796422680000035, -58.97538657499996),  # Guyana
            ('JE', 'EU'): (49.21402591200007, -2.1327190749999545),  # Jersey
            ('MD', 'EU'): (47.20102827100004, 28.46370618900005),  # Moldova
            ('PW', 'OC'): (7.421479662000024, 134.511600068),  # Palau
            ('MF', 'NA'): (18.080477531000042, -63.06021562199999),  # Saint Martin
            ('SR', 'SA'): (3.9317774090000626, -56.01360780899995),  # Suriname
            ('VU', 'OC'): (-15.241355872999975, 166.8727570740001),  # Vanuatu
            ('AG', 'NA'): (17.0869391, -61.783491),
            ('BA', 'EU'): (43.9165389, 17.6721508),
            ('NL', 'NA'): (12.201890, -68.262383),
            ('CI', 'AF'): (7.5455112, -5.547545),  # Ivory Coast
            ('CW', 'NA'): (12.2135221, -68.9495816),
            ('CD', 'AF'): (-4.0335162, 21.7500603),
            ('FR', 'EU'): (46.71109, 1.7191036),
            ('IM', 'EU'): (54.2312716, -4.569504),
            ('MK', 'EU'): (41.6137143, 1.743258),  # North Macedonia
            ('RE', 'AF'): (-21.1306889, 55.5264794),
            ('TT', 'NA'): (10.4437128, -61.4191414),
            ('VI', 'NA'): (18.3434415, -64.8671634),
            ('BB', 'NA'): (13.1901325, -59.5355639),  # Barbados
            ('AW', 'NA'): (12.517572, -69.9649462),  # Aruba ('AW', 'NA') (nan, nan)
            ('GG', 'EU'): (49.4630653, -2.5881123),  # Guernesey
            ('PS', 'AS'): (31.947351, 35.227163),  # State of Palestine
            ("KN", "NA"): (17.2561791, -62.7019638),  # Saint Kitts and Nevis
            ("PM", "NA"): (46.9466881, -56.2622848),  # "Saint Pierre and Miquelon"
            ('VC', 'NA'): (13.252818, -61.197096),  # Saint Vincent And The Grenadines
            ('SX', 'NA'): (18.0347188, -63.0681114),  # Sint Maarten
            ('TT', 'SA'): (10.536421,-61.311951)      # 'Trinidad And Tobago'
            }

countries_dict = {}

# ---------------------------------------------------------------------------------------------
#                          Functions pour les données géographiques
# ---------------------------------------------------------------------------------------------
_dic_alpha2 = {
                'antigua and barbuda': ('AG', 'NA'), 
                'bosnia and herzegovina': ('BA', 'EU'), 
                'caribbean netherlands': ('NL', 'NA'), 
                'cote d ivoire': ('CI', 'AF'), 
                'curacao': ('CW', 'NA'), 
                'Czech Republic' : ('CZ', 'EU'), # Czech Republic nan
                'czech republic' : ('CZ', 'EU'), # Czech Republic nan
                'democratic republic of the congo': ('CD', 'AF'), 
                'dominican republic': ('DO', 'SA'),
                'european union': ('FR', 'EU'), 
                'isle of man': ('IM', 'EU'), 
                'north macedonia': ('MK', 'EU'), 
                'republic of macedonia': ('MK', 'EU'), 
                'reunion': ('RE', 'AF'), 
                'saint kitts and nevis': ('KN', 'NA'), 
                'saint pierre and miquelon': ('PM', 'NA'), 
                'sint maarten': ('SX', 'NA'), 
                'state of palestine': ('PS', 'AS'), 
                'trinidad and tobago': ('TT', 'SA'), 
                'united kingdom': ('GB', 'EU'), 
                'virgin islands of the united states': ('VI', 'NA'), 
                }

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#                                              FUNCTION
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def get_dic_alpha2(country_name, verbose=0):
    if country_name is not None:
        return _dic_alpha2.get(country_name.lower(), None)
    return None

# 1. Conversion to Alpha 2 codes and Continents
def get_continent(country_name_param, include_format=False, verbose=False):
    """
    :param country_name_param (str): nom du pays recherché en anglais, attention, doit avoir des majuscules aux premières lettre de chaque mot, mais pas sur les petits mot
    :param include_format (boolean): True pour lancer le formatage (majuscules aux 1ère lettre et pas pour les petits mots du type : of,and, ...)
    :param verbose (boolean): True pour mode debug
    :return: (str, str) :(country a2 code, continent code)
    """
    country_name = country_name_param
    cn_a2_code = np.nan
    cn_a3_code = np.nan
    cn_continent = np.nan
    if country_name is not None and len(country_name) > 0:
        if include_format:
            country_name = country_name.title()
            country_name = country_name.replace(" Of ", " of ")
            country_name = country_name.replace(" The ", " the ")
            country_name = country_name.replace(" And ", " and ")
        try:
            cn_a2_code = country_name_to_country_alpha2(country_name)
        except:
            cn_a2_code = _dic_alpha2.get(country_name.lower(), np.nan)
            if cn_a2_code != np.nan:
                try:
                    cn_a2_code = cn_a2_code[0]
                except:
                    if verbose:
                        print("cn_a2_code ",country_name, "=> FAIL : ", cn_a2_code)
                    cn_a2_code = np.nan
        try:
            cn_a3_code = country_name_to_country_alpha3(country_name)
        except:
            if verbose:
                print("cn_a3_code ",country_name, "=> FAIL alpha3 : ", cn_a3_code)
            cn_a3_code = np.nan

        try:
            cn_continent = country_alpha2_to_continent_code(cn_a2_code)
        except:
            cn_continent = _dic_alpha2.get(country_name.lower(), np.nan)
            if cn_continent != np.nan:
                try:
                    cn_continent = cn_continent[1]
                except:
                    pass
    return cn_a2_code, cn_continent, cn_a3_code


def get_geolocation(country, geolocator=None, verbose=False):
    """
    :param country :(str, str)(country a2 code, continent code)
    :param geolocator: Nominatim
    :param verbose (boolean): True pour mode debug
    :return:(float, float):(latitude, longitude) or (nan, nan)
    """
    if geolocator is None:
        geolocator = Nominatim(user_agent="catuserbot")
    try:
        if country in lat_long.keys():
            return lat_long.get(country, np.nan)
        else:
            # Geolocate the center of the country
            loc = geolocator.geocode(country)
            # And return latitude and longitude
            return loc.latitude, loc.longitude
    except:
        # Return missing value
        return np.nan, np.nan
   

def get_country_alpha3(country_name, alpha2, verbose=False):
    alpha3 = None

    if alpha2 is not None:
        try:
            pcountry = pycountry.countries.get(alpha_2=alpha2)
            if pcountry is not None:
                alpha3 = pcountry.alpha_3
        except:
            pass
    # On essaie avec le nom
    if country_name is not None and country_name is None:
        alpha3 = __get_country_alpha3_with_name(country_name, verbose)

    if alpha3 is None and verbose:
        print("cn_a3_code FAIL => ", country_name, alpha2)
    return alpha3


def get_country_official_name(country_name, alpha2, alpha3, verbose=False):
    official_name = None
    if alpha2 is not None:
        official_name = __get_country_official_name_with_alpha2(alpha2, verbose)
    
    if official_name is None and country_name is not None:
        official_name = __get_country_official_name_with_name(country_name, verbose)

    if official_name is None and alpha3 is not None:
        official_name = __get_country_official_name_with_alpha3(alpha3, verbose)

    return official_name
# ---------------------------------------------------------------------------------------------
#                               Préparation des données
# ---------------------------------------------------------------------------------------------
def get_country_data(country_name_param, geolocator=None, include_format=False, verbose=False, alpha3_param=None):
    """
    Récupère les données du pays
    :param country_name_param (str): nom du pays recherché en anglais, attention, doit avoir des majuscules aux premières lettre de chaque mot, mais pas sur les petits mot
    :param geolocator: Nominatim
    :param include_format (boolean): True pour lancer le formatage (majuscules aux 1ère lettre et pas pour les petits mots du type : of,and, ...)
    :param verbose (boolean): True pour mode debug
    :return: (str, str, float, float) : alpha2, continent_code, latitude, longitude
    """
    t0 = time()
    country_name = country_name_param
    alpha3 = alpha3_param
    country_id = np.nan
    alpha2 = None
    continent_code = None
    latitude = np.nan
    longitude = np.nan
    official_name = None

    if alpha3_param is not None:
        try:
            pcountry = pycountry.countries.get(alpha_3=alpha3_param)
            if pcountry is not None:
                try:
                    alpha2 = pcountry.alpha_2
                    if alpha2 is not None:
                        continent_code = convert_country_alpha2_to_continent_code.country_alpha2_to_continent_code(alpha2)
                except:
                    pass
                try:
                    official_name = pcountry.official_name
                except:
                    official_name = pcountry.name
                try:
                    country_id = pcountry.numeric
                except:
                    pass
        except:
            pass

    if alpha2 is None or continent_code is None:
        alpha2b, continent_codeb, alpha3b = get_continent(country_name, include_format, verbose)
        if alpha2 is None:
            alpha2 = alpha2b
        if continent_code is None:
            continent_code = continent_codeb
        if alpha3 is None:
            alpha3 = alpha3b

    if alpha3 is None and (alpha2 is not None or country_name is not None) :
        alpha3 = get_country_alpha3(country_name, alpha2)

    if official_name is None:
        official_name = get_country_official_name(country_name, alpha2, alpha3)

    # Récupération des coordonnées
    if alpha2 != np.nan and continent_code != np.nan:
        if geolocator is None:
            geolocator = Nominatim(user_agent="catuserbot")

        geoloc = get_geolocation((alpha2, continent_code), geolocator)
        if geoloc != np.nan:
            try:
                latitude = geoloc[0]
                longitude = geoloc[1]
            except TypeError:
                print("TypeError for :", (alpha2, continent_code), country_name, geoloc)
        else:
            print("Country not found geoloc :", (alpha2, continent_code), country_name)
    else:
        if country_name == 'Holy See':
            latitude = 41.902916
            longitude = 12.453389
        else:
            print("Country not known :", country_name)

    t1 = time() - t0
    if verbose:
        print("get_country_data", country_name,
              " in {0:.3f} secondes............................................... END".format(t1))
    return alpha2, continent_code, latitude, longitude, alpha3, official_name, country_id

# ---------------------------------------------------------------------------------------------
#                               MAIN
# ---------------------------------------------------------------------------------------------

def __get_country_alpha3_with_name(country_name, verbose=False):
    alpha3 = None
    if country_name is not None:
        # On essaie avec le nom
        try:
            pcountry = pycountry.countries.get(name=country_name)
            if pcountry is not None:
                alpha3 = pcountry.alpha_3
        except:
            pass
    if alpha3 is None and verbose:
        print("cn_a3_code FAIL => ", country_name)
    return alpha3

def __get_country_official_name_with_name(country_name, verbose=False, correct=False, lower=False):
    official_name = None
    if country_name is not None:
        # On essaie avec le nom
        try:
            pcountry = pycountry.countries.get(name=country_name)
            if pcountry is not None:
                try:
                    official_name = pcountry.official_name
                except:
                    official_name = pcountry.name
        except:
            pass

        # On recherche avec le nom corrigé
        if official_name is None and not correct:
            temp = _correct_official_name(country_name, verbose=verbose)
            if temp != country_name:
                official_name = __get_country_official_name_with_name(temp, verbose=verbose, correct=True)
                if official_name is None:
                    official_name = temp
        
        # On recherche avec le nom en minuscule
        if official_name is None and not lower:
            official_name = __get_country_official_name_with_name(country_name.lower(), verbose=verbose, correct=True, lower=True)
            
    if official_name is None and verbose:
        print("official_name FAIL => ", country_name)
    return official_name


def _correct_official_name(country_name, verbose=0):   
    if len(_correspondance_with_official_name_lower_keys) == 0:
        for keys, value in countries_possibilities.items():
            for val in value:
                _correspondance_with_official_name_lower_keys[val] = keys
    
    if country_name is not None:
        return _correspondance_with_official_name_lower_keys.get(country_name.lower(),country_name)
    return None

def __get_country_official_name_with_alpha3(alpha3, verbose=False):
    official_name = None
    if alpha3 is not None:
        # On essaie avec le nom
        try:
            pcountry = pycountry.countries.get(alpha_3=alpha3)
            if pcountry is not None:
                try:
                    official_name = pcountry.official_name
                except:
                    official_name = pcountry.name
        except:
            pass
    if official_name is None and verbose:
        print("official_name FAIL => ", alpha3)
    return official_name


def __get_country_official_name_with_alpha2(alpha2, verbose=False):
    official_name = None
    if alpha2 is not None:
        try:
            pcountry = pycountry.countries.get(alpha_2=alpha2)
            if pcountry is not None:
                try:
                    official_name = pcountry.official_name
                except:
                    official_name = pcountry.name
        except:
            pass
    if official_name is None and verbose:
        print("official_name FAIL => ", alpha2)

    return official_name

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#                                              TESTS
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

if __name__ == "__main__":

    verbose = 0

    test_list = ["Finland", "Finlande", "France", "Finland","Denmark","Norway","Iceland","Netherlands","Switzerland","Sweden","New Zealand","Canada","Austria","Australia","Costa Rica","Israel","Luxembourg","United Kingdom","Ireland",
"Germany","Belgium","United States","Czech Republic","United Arab Emirates","Malta","Mexico","France","Taiwan","Chile","Guatemala","Saudi Arabia","Qatar","Spain","Panama","Brazil",
"Uruguay","Singapore","El Salvador","Italy","Bahrain","Slovakia","Trinidad & Tobago","Poland","Uzbekistan","Lithuania","Colombia","Slovenia","Nicaragua","Kosovo","Argentina","Romania",
"Cyprus","Ecuador","Kuwait","Thailand","Latvia","South Korea","Estonia","Jamaica","Mauritius","Japan","Honduras","Kazakhstan","Bolivia","Hungary","Paraguay","Northern Cyprus",
"Peru","Portugal","Pakistan","Russia","Philippines","Serbia","Moldova","Libya","Montenegro","Tajikistan","Croatia","Hong Kong","Dominican RepublicvBosnia and Herzegovina",
"Turkey","Malaysia","Belarus","Greece","Mongolia","North Macedonia","Nigeria","Kyrgyzstan","Turkmenistan","Algeria","Moroccov","Azerbaijan","Lebanon","Indonesia","China",
"Vietnam","Bhutan","Cameroon","Bulgaria","Ghana","Ivory Coast","Nepal","Jordan","Benin","Congo (Brazzaville)","Gabon","Laos","South Africa","Albania","Venezuela","Cambodia",
"Palestinian Territories","Senegal","Somalia","Namibia","Niger","Burkina Faso","Armenia","Iran","Guinea","Georgia","Gambia","Kenya","Mauritania","Mozambique","Tunisia",
"Bangladesh","Iraq","Congo (Kinshasa)","Mali","Sierra Leone","Sri Lanka","Myanmar","Chad","Ukraine","Ethiopia","Swaziland","Uganda","Egypt","Zambia","Togo","India","Liberia",
"Comoros","Madagascar","Lesotho","Burundi","Zimbabwe","Haiti","Botswana","Syria","Malawi","Yemen","Rwanda","Tanzania","Afghanistan","Central African Republic","South Sudan"]

    for country_feature in tqdm(test_list):
        off = __get_country_official_name_with_name(country_feature, verbose=verbose)
        if off is None:
            alpha2, continent_code, latitude, longitude, alpha3, off, country_id = get_country_data(country_feature, verbose=verbose)
            if off is None:
                print(f"\nASK : {country_feature} => GET : {off} => : {alpha2, continent_code, latitude, longitude, alpha3, off, country_id}")





