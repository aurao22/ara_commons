# -*- coding: utf-8 -*- 
from tqdm import tqdm
import argparse
import configparser
from collections import defaultdict

from os import getcwd
from os.path import join, exists
import sys

from util_print import *

class MyConfiguration(object):

    def __init__(self, config_file, verbose=0):
        self.config_file = config_file
        self.verbose = verbose

        if config_file is None or len(config_file)==0:
            log = "config_file missing"
            if verbose>0:
                error(short_name, log)
            raise AttributeError(f"[{short_name}] {log}")

        if not exists(config_file):
            log = f"The configuration file {config_file} doesn't exist. Current execution path {getcwd()}"
            if verbose>0:
                error(short_name, log)
            raise AttributeError(f"{short_name} : {log}")
        
        self.parser = configparser.ConfigParser() 
        # Ouverture du fichier de configuration en mode lecture et écriture
        self.parser.read(config_file) 


    def sections(self):
        """
        Returns:
            list: section list (ex: ['settings', 'Safe_Mode', 'File_Uploads'])
        """
        short_name = "sections"
        sec = self.parser.sections() 
        if self.verbose>1:
            debug(short_name, sec)
        return sec

    def sections_and_attributes(self):
        """
        Returns:
            dict(str, list): section list (ex: settings : ['host', 'username', 'password'])
        """
        short_name = "sections_and_attributes"
        sections_attributes = {}
        # Parcourt de toutes les sections
        for sec in self.parser.sections():
            # parcourt des options des différentes sections
            sections_attributes[sec] = self.parser.options(sec)
        if self.verbose>1:
            debug(short_name, f"{sections_attributes}")
        return sections_attributes

    def read_parameters(self):
        short_name = "read_parameters"
        configuration = defaultdict(dict)
        # Parcourt des sections
        for sec in self.parser.sections():    
            if self.verbose>1: debug(short_name, f"{sec} : ") 
            # parcourt des paramètres et valeurs   
            for name, value in self.parser.items(sec):        
                if self.verbose>1: debug(short_name, f"\t- {name} = {value} ") 
                configuration[sec][name] = value
        return configuration

    def value(self, section, parameter):
        return self.parser.get(section, parameter)

    def update_parameters(self, parameters, config_file):
        """ mettre à jours la valeur du paramètre

        Args:
            parameters (dict(str, dict(str, str))): key = section, Value = Parameter name, Parameter value
            config_file (str, optional): _description_. Defaults to 'configuration.ini'.
        """
        short_name = "update_parameters"
        if parameters is None or len(parameters)==0:
            log = "No input parameter"
            raise AttributeError(f"[{short_name}] {log}")
        
        for section, (param, value) in parameters.items():
            self.parser.set(section, param,value) 

        # Ouverture du fichier de configuration en mode lecture et écriture
        file = open(config_file,'r+')
        self.parser.write(file) 
        file.close()



def write_parameters(parameters, config_file, verbose=0):
    """ mettre à jours la valeur du paramètre

    Args:
        parameters (dict(str, dict(str, str))): key = section, Value = Parameter name, Parameter value
        config_file (str, optional): _description_. Defaults to 'configuration.ini'.
        verbose (int, optional): _description_. Defaults to 0.
    """
    short_name = "write_parameters"
    if parameters is None or len(parameters)==0:
        log = "No input parameter"
        raise AttributeError(f"[{short_name}] {log}")
    
    parser = configparser.ConfigParser()
    parser.read_dict(parameters)
    
    # Ouverture du fichier de configuration en mode lecture et écriture
    file = open(config_file,'r+')
    parser.write(file) 
    file.close()
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# %%                                              ARGUMENTS
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
parser = argparse.ArgumentParser(description='The configuration utilities', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-p', '--path',      default= '',                 help='Configuration path')
parser.add_argument('-f', '--file',      default='configuration.ini',             help='Configuration file name (.ini expected)')
parser.add_argument('-v', '--verbosity', default='0',   type=int, choices=[0, 1, 2, 3], help='Verbosity level')
parser.add_argument('-t', '--test',      default='True',                               help='Run tests')

if __name__ == "__main__":
    args = parser.parse_args(sys.argv[1:])
    args.test = True # TODO à supprimer lorsqu'il ne s'agit plus de tests
    
    verbose                    = args.verbosity      if not args.test       else 2
    path = args.path if args.path is not None and len(args.path)>0 else getcwd()

    if args.test:
        if "PERSO" not in path:
            path = join(path, "PERSO")
        if "ara_commons" not in path:
            path = join(path, "ara_commons")

        config_file=join(path, args.file)
        ma_conf = MyConfiguration(config_file=config_file, verbose=verbose)
        assert ma_conf is not None

        secs = ma_conf.sections()
        assert secs is not None and len(secs)>0

        secs = ma_conf.sections_and_attributes()        
        assert secs is not None and len(secs)>0
    