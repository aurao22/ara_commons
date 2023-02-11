# %% import
from datetime import datetime
from termcolor import colored

COLORS = {
    "default" : (6,42,30),
    "green" : (6,42,30),
    "blue" : (13,36,81),
    "red" : (74,40,34),
    "yellow" : (87,82,9),
}

COLOR_MOD = {
    'DEBUG' : 'blue',
    'WARN'  : 'yellow',
    'ERROR' : 'red',
}

def display(function_name, text, now=None, duration=None, key='INFO', color=None, log_end=""):
    log = f"[{function_name:<20}]\t{key.upper():<5} {text}"
    
    log = f"{log} {log_end:>25}"
    
    if now is not None and duration is not None:
        now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        end_ = f"\t\t END {now} ---> in {duration}"
        log = log + f"{end_:>60}"
    elif now is not None:
        now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        end_ = f"\t\t START {now}"
        log = log + f"{end_:>38}" 
            
    if color is not None:
        log = colored(log, color)
        
    print(log)
    return log

# %% colored
def info(function_name, text, now=None, duration=None, log_end=""):
    key='INFO'
    return display(function_name=function_name, text=text, now=now, duration=duration, key=key, color= COLOR_MOD.get(key, None), log_end=log_end)

def debug(function_name, text, now=None, duration=None, log_end=""):
    key='DEBUG'
    return display(function_name=function_name, text=text, now=now, duration=duration, key=key, color= COLOR_MOD.get(key, None), log_end=log_end)

def warn(function_name, text, now=None, duration=None, log_end=""):
    key='WARN'
    return display(function_name=function_name, text=text, now=now, duration=duration, key=key, color= COLOR_MOD.get(key, None), log_end=log_end)
    
def error(function_name, text, now=None, duration=None, log_end=""):
    key='ERROR'
    return display(function_name=function_name, text=text, now=now, duration=duration, key=key, color= COLOR_MOD.get(key, None), log_end=log_end)

def surligne_text(text, color="green"):
    r = COLORS.get(color,COLORS["default"])[0] 
    g = COLORS.get(color,COLORS["default"])[1] 
    b = COLORS.get(color,COLORS["default"])[2] 
    return f'\x1b[{r};{b};{g}m{text}\x1b[0m'


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# %%                                              MAIN
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

if __name__ == ('__main__'):
    from datetime import datetime
    start_time = datetime.now() # current date and time
    short_name = "util_print"
    verbose = 2
    
    info(short_name, "---------------- MAIN ------------------", now=True)
    
    info(short_name, "Mon message info")
    debug(short_name, "Mon message debug")
    warn(short_name, "Mon message warn")
    error(short_name, "Mon message error")
    info(short_name, surligne_text("mon message surligné"))
    info(short_name, surligne_text("mon message surligné", "blue"))
    debug(short_name, surligne_text("mon message surligné", "red"))
    date_time = datetime.now() - start_time
    info(short_name, "---------------- MAIN ------------------", now=True, duration=str(date_time), log_end="FINISH")
