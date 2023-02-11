from random import randint

def format_str_switch_alea_value(input=None, alea=0, verbose=0):
    """_summary_

    Args:
        input (str or list(str), optional): _description_. Defaults to None.
        alea (int, optional): 0 to 4. Defaults to 0, no change.
        verbose (int, optional): log level. Defaults to 0.

    Returns:
        str or list(str): _description_
    """
    if alea is None or alea > 4:
        alea = randint(0,4)
        
    if input is not None and isinstance(input, str):        
        str_ = input
        if alea == 1:
            str_ = input.upper()
        elif alea == 2:
            str_ = input.title()
        elif alea == 3:
            str_ = input.lower()
        elif alea == 4:
            str_ = input.capitalize()
        return str_
        
    res = input
    if alea == 1:
        res = [str_.upper() for str_ in input]
    elif alea == 2:
        res = [str_.title() for str_ in input]
    elif alea == 3:
        res = [str_.lower() for str_ in input]
    elif alea == 4:
        res = [str_.capitalize() for str_ in input]
        
    return res
    