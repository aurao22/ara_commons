import re
from tqdm import tqdm
import spacy

NLP = spacy.load(name="fr_core_news_md")

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# TOKENIZATION functions
def get_regex_tokens(verbose=0):
    pattern = re.compile(r"\b\w+\b")
    return pattern

def ara_tokenize(text, verbose=0):
    res = None
    if isinstance(text, str):
        res = []
        doc = NLP(text)
        for token in doc:
            if re.search(r'([0-1]+[A-z]+|[A-z]+[0-1]+)'):
                
                pass
            else:
                res.append(token)
    elif isinstance(text, list):
        res = [ara_tokenize(sentence, verbose=verbose)  for sentence in text]
        res = list(filter(None, res))
    return res

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#                                              TESTS
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def _test_tokenise(verbose=1):
    to_test = {
        "Ma phrase de test écrite à 11h49 à 10 km/h sur 5 m² pour voir la tokenisation qui sera réalisée, même si certains ont le COVID19 ou plutôt le COVID-19." : [],
    }

    for text, expected in tqdm(to_test.items(), desc="ara_tokenize"):
        res = ara_tokenize(text=text, verbose=verbose)
        assert res == expected


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#                                              MAIN
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
if __name__ == "__main__":
    _test_tokenise()