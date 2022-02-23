from tabnanny import verbose
import pandas as pd
import numpy as np
import unicodedata
import re
import string 
import spacy
import warnings
warnings.filterwarnings("ignore")

from bs4 import BeautifulSoup
from pathlib import Path  
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords, wordnet
from nltk import pos_tag, ne_chunk, ngrams
from nltk.stem.porter import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus.reader import WordListCorpusReader
from nltk.probability import FreqDist
from collections import defaultdict


from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#                                              FUNCTIONS
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Regex
# @see : https://openclassrooms.com/fr/courses/6532301-introduction-to-natural-language-processing/6980751-extract-information-with-regular-expression#/id/r-7072440
def get_regex_emails(verbose=0):
    pattern = re.compile(r'\S*@\S*\s?')
    return pattern

def get_regex_urls(verbose=0):
    # Version moins performante :
    # pattern = r'https?://\S+|www\.\S+'
    pattern = re.compile(r'http.+?(?=\?|"|<)')
    return pattern

def get_regex_usernames(verbose=0):
    pattern = re.compile(r'@\S+')
    return pattern

def get_regex_html_tag(verbose=0):
    pattern = re.compile(r"<[^>]*>")
    return pattern

def get_regex_hashtags_tag(verbose=0):
    pattern = re.compile(r'#\S+')
    return pattern

def get_regex_non_punctuations(verbose=0):
    pattern = re.compile(r'[^\w\s]')
    return pattern

def get_regex_alphabetique_simple(verbose=0):
    pattern = re.compile(r'[^a-zA-Z]')
    return pattern

def get_regex_alphanumeric_simple(verbose=0):
    pattern = re.compile(r'[^A-Za-z0-9]')
    return pattern

def get_regex_digits(verbose=0):
    pattern = re.compile(f'\d+')
    return pattern

def get_regex_inline_latext(verbose=0):
    pattern = re.compile(r'\$[^>]*\$')
    return pattern

def get_regex_extra_spaces(verbose=0):
    pattern = re.compile(r'^\s*|\s\s*')
    return pattern

def get_regex_tokens(verbose=0):
    pattern = re.compile(r"\b\w+\b")
    return pattern


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Typing function

def is_verb(word, verbose=0):
    '''
    Maps the respective POS tag of a word to the format accepted by the lemmatizer of wordnet
    
    Args: word (str or list): Word to which the function is to be applied, string
    Returns: POS tag or list[POS tag], readable for the lemmatizer of wordnet
    '''
    return is_type(word, wordnet.VERB, verbose=verbose)

def is_type(word, testing_type, verbose=0):
    '''
    Maps the respective POS tag of a word to the format accepted by the lemmatizer of wordnet
    
    Args: word (str or list): Word to which the function is to be applied, string
    Returns: POS tag or list[POS tag], readable for the lemmatizer of wordnet
    '''
    if isinstance(word, str):   
        tag = get_wordnet_pos_func(word)
        return tag == testing_type.upper()
    return False

"""
CC coordinating conjunction
CD cardinal digit
DT determiner
EX existential there (like: “there is” … think of it like “there exists”)
FW foreign word
IN preposition/subordinating conjunction
JJ adjective ‘big’
JJR adjective, comparative ‘bigger’
JJS adjective, superlative ‘biggest’
LS list marker 1)
MD modal could, will
NN noun, singular ‘desk’
NNS noun plural ‘desks’
NNP proper noun, singular ‘Harrison’
NNPS proper noun, plural ‘Americans’
PDT predeterminer ‘all the kids’
POS possessive ending parent’s
PRP$ possessive pronoun my, his, hers
PRP personal pronoun I, he, she
RB adverb very, silently,
RBR adverb, comparative better
RBS adverb, superlative best
RP particle give up
TO, to go ‘to’ the store.
UH interjection, errrrrrrrm
VB verb, base form take
VBD verb, past tense took
VBG verb, gerund/present participle taking
VBN verb, past participle taken
VBP verb, sing. present, non-3d take
VBZ verb, 3rd person sing. present takes
WDT wh-determiner which
WP wh-pronoun who, what
WP$ possessive wh-pronoun whose
WRB wh-abverb where, when
"""
wordnet_pos_dict = { wordnet.ADJ : ["JJ", "JJR", 'JJS'],
                     wordnet.NOUN : ["NN", "NNS", "NNP", "NNPS"],
                     wordnet.VERB : ["MD", "VB", "VBD","VBG", "VBN", "VBP", "VBZ"] , # modal could, will, VB verb, base form take
                     wordnet.ADV : ["RB", "RBR", "RBS"], # adverb very, silently, comparative better, superlative best
                     "DIGIT"         : ["CD"],
                     "PREP"      : ["IN", "TO", "WDT", "WP", "WRB"],
                     "PRP"       : ["PRP"], # PRP personal pronoun I, he, she
                     "PDT"       : ["PDT"], # predeterminer ‘all the kids’
                     "POS"       : ["POS", "PRP$","WP$"], # POS possessive ending parent’s, PRP$ possessive pronoun my, his, hers
                     "OTHER"     : ["CC", "DT", "EX", "FW", "LS", "RP", "UH"]
             }


def get_wordnet_pos_func(word, only_wordnet_types=False):
    '''
    Maps the respective POS tag of a word to the format accepted by the lemmatizer of wordnet
    
    Args: word (str or list): Word to which the function is to be applied, string
    Returns: POS tag or list[POS tag], readable for the lemmatizer of wordnet
    '''
    if isinstance(word, str):
        p_tag = pos_tag([word])[0][1]

        for k, v in wordnet_pos_dict.items():
            if p_tag in v:
                if only_wordnet_types:
                    if k != wordnet.ADJ and k != wordnet.NOUN and k != wordnet.VERB and k != wordnet.ADV: 
                        return wordnet.NOUN
                return k               
        return ""
    elif isinstance(word, list):
        res = []
        for w in word:
            r = get_wordnet_pos_func(w)
            res.append(r)
        return res

tag_dict = { "J": wordnet.ADJ,
             "N": wordnet.NOUN,
             "V": wordnet.VERB,
             "R": wordnet.ADV
             }

def get_wordnet_pos_func_old(word):
    '''
    Maps the respective POS tag of a word to the format accepted by the lemmatizer of wordnet
    
    Args: word (str or list): Word to which the function is to be applied, string
    Returns: POS tag or list[POS tag], readable for the lemmatizer of wordnet
    '''
    if isinstance(word, str):
        p_tag = pos_tag([word])[0][1]

        tag = p_tag[0].upper()

        if p_tag.startswith('RB'):
            """
            RB adverb very, silently,
            RBR adverb, comparative better
            RBS adverb, superlative best
            """
            return wordnet.ADV
        elif p_tag.startswith('MD'):
            return wordnet.VERB
        elif p_tag.startswith('ADJ') or p_tag.startswith('ADV'):
            tag = p_tag[:3].upper()

        res = tag_dict.get(tag, "")
        return res
    elif isinstance(word, list):
        res = []
        for w in word:
            r = get_wordnet_pos_func(w)
            res.append(r)
        return res

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Cleanning function

def normalize_accented_chars(text):
    '''
    Removes all accented characters from a string, if present
    Args: text (str): String to which the function is to be applied, string
            ex: Hello, is your name bob 55? Jean-Marie est 3ème ! Est-ce que tu l'as vu aujourd'hui ?é où à î @ # &
            > Hello, is your name bob 55? Jean-Marie est 3eme ! Est-ce que tu l'as vu aujourd'hui ?e ou a i @ # &
    Returns: Clean string without accented characters
    '''
    if isinstance(text, str):
        return unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore').strip()
    elif isinstance(text, list):
        res = []
        for s in text:
            res.append(normalize_accented_chars(s))
        res = list(filter(None, res))
        return res
    

def remove_html_tags_with_regex(text):
    '''
    Removes HTML-Tags from a string, if present
    Args: text (str): String to which the function is to be applied, string
    Returns: Clean string without HTML-Tags
    ''' 
    pattern = get_regex_html_tag()
    text = re.sub(pattern,'', text)
    return text


def remove_url_func(text):
    '''
    Removes URL addresses from a string, if present
    Args: text (str): String to which the function is to be applied, string
    Returns: Clean string without URL addresses
    ''' 
    pattern = get_regex_urls()
    return re.sub(pattern, '', text)


def remove_punctuation_func(text):
    '''
    Removes all punctuation from a string, if present
    Args:
        text (str): String to which the function is to be applied, string
        ex: Hello, is your name bob 55? Jean-Marie est 3ème !
            > Hello is your name bob 55 JeanMarie est 3ème 
    Returns: Clean string without punctuations /!\ Attention, supprime les tirets, donc les mots composés sont collés
    '''
    # pourrait être remplacé par : string.punctuation ?
    return re.sub(get_regex_non_punctuations(), '', text)

def remove_punctuation_on_tokens(tokens):
    '''
    Removes all punctuation from a string, if present
    
    Args:
        text (str): String to which the function is to be applied, string
    
    Returns:
        Clean string without punctuations
    '''
    stopwords = list()

    for p in string.punctuation:
        stopwords.append(p)

    words_res_list = [ word for word in tokens if word not in stopwords]
    return words_res_list


def remove_types_on_tokens(tokens, types_to_remove=[], types_to_keep=[], verbose=0):
    '''
    Removes all punctuation from a string, if present
    
    Args:
        text (str): String to which the function is to be applied, string
    
    Returns:
        Clean string without punctuations
    '''
    if isinstance(tokens, str):
        return remove_verbs_on_tokens(ara_tokenize(tokens, verbose=verbose))
    else:    
        if types_to_remove is not None and len(types_to_remove)>0:
            types_to_remove = list(map(lambda x: x.upper(), types_to_remove))
        elif types_to_keep is not None and len(types_to_keep) > 0:
            types_to_remove = [ele for ele in list(wordnet_pos_dict.keys()) if ele not in types_to_keep]
        words_res_list = [ word for word in tokens if get_wordnet_pos_func(word=word) not in types_to_remove ]
        return words_res_list


def remove_verbs_on_tokens(tokens, verbose=0):
    '''
    Removes all punctuation from a string, if present
    
    Args:
        text (str): String to which the function is to be applied, string
    
    Returns:
        Clean string without punctuations
    '''
    if isinstance(tokens, str):
        return remove_verbs_on_tokens(ara_tokenize(tokens, verbose=verbose))
    else:    
        words_res_list = [ word for word in tokens if not is_verb(word=word, verbose=verbose) ]
        return words_res_list


def remove_irr_char_func(text):
    '''
    Removes all irrelevant characters (numbers and punctuation) from a string, if present
    
    Args: text (str): String to which the function is to be applied, string
    Returns: Clean string without irrelevant characters
    '''
    if isinstance(text, str):
        return re.sub(get_regex_alphabetique_simple(), ' ', text).strip()
    elif isinstance(text, list):
        res = []
        for s in text:
            res.append(remove_irr_char_func(s))
        res = list(filter(None, res))
        return res





def remove_extra_whitespaces_func(text):
    '''
    Removes extra whitespaces from a string, if present
    
    Args: text (str): String to which the function is to be applied, string
    Returns: Clean string without extra whitespaces
    ''' 
    if isinstance(text, str):
        return re.sub(get_regex_extra_spaces(), ' ', text).strip()
    elif isinstance(text, list):
        res = []
        for s in text:
            res.append(remove_extra_whitespaces_func(s))
        return res


def remove_digit_from_tokens(tokens):
    '''
    Removes all punctuation from a string, if present
    
    Args: text (str): String to which the function is to be applied, string
    
    Returns: Clean string without punctuations
    '''
    stopwords = list()

    for p in string.digits:
        stopwords.append(p)

    words_res_list = [ word for word in tokens if word not in stopwords]
    return words_res_list


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# TOKENIZATION functions

def ara_tokenize(text, use_reg=True, use_nlk=False, verbose=0):
    res = None

    if isinstance(text, str):
        if use_reg:
            res = re.findall(get_regex_tokens(), text.lower())
        if use_nlk:
            res = word_tokenize(text.lower())

    elif isinstance(text, list):
        res = [ ara_tokenize(sentence,use_reg=use_reg, use_nlk=use_nlk, verbose=verbose)  for sentence in text]
        res = list(filter(None, res))

    return res

def df_word_tokenize(df, text_col_name, token_col_name="word_tokenize", use_reg=True, use_nlk=False, verbose=0):
    df_token = df.copy()

    df_token[token_col_name] = df_token[text_col_name].apply(lambda x: ara_tokenize(x, use_reg=use_reg, use_nlk=use_nlk, verbose=verbose))
    return df_token

import spacy

spacy_lg = {"english" : "en_core_web_sm", "french":"fr_core_news_sm"}
spacy_lg_accuracy = {"english" : "en_core_web_trf", "french":"fr_dep_news_trf"}

def lemmatize(text, language="english", applyStopWords=True, stop_words_to_add=[] ,text_expected=False, versbose=0):
    nlp = spacy.load(spacy_lg[language])
    doc = nlp(text)

    if stop_words_to_add is not None and isinstance(stop_words_to_add, list):
        for st in stop_words_to_add:
            nlp.Defaults.stop_words.add(st)

    tokens = None
    if applyStopWords:
        tokens = [token.lemma_ for token in doc if not (token.is_stop or token.is_punct)]
    else:
        tokens = [token.lemma_ for token in doc]

    if text_expected:
        return ' '.join(tokens)
    return tokens

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Evaluating functions

def words_by_weight(words_dic):
    res_dic = {}
    count_keys = sorted(words_dic.values(), reverse=True)
    for k in count_keys:
        res_dic[k] = []

    for key, v in words_dic.items():
        res_dic[v].append(key)
    
    return res_dic


def word_count_func(text):
    '''
    Counts words within a string
    
    Args: text (str or list[str] ): String to which the function is to be applied, string
    Returns: Number of words within a string, integer
    ''' 
    if isinstance(text, str):
        return len(text.split())
    elif isinstance(text, list):
        return word_count_func(' '.join(text))
    return len(text.split())


def norm_stemming_func(text):
    '''
    Stemming tokens from string
    
    Step 1: Use word_tokenize() to get tokens from string
    Step 2: Use PorterStemmer() to stem the created tokens
    
    Args: text (str or list[str] ): String to which the functions are to be applied, string
    Returns: str or list[str] with stemmed words
    ''' 
    words = None 
    if isinstance(text, str):
        words = word_tokenize(text)
        text = ' '.join([PorterStemmer().stem(word) for word in words])
    elif isinstance(text, list):
        words = text 
        text = [PorterStemmer().stem(word) for word in words]
    return text


def norm_lemm_func(text_param, pos=None, auto_type=False, is_text_list=False):
    '''
    Lemmatize tokens from string
    
    Step 1: Use word_tokenize() to get tokens from string
    Step 2: Use WordNetLemmatizer() to lemmatize the created tokens
    
    Args: text (str or list[str] ): String to which the functions are to be applied, string
    Returns: str or list[str] with lemmatized words
    '''  
    text = []
    words = None 
    if isinstance(text_param, str):
        words = word_tokenize(text_param)
        
        if pos is not None and isinstance(pos, str) and len(pos) > 0:
            text = ' '.join([WordNetLemmatizer().lemmatize(word, pos=pos.lower()) for word in words])
        elif auto_type:
            text = ' '.join([WordNetLemmatizer().lemmatize(word, pos=get_wordnet_pos_func(word, only_wordnet_types=True)) for word in words])
        else:
            text = ' '.join([WordNetLemmatizer().lemmatize(word) for word in words])
    elif isinstance(text_param, list):
        if is_text_list:
            for t in text_param:
                text.append(norm_lemm_func(text_param=t, pos=pos, auto_type=auto_type, is_text_list=is_text_list))
        else:
            words = text_param 
            text = [norm_lemm_func(text_param=word, pos=pos, auto_type=auto_type, is_text_list=is_text_list) for word in words]
        text = list(filter(None, text))
    return text

def norm_lemm_v_func(text_param, is_text_list=False):
    '''
    Lemmatize tokens from string 
    
    Step 1: Use word_tokenize() to get tokens from string
    Step 2: Use WordNetLemmatizer() to lemmatize the created tokens
            POS tag is set to 'v' for verb
    
    Args: text (str or list[str] ): String to which the functions are to be applied, string
    Returns: str or list[str] with lemmatized words
    '''  
    return norm_lemm_func(text_param, pos='v', is_text_list=is_text_list)


def norm_lemm_a_func(text_param, is_text_list=False):
    '''
    Lemmatize tokens from string
    
    Step 1: Use word_tokenize() to get tokens from string
    Step 2: Use WordNetLemmatizer() to lemmatize the created tokens
            POS tag is set to 'a' for adjective
    
    Args: text (str or list[str] ): String to which the functions are to be applied, string
    Returns: str or list[str] with lemmatized words
    ''' 
    return norm_lemm_func(text_param, pos='a', is_text_list=is_text_list)


def norm_lemm_POS_tag_func(text_param, is_text_list=False):
    '''
    Lemmatize tokens from string
    
    Step 1: Use word_tokenize() to get tokens from string
    Step 2: Use WordNetLemmatizer() to lemmatize the created tokens
            POS tag is determined with the help of function get_wordnet_pos()
    
    Args: text (str or list[str] ): String to which the functions are to be applied, string
    Returns: str or list[str] with lemmatized words
    ''' 
    if isinstance(text, str):   
        words = word_tokenize(text)
        text = ' '.join([WordNetLemmatizer().lemmatize(word, get_wordnet_pos_func(word)) for word in words])
        return text
    elif isinstance(text, list):
        words = text
        text = [WordNetLemmatizer().lemmatize(word, get_wordnet_pos_func(word)) for word in words]
    return text


def token_to_sentence(token):
    res = ""
    if isinstance(token, list):
        for item in token:
            it = item
            if isinstance(it, str):
                res += ' '.join(token)
                break

            # # Pour gérer les liste de listes de phrases
            res += ' '.join(it)
            res += ' '
    elif isinstance(token, str):
        res = token
    return res.strip()


def calculate_corpus_tf_idf_with_scikitlearn(text_files, input='filename', stop_words='english', stack=True, with_total_row=False, titles =None, verbose=0):
    """_summary_

    Args:
        text_files (list(str)): liste des noms de fichiers
        input (str, optional): _description_. Defaults to 'filename'.
        stop_words (str, optional): _description_. Defaults to 'english'.
        stack (bool, optional): _description_. Defaults to True.
        with_total_row (bool, optional): _description_. Defaults to False.
        verbose (int, optional): _description_. Defaults to 0.

    Returns:
        _type_: _description_
    """
    tfidf_vectorizer = TfidfVectorizer(input=input, stop_words=stop_words)
    tfidf_vector = tfidf_vectorizer.fit_transform(text_files)
    if titles is None:
        titles = [Path(text).stem for text in text_files]
    
    tfidf_df = pd.DataFrame(tfidf_vector.toarray(), index=titles, columns=tfidf_vectorizer.get_feature_names())
    if with_total_row: tfidf_df.loc['00_Document Frequency'] = (tfidf_df > 0).sum()
    if stack:
        tfidf_df = tfidf_df.stack().reset_index()
        tfidf_df = tfidf_df.rename(columns={0:'tfidf', 'level_0': input,'level_1': 'term', 'level_2': 'term'})
        if verbose:
            print(tfidf_df.columns)
        tfidf_df = tfidf_df.sort_values(by=[input,'tfidf'], ascending=[True,False]).groupby([input])
    return tfidf_df


def co_occurrence(sentences, window_size):
    d = defaultdict(int)
    vocab = set()
    for text in sentences:
        # preprocessing (use tokenizer instead)
        text = text.lower().split()
        # iterate over sentences
        for i in range(len(text)):
            token = text[i]
            vocab.add(token)  # add to vocab
            next_token = text[i+1 : i+1+window_size]
            for t in next_token:
                key = tuple( sorted([t, token]) )
                d[key] += 1
    
    # formulate the dictionary into dataframe
    vocab = sorted(vocab) # sort vocab
    df = pd.DataFrame(data=np.zeros((len(vocab), len(vocab)), dtype=np.int16),
                      index=vocab,
                      columns=vocab)
    for key, value in d.items():
        df.at[key[0], key[1]] = value
        df.at[key[1], key[0]] = value
    return df


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# STOP WORDS functions

def remove_english_stopwords_func(text, stopwords=None):
    '''
    Removes Stop Words (also capitalized) from a string, if present
    
    Args: text (str): String to which the function is to be applied, string
    Returns: Clean string without Stop Words
    ''' 
    return remove_stopwords_func(text, language="english", sw=stopwords)


def remove_french_stopwords_func(text, stopwords=None):
    '''
    Removes Stop Words (also capitalized) from a string, if present
    
    Args: text (str): String to which the function is to be applied, string
    Returns: Clean string without Stop Words
    ''' 
    return remove_stopwords_func(text, language="french", sw=stopwords)

def remove_stopwords_func(text, language="french", sw=None):
    '''
    Removes Stop Words (also capitalized) from a string, if present
    
    Args:
        text (str or list): String to which the function is to be applied, string
        language (str, optionnal) : french, english, ... default : french
    Returns: Clean (str or list) without Stop Words
    ''' 
    res = None
    # check in lowercase 
    if isinstance(text, str):
        res_list = remove_stopwords_func(text.split(" "), language=language, sw=stopwords)
        text = ' '.join(res_list)    
        res = text
    elif isinstance(text, list):
        if sw is None :
            sw = list(stopwords.words(language))
        if isinstance(sw, WordListCorpusReader):
            sw = sw.words()
        
        res = []
        for sentence in text:
            token_temp = sentence.split()
            t = [token.strip() for token in token_temp if token.lower() not in sw]
            res.append(' '.join(t))
        res = list(filter(None, res))
    return res

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#                                              N_GRAMS
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def ara_ngrams(text, min_n=1, max_n=3, pas_n=1, verbose=0, is_token=False):

    tokens = []

    if isinstance(text, str):
        # Tokenize
        unigrams = ara_tokenize(text)
        if verbose:
            print(f"{len(unigrams)} => unigrams found")
        for i in range(max_n, min_n, -pas_n):
            if len(unigrams) >= i:
                gram = ['_'.join(w) for w in  ngrams(unigrams,n=i)]
                if verbose:
                    print(f"{len(gram)} => {i}grams found")
                    if verbose > 1:
                        print(gram)
                tokens += gram
                if i == 2:
                    unigrams = []
            elif verbose :
                print(f"0 => {i}grams ({len(unigrams)} unigrams only)")

        if min_n == 1:
            tokens += unigrams
    elif isinstance(text, list):
        if is_token:
            st = token_to_sentence(text)
            tokens = ara_ngrams(st, min_n=min_n, max_n=max_n, pas_n=pas_n, verbose=verbose)
        else:
            for txt in text:
                tokens += ara_ngrams(txt, min_n=min_n, max_n=max_n, pas_n=pas_n, verbose=verbose)
   
    return tokens


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#                                              SENTIMENTS ANALYSIS
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# TODO


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#                                              GRAPHICAL FUNCTIONS
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
import altair as alt

def draw_heatmap_tf_idf(top_tfidf, document_col_name = "filename", tdf_idf_col_name='tfidf', term_list=[], alt_renderer='altair_viewer', verbose=0):
    """Let’s make a heatmap that shows the highest TF-IDF scoring words, and let’s put a red dot next to received terms

    src : https://melaniewalsh.github.io/Intro-Cultural-Analytics/05-Text-Analysis/03-TF-IDF-Scikit-Learn.html
    Args:
        top_tfidf (DataFrame or DataFrameGroup): _description_
        tdf_idf_col_name (str) : the tf-idf column name
        term_list (list, optional): Terms in this list will get a red dot in the visualization. Defaults to [].
        alt_renderer (str) : ['html','mimetype', 'notebook', 'altair_viewer'], `mimetype` => Vs Code or `altair_viewer` with pip install altair_viewer, `html` => Jupyter
    """
    alt.renderers.enable(alt_renderer)
    # adding a little randomness to break ties in term ranking
    top_tfidf_plusRand = top_tfidf.copy()
    top_tfidf_plusRand[tdf_idf_col_name] = top_tfidf_plusRand[tdf_idf_col_name] + np.random.rand(top_tfidf.shape[0])*0.0001

    # base for all visualizations, with rank calculation
    base = alt.Chart(top_tfidf_plusRand).mark_point().encode(
        x = 'rank:O',
        y = document_col_name+':N'
    ).transform_window(
        rank = "rank()",
        sort = [alt.SortField(tdf_idf_col_name, order="descending")],
        groupby = [document_col_name],
    )

    # heatmap specification
    heatmap = base.mark_rect().encode(
        color = tdf_idf_col_name+':Q'
    )
    circle = None
    if term_list is not None and len(term_list)>0:
        # red circle over terms in above list
        circle = base.mark_circle(size=100).encode(
            color = alt.condition(
                alt.FieldOneOfPredicate(field='term', oneOf=term_list),
                alt.value('red'),
                alt.value('#FFFFFF00')        
            )
        )

    # text labels, white for darker heatmap colors
    text = base.mark_text(baseline='middle').encode(
        text = 'term:N',
        color = alt.condition(alt.datum.tfidf >= 0.23, alt.value('white'), alt.value('black'))
    )
    if circle is not None:
        # display the three superimposed visualizations
        (heatmap + circle + text).properties(width = 600)
    else:
        # display the three superimposed visualizations
        (heatmap + text).properties(width = 600)

    return base
    
from wordcloud import WordCloud
import matplotlib.pyplot as plt

def draw_word_cloud(texte, stopwords = None, include_numbers = False, min_word_length=4, max_words = 400, random_state = 8, collocations=False, normalize_plurals = True, width = 800, height= 400, verbose=0):
    """_summary_

    Args:
        texte (str or list(str) or dict(str:val)): _description_
        stopwords (list(str), optional): _description_. Defaults to None.
        include_numbers (bool, optional): _description_. Defaults to False.
        min_word_length (int, optional): _description_. Defaults to 4.
        max_words (int, optional): _description_. Defaults to 400.
        random_state (int, optional): _description_. Defaults to 8.
        collocations (bool, optional): _description_. Defaults to False.
        normalize_plurals (bool, optional): _description_. Defaults to True.
        width (int, optional): _description_. Defaults to 800.
        height (int, optional): _description_. Defaults to 400.
        verbose (int, optional): _description_. Defaults to 0.
    """
    # Instantiate a new wordcloud.
    wordcloud = WordCloud(
                random_state = random_state,
                collocations=collocations,
                normalize_plurals = normalize_plurals,
                width = width, 
                height= height,
                include_numbers = include_numbers,
                min_word_length=min_word_length,
                max_words = max_words )
                
    if stopwords is not None:
        wordcloud = WordCloud(
                random_state = random_state,
                collocations=collocations,
                normalize_plurals = normalize_plurals,
                width = width, 
                height= height,
                include_numbers = include_numbers,
                min_word_length=min_word_length,
                max_words = max_words,
                stopwords = [])

    if isinstance(texte, pd.Series):
        texte = texte.to_dict()

    if isinstance(texte, dict):
        wordcloud.fit_words(texte)
    else:
        if not isinstance(texte, str) and isinstance(texte, list):
            # Transform the list of words back into a string 
            texte  = ' '.join(texte)

        # Apply the wordcloud to the text.
        wordcloud.generate(texte)

    # And plot
    fig, ax = plt.subplots(1,1, figsize = (9,6))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")



# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#                                              TO_REMOVE OR IMPLEMENTS
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def calculate_tf_for_word(word, document):
    """ Term frequency (TF) is how often a word appears in a document, divided by how many words there are.
        TF(t) = (Number of times term t appears in a document) / (Total number of terms in the document)

    Args:
        word (_type_): _description_
        document (_type_): _description_
    """
    nltk.Counter()

def calculate_idf_for_word(word, document):
    """ Term frequency is how common a word is, inverse document frequency (IDF) is how unique or rare a word is.
        IDF(t) = log_e(Total number of documents / Number of documents with term t in it)

    Args:
        word (_type_): _description_
        document (_type_): _description_
    """
    pass

def calculate_tf_idf(word, document):
    """tf-idf = term_frequency * inverse_document_frequency

    Args:
        word (_type_): _description_
        document (_type_): _description_
    """
    tf_idf = calculate_tf_for_word(word=word, document=document) * calculate_idf_for_word(word=word, document=document)
    return tf_idf

def _remove_non_alphanumeric_func(text):
    '''
    Removes all punctuation, accented chars, ... from a string, if present
    
    Args:
        text (str): String to which the function is to be applied, string
        ex : Hello, is your name bob 55? Jean-Marie est 3ème !
             > Hello  is your name bob 55  Jean Marie est 3 me  
    Returns:
        Clean string without punctuations, /!\ Attention, supprime les accents, tiret et les remplace par des espaces
    '''
    return re.sub(r'[^a-zA-Z0-9]', ' ', text)

# C'est mieux s'utiliser les regex
def _remove_html_tags_func(text):
    '''
    Removes HTML-Tags from a string, if present
    Args: text (str): String to which the function is to be applied, string
    Returns: Clean string without HTML-Tags
    ''' 
    return BeautifulSoup(text, 'html.parser').get_text()


# def stem_tokens(tokens, stemmer):
#     stemmed = []
#     for item in tokens:
#         stemmed.append(stemmer.stem(item))
#     return stemmed

# def tokenize(text):
#     tokens = nltk.word_tokenize(text)
#     stems = stem_tokens(tokens, stemmer)
#     return stems

# def calculate_document_tf_idf(word_list, stop_words=sw, verbose=0):
#     tfidf = TfidfVectorizer(tokenizer=tokenize, stop_words=sw)
#     values = tfidf.fit_transform(word_list)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#                                              TESTS
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def test_remove_stopwords_func():
    list_text = ['cnn',  'rory',  'mcilroy',  'is',  'off',  'to',  'a',  'good',
                'start',  'at',   'the',  'scottish', 'open', 'he', 's', 'hoping', 'for', 'a', 'good', 'finish',
                'too', 'after', 'missing', 'the', 'cut', 'at', 'the', 'irish', 'open', 'mcilroy', 'shot', 'a',
                'course', 'record', '7', 'under', 'par', '64', 'at', 'royal', 'aberdeen', 'on', 'thursday', 'and',
                'he', 'was', 'actually', 'the','second', 'player', 'to', 'better', 'the', 'old', 'mark', 'sweden',
                's', 'kristoffer', 'broberg', 'had', 'earlier', 'fired', 'a', '65', 'mcilroy', 'carded', 'eight',
                'birdies', 'and', 'one', 'bogey', 'in', 'windy', 'chilly', 'conditions', 'going', 'out', 'this',
                'morning', 'in', 'these', 'conditions', 'i', 'thought', 'anything', 'in', 'the', '60s', 'would',
                'be', 'a', 'good', 'score', 'so', 'to', 'shoot', 'something', 'better', 'than', 'that', 'is',
                'pleasing', 'mcilroy', 'was', 'quoted', 'as', 'saying', 'by', 'the', 'european', 'tour', 's',
                'website', 'a', 'win', 'sunday', 'would', 'be', 'the', 'perfect', 'way', 'for', 'former', 'no',
                '1', 'mcilroy', 'to', 'prepare', 'for', 'the', 'british', 'open', 'which', 'starts', 'next', 'week',
                'at', 'royal', 'liverpool', 'he', 'won', 'the', 'last', 'of', 'his','two', 'majors', 'in', '2012', 'everything',
                'was', 'pretty', 'much', 'on', 'mcilroy', 'said', 'i', 'controlled', 'my', 'ball', 'flight', 'really', 'well',
                'which', 'is', 'the', 'key', 'to', 'me', 'playing', 'well', 'in', 'these', 'conditions', 'and', 'on',
                'these', 'courses', 'i', 've', 'been', 'working', 'the', 'last', '10', 'days', 'on', 'keeping', 'the',
                'ball', 'down', 'hitting', 'easy', 'shots', 'and', 'taking', 'spin', 'off', 'it', 'and', 'i', 'went', 'out',
                'there', 'today', 'and', 'really', 'trusted', 'what', 'i', 'practiced', 'last', 'year', 'phil', 'mickelson', 'used',
                'the', 'scottish', 'open', 'at', 'castle', 'stuart', 'as', 'the', 'springboard', 'to', 'his', 'british', 'open',
                'title', 'and', 'his', '68', 'leaves', 'him', 'well', 'within', 'touching', 'distance', 'of', 'mcilroy', 'mickelson',
                'needs', 'a', 'jolt', 'of', 'confidence', 'given', 'that', 'lefty','has', 'slipped', 'outside', 'the', 'top',
                '10', 'in', 'the', 'rankings', 'and', 'hasn', 't', 'finished', 'in', 'the', 'top', '10', 'on', 'the',
                'pga', 'tour', 'this', 'season', 'i', 'thought', 'it', 'was', 'tough', 'conditions', 'mickelson', 'said',
                'in', 'an', 'audio', 'interview', 'posted', 'on', 'the', 'european', 'tour', 's', 'website', 'i', 'was',
                'surprised', 'to', 'see', 'some', 'low', 'scores', 'out', 'there', 'because', 'it', 'didn', 't', 'seem', 'like',
                'it', 'was', 'playing', 'easy', 'and', 'the', 'wind', 'was', 'pretty', 'strong', 'i', 'felt', 'like', 'i', 'played', 'well',
                'and', 'had', 'a', 'good', 'putting', 'day', 'it', 'was', 'a', 'good', 'day', 'last', 'year', 's', 'u', 's', 'open', 'champion',
                'justin', 'rose', 'was', 'tied', 'for', '13th', 'with', 'a', '69', 'but', 'jonas', 'blixt', 'who', 'tied', 'for', 'second', 'at',
                'the', 'masters', 'was', 'well', 'adrift', 'following', 'a', '74']
    
    res_list = remove_english_stopwords_func(list_text)
    print(res_list)

    text = 'cnn rory mcilroy is off to a good start at the scottish open he hoping for a good finish too after missing the cut at the irish open mcilroy shot a course record 7 under 64 at royal aberdeen thursday and he was actually the second player to better the old mark sweden kristoffer broberg had earlier fired a 65 mcilroy carded eight birdies and one bogey in windy chilly conditions going out this morning in these conditions i thought anything in the 60s would be a good score so to shoot something better than that is pleasing mcilroy was quoted saying by the european tour website a win sunday would be the perfect way for former no 1 mcilroy to prepare for the british open which starts next week at royal liverpool he won the last of his two majors in 2012 everything was pretty much mcilroy said i controlled my ball flight really well which is the key to playing well in these conditions and these courses i ve been working the last 10 days keeping the ball down hitting easy shots and taking spin off it and i went out there today and really trusted what i practiced last year phil mickelson used the scottish open at castle stuart the springboard to his british open title and his 68 leaves him well within touching distance of mcilroy mickelson needs a jolt of confidence given that lefty has slipped outside the top 10 in the rankings and hasn finished in the top 10 the pga tour this season i thought it was tough conditions mickelson said in an audio interview posted the european tour website i was surprised to see some low scores out there because it didn seem like it was playing easy and the wind was pretty strong i felt like i played well and had a good putting day it was a good day last year u open champion justin rose was tied for 13th with a 69 but jonas blixt who tied for second at the masters was well adrift following a 74'
    res_txt = remove_english_stopwords_func(text)
    print(res_txt)


def test_tfd_idf():
    # Test du graph
    import matplotlib.pyplot as plt
    from os import getcwd
    import sys
    file_path = getcwd() + "\\"
    print(file_path)
    sys.path.append("C:\\Users\\User\\WORK\\workspace-ia\\PERSO\\")
    from ara_commons.ara_file import list_dir_files
    dir_path = file_path
    if "simplon\\NLP-CNN" not in file_path:
        dir_path += "simplon\\NLP-CNN\\cnn2\\"
    else:
        dir_path += "cnn2\\"
    print(dir_path)
    text_files = list_dir_files(dir_path=dir_path, endwith=".story", verbose=1)
    print(text_files)
    tfidf_df = calculate_corpus_tf_idf_with_scikitlearn(text_files, input='filename', stop_words='english', stack=True, with_total_row=True, verbose=1)
    print(tfidf_df.size)
    print(tfidf_df.head(20))
    base = draw_heatmap_tf_idf(tfidf_df.head(20), tdf_idf_col_name='tfidf', term_list=[], verbose=1)
    return base

def test_ara_ngrams():
    list_text = ['olive oil',
                'knob butter',
                'onion',
                'sausagemeat skinned sausages',
                'zest lemon',
                'fresh white breadcrumbs',
                'ready eat dried apricots',
                'chestnut canned vacuum packed',
                'fresh dried thyme',
                'cranberries fresh frozen',
                'boneless skinless chicken breasts',
                'pack ready made shortcrust pastry',
                'beaten egg glaze',
                'butter',
                'dark muscovado sugar',
                'luxury dried fruit one includes peel glace cherries',
                'zest juice orange',
                'zest lemon',
                'fl cherry brandy brandy',
                'macadamia nut',
                'eggs lightly beaten',
                'almond',
                'plain flour',
                'baking powder',
                'spice']
    ara_ngrams(list_text, 1, 3, 1, 1)

def test_remove_verbs_on_tokens():
    tokens = ['vegetable','do','make','stock']
    print(remove_verbs_on_tokens(tokens))

    tokens = ['vegetable','do','make','stock']
    print(remove_verbs_on_tokens(" ".join(tokens)))



def test_norm_lemma():
    temp = ["cooking apples chunks", "eating apple chunks", "onion sliced", "root ginger", "peppercorn", "granulated sugar", "cider vinegar", "cranberry",
            "plain flour", "almonds", "baking powder", "bicarbonate soda", "cinnamon", "golden caster sugar", "marzipan diced", "pistachios", "toasted flaked almonds", "sultanasor raisins", "cherriesor cranberries", "apricots diced", "eggs", "unsalted butter melted cooled", "full fat natural yogurt", "almond extract", "icing sugar", "paper muffin cases used tulip cases"
            ]
    norm_lemm_func(temp, is_text_list=True)
    norm_lemm_func(temp, auto_type=True, is_text_list=True)
    norm_lemm_a_func(temp, is_text_list=True)
    norm_lemm_v_func(temp, is_text_list=True)


def test_get_type():
    word_list = ['chestnut', 'canned', 'or', 'vacuum', 'packed', 'chopped', 'boneless', 'skinless', 'chicken', 'breasts', 'beaten', 'egg', 'to', 'glaze', 'butter', 'chopped', 'chop', 'eggs', 'beaten', 'beat', 'mixed', 'spice', 'g', 'luxury', 'mixed', 'fruit', 'oz', 'pecannuts']
    for word in word_list:
        print(word, "=", get_wordnet_pos_func(word))

if __name__ == "__main__":
    
    test_norm_lemma()
    print("------------------------------------------------")

    test_get_type()
    print("------------------------------------------------")

    test_remove_verbs_on_tokens()
    print("------------------------------------------------")

    test_remove_stopwords_func()
    test ={'march': 4, '14': 11, 'be': 41, 'my': 1}
    print(words_by_weight(test))
    print("------------------------------------------------")

    # Test du graph
    test_tfd_idf()
    print("------------------------------------------------")

    
    text = ["I go to school every day by bus .",
            "i go to theatre every night by bus"]
    df = co_occurrence(text, 2)
    print(df)
    print("------------------------------------------------")
    test_ara_ngrams()
    print("------------------------------------------------")

    

