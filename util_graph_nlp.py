import seaborn as sns
import missingno as msno
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
import numpy as np
import pandas as pd

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report

from util_print import *
from util_graph import color_graph_background, get_color_names , PLOT_FIGURE_BAGROUNG_COLOR, PLOT_BAGROUNG_COLOR


from wordcloud import WordCloud
import matplotlib.pyplot as plt

def draw_word_cloud(texte, stopwords = None, include_numbers = False, min_word_length=4, max_words = 400, random_state = 8, collocations=False, normalize_plurals = True, width = 800, height= 400, ax=None, verbose=0):
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

    if ax is None:
        # And plot
        fig, ax = plt.subplots(1,1, figsize = (9,6))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis("off")
    else:
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis("off")

from statsmodels.graphics.mosaicplot import mosaic

def draw_mosaic(df_param, col_mosaic_y, col_mosaic_x_categorie = 'Categorie', categorie=None, size=(15,20), verbose=0):
    
    df = df_param.copy()
    title = f"ALL > {col_mosaic_y} x {col_mosaic_x_categorie}"
    if categorie is not None:
        df = df_param[df_param['Categorie']==categorie]
        title = f"{categorie} > {col_mosaic_y} x {col_mosaic_x_categorie}"
    
    crosstable = pd.crosstab(df[col_mosaic_y], df[col_mosaic_x_categorie])
    crosstable = pd.DataFrame(crosstable)
    
    labelizer=lambda k: f"{k[1]} :{crosstable.loc[k[1],k[0]]}"
    
    fig, ax = color_graph_background()
    fig, _ = mosaic(df,[col_mosaic_x_categorie, col_mosaic_y],labelizer=labelizer, ax=ax)
    ax.set_title(title)
    ax.set_yticks([])
    fig.set_size_inches(size[0],size[1], forward=True)
    
    return fig, ax, crosstable
    

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report

def draw_confusion(y_test, pred, labels=None, size=(20, 20), verbose=0):
    short_name = "draw_confusion"
    if labels is None:
        labels = set(y_test.unique()) | set(pred.unique())
        if verbose>0:
            print(f"[{short_name}]\tINFO : {len(labels)} labels")
            if verbose>1:
                print(f"[{short_name}]\tDEBUG : {labels}")
    
    nb_col = len(pred)
    figure, axe = color_graph_background()
    
    # Matrice de confusion
    cm = confusion_matrix(y_test, pred)
    if verbose>0:
        print(f"[{short_name}]\tINFO :\n{cm}")
        
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(ax=axe)
    
    if verbose>1:
        print(classification_report(y_test, pred))
        
    figure.set_size_inches(size[0], size[1], forward=True)
    figure.set_dpi(100)
    plt.xticks(rotation=45)
    plt.show()


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# %% TODO                                              TEST
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# %%                                              MAIN
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
if __name__ == ('__main__'):
    short_name = "util_graph_nlp"
    info(short_name, "---------------- TESTS ------------------ START")
    
    info(short_name, "---------------- TESTS ------------------ END")
    