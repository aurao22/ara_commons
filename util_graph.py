import seaborn as sns
import missingno as msno
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
import numpy as np
import pandas as pd

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report

from util_print import *

PLOT_FIGURE_BAGROUNG_COLOR = 'white'
PLOT_BAGROUNG_COLOR = PLOT_FIGURE_BAGROUNG_COLOR

def get_color_names():
    colors = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)

    # Sort colors by hue, saturation, value and name.
    by_hsv = sorted((tuple(mcolors.rgb_to_hsv(mcolors.to_rgba(color)[:3])), name)
                    for name, color in colors.items())
    sorted_names = [name for hsv, name in by_hsv]
    return sorted_names

def color_graph_background(ligne=1, colonne=1):
    figure, axes = plt.subplots(ligne,colonne)
    figure.patch.set_facecolor(PLOT_FIGURE_BAGROUNG_COLOR)
    if isinstance(axes, np.ndarray):
        for axe in axes:
            # Traitement des figures avec plusieurs lignes
            if isinstance(axe, np.ndarray):
                for ae in axe:
                    ae.set_facecolor(PLOT_BAGROUNG_COLOR)
            else:
                axe.set_facecolor(PLOT_BAGROUNG_COLOR)
    else:
        axes.set_facecolor(PLOT_BAGROUNG_COLOR)
    return figure, axes


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


import seaborn as sns

def draw_barh_stacked(data, title="Graphe de confusion", size=(20, 40), save_img_path=None, display=True, label_col_name='label', verbose=0):
    sns.set()
    colors = ["green", "red", "orange"]
    
    figure, axes = plt.subplots(1, 1)
    data.set_index(label_col_name).plot(kind='barh',stacked=True, ax=axes, color=colors)
    figure.set_size_inches(size[0], size[1], forward=True)
    figure.set_dpi(100)    
    plt.title(title, fontsize=14)
    if save_img_path is not None:
        plt.savefig(save_img_path)
        
    if display:
        plt.show()
    

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
# %%                                              TEST
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def _test_color_name(verbose=1):
    
    c = get_color_names()
    
    assert c is not None and len(c)>0

    if verbose>0:
        debug(short_name, c)


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# %%                                              MAIN
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
if __name__ == ('__main__'):
    short_name = "util_graphe"
    info(short_name, "---------------- TESTS ------------------ START")
    _test_color_name()
    info(short_name, "---------------- TESTS ------------------ END")
    