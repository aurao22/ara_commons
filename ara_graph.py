
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import seaborn as sns
from matplotlib import cm

# ----------------------------------------------------------------------------------
#                        GRAPHIQUES
# ----------------------------------------------------------------------------------
PLOT_FIGURE_BAGROUNG_COLOR = 'white'
PLOT_BAGROUNG_COLOR = PLOT_FIGURE_BAGROUNG_COLOR


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


def draw_correlation_graphe(df, title, verbose=False, annot=True, fontsize=5):
    """Dessine le graphe de corrélation des données

    Args:
        df (DataFrame): Données à représenter
        verbose (bool, optional): Mode debug. Defaults to False.
    """
    corr_df = df.corr()
    if verbose:
        print("CORR ------------------")
        print(corr_df, "\n")
    figure, ax = color_graph_background(1,1)
    figure.set_size_inches(18, 15, forward=True)
    figure.set_dpi(100)
    figure.suptitle(title, fontsize=16)
    sns.heatmap(corr_df, annot=annot, annot_kws={"fontsize":fontsize})
    plt.xticks(rotation=45, ha="right", fontsize=7)
    plt.yticks(fontsize=7)
    plt.show()


def draw_y(X_test, x_col_name, y_test, y_pred, dict_y_pred):
    figure, axes = color_graph_background(2,3)
    i = 0
    j = 0
    for model_name, y_pred in dict_y_pred.items():
        if "SGDRegressor" in model_name:
            continue
        else:
            axe = axes[i][j]
            axe.scatter(X_test[x_col_name], y_test/1000, color='blue', label='expected')
            axe.scatter(X_test[x_col_name], y_pred/1000, color='red', marker='+', label='prediction')
            axe.set_title(model_name)
            axe.yaxis.set_major_formatter(FuncFormatter(lambda x, p: "{:,.0f} K€". format(x)))
            axe.set_ylabel('median_house_value')
            axe.legend()
            j += 1
            if j == 3:
                j = 0
                i += 1
    figure.suptitle('Comparaison predictions vs expected > x='+x_col_name, fontsize=16)
    figure.set_size_inches(15, 5*3, forward=True)
    figure.set_dpi(100)
    plt.show()



def draw_pie_multiple_by_value(df, column_name, values, compare_column_names, titre="", legend=True, verbose=False, max_col = 4 , colors=None):
    """ Fonction pour dessiner un graphe pie pour la colonne reçue

    Args:
        df (DataFrame): Données à représenter
        column_name (String): colonne à représenter
        country_name_list (list[String], optional): Liste des pays à traiter. Defaults to [].
        verbose (bool, optional): Mode debug. Defaults to False.
    """
    nb_col = len(values)
    nb_row = 1
    if nb_col > max_col:
        more = 1
        if (nb_col % max_col) == 0:
            more = 0
        nb_row = (nb_col//max_col) + more
        nb_col = max_col

    figure, axes = color_graph_background(nb_row,nb_col)
    i = 0
    j = 0
    for val in values:
        ax = axes
        if nb_row == 1:
            ax = axes[i]
            i += 1
        else:
            ax = axes[i][j]
            j += 1
            if j == nb_col:
                i += 1
                j = 0
        _draw_pie(df[df[column_name]==val], compare_column_names, ax, colors=colors, legend=legend, verbose=verbose)
        ax.set_title(column_name+"="+str(val))
        ax.set_facecolor(PLOT_BAGROUNG_COLOR)   
        
    figure.set_size_inches(15, 5*nb_row, forward=True)
    figure.set_dpi(100)
    figure.suptitle(titre, fontsize=16)
    plt.show()
    print("draw_pie_multiple_by_value", column_name," ................................................. END")


def draw_pie_multiple(df, column_names, colors=None, verbose=False, legend=True):
    """Fonction pour dessiner un graphe pie pour la colonne reçue

    Args:
        df (DataFrame): Données à représenter
        column_name (String): colonne à représenter
        country_name_list (list[String], optional): Liste des pays à traiter. Defaults to [].
        verbose (bool, optional): Mode debug. Defaults to False.
    """
    figure, axes = color_graph_background(1,len(column_names))
    i = 0
    for column_name in column_names:
        if len(column_names) > 1:
            _draw_pie(df, column_name, axes[i], colors=colors, legend=legend, verbose=verbose)
        else:
            _draw_pie(df, column_name, axes, colors=colors, legend=legend, verbose=verbose)
        i += 1
    figure.set_size_inches(15, 5, forward=True)
    figure.set_dpi(100)
    figure.patch.set_facecolor(PLOT_FIGURE_BAGROUNG_COLOR)
    figure.suptitle(column_name+" REPARTITION", fontsize=16)
    plt.show()
    print("draw_pie_multiple", column_name," ................................................. END")


def draw_pie_multiple_by_value(df, column_name, values, compare_column_names, titre="", legend=True, verbose=False, max_col = 4 , colors=None):
    """ Fonction pour dessiner un graphe pie pour la colonne reçue

    Args:
        df (DataFrame): Données à représenter
        column_name (String): colonne à représenter
        country_name_list (list[String], optional): Liste des pays à traiter. Defaults to [].
        verbose (bool, optional): Mode debug. Defaults to False.
    """
    nb_col = len(values)
    nb_row = 1
    if nb_col > max_col:
        more = 1
        if (nb_col % max_col) == 0:
            more = 0
        nb_row = (nb_col//max_col) + more
        nb_col = max_col

    figure, axes = color_graph_background(nb_row,nb_col)
    i = 0
    j = 0
    for val in values:
        ax = axes
        if nb_row == 1:
            ax = axes[i]
            i += 1
        else:
            ax = axes[i][j]
            j += 1
            if j == nb_col:
                i += 1
                j = 0
        _draw_pie(df[df[column_name]==val], compare_column_names, ax, colors=colors, legend=legend, verbose=verbose)
        ax.set_title(column_name+"="+str(val))
        ax.set_facecolor(PLOT_BAGROUNG_COLOR)   
        
    figure.set_size_inches(15, 5*nb_row, forward=True)
    figure.set_dpi(100)
    figure.suptitle(titre, fontsize=16)
    plt.show()
    print("draw_pie_multiple_by_value", column_name," ................................................. END")


def _draw_pie(df, column_name, axe, colors=None, legend=True, verbose=False):
    """Fonction pour dessiner un graphe pie pour la colonne reçue

    Args:
        df (DataFrame): Données à représenter
        column_name (String): colonne à représenter
        axe ([type]): [description]
        colors ([type], optional): [description]. Defaults to None.
        verbose (bool, optional): Mode debug. Defaults to False.
    """
    df_nova = df[~df[column_name].isna()][column_name].value_counts().reset_index()
    df_nova = df_nova.sort_values("index")
    # Affichage des graphiques
    axe.pie(df_nova[column_name], labels=df_nova["index"], colors=colors, autopct='%.0f%%')
    if legend:
        axe.legend(df_nova["index"], loc="upper left")
    else:
        legend = axe.legend()
        legend.remove()
    axe.set_title(column_name)
    axe.set_facecolor(PLOT_BAGROUNG_COLOR)