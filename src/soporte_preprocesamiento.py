# Tratamiento de datos
# -----------------------------------------------------------------------
import numpy as np
import pandas as pd

# Otros objetivos
# -----------------------------------------------------------------------
import math

# Gráficos
# -----------------------------------------------------------------------
import seaborn as sns
import matplotlib.pyplot as plt

from scipy.stats import zscore # para calcular el z-score
from sklearn.neighbors import LocalOutlierFactor # para detectar outliers usando el método LOF
from sklearn.ensemble import IsolationForest # para detectar outliers usando el metodo IF
from sklearn.neighbors import NearestNeighbors # para calcular la epsilon
import soporte_preprocesamiento as f

def exploracion_dataframe(dataframe, columna_control):
    """
    Realiza un análisis exploratorio básico de un DataFrame, mostrando información sobre duplicados,
    valores nulos, tipos de datos, valores únicos para columnas categóricas y estadísticas descriptivas
    para columnas categóricas y numéricas, agrupadas por la columna de control.

    Params:
    - dataframe (DataFrame): El DataFrame que se va a explorar.
    - columna_control (str): El nombre de la columna que se utilizará como control para dividir el DataFrame.

    Returns: 
    No devuelve nada directamente, pero imprime en la consola la información exploratoria.
    """
    print(f"El número de datos es {dataframe.shape[0]} y el de columnas es {dataframe.shape[1]}")
    print("\n ..................... \n")

    print(f"Los duplicados que tenemos en el conjunto de datos son: {dataframe.duplicated().sum()}")
    print("\n ..................... \n")
    
    
    # generamos un DataFrame para los valores nulos
    print("Los nulos que tenemos en el conjunto de datos son:")
    df_nulos = pd.DataFrame(dataframe.isnull().sum() / dataframe.shape[0] * 100, columns = ["%_nulos"])
    display(df_nulos[df_nulos["%_nulos"] > 0])
    
    print("\n ..................... \n")
    print(f"Los tipos de las columnas son:")
    display(pd.DataFrame(dataframe.dtypes, columns = ["tipo_dato"]))
    
    
    print("\n ..................... \n")
    print("Los valores que tenemos para las columnas categóricas son: ")
    dataframe_categoricas = dataframe.select_dtypes(include = "O")
    
    for col in dataframe_categoricas.columns:
        print(f"La columna {col.upper()} tiene las siguientes valore únicos:")
        display(pd.DataFrame(dataframe[col].value_counts()).head())    
    
    # como estamos en un problema de A/B testing y lo que realmente nos importa es comparar entre el grupo de control y el de test, los principales estadísticos los vamos a sacar de cada una de las categorías
    
    # for categoria in dataframe[columna_control].unique():
    #     dataframe_filtrado = dataframe[dataframe[columna_control] == categoria]
    
    #     print("\n ..................... \n")
    #     print(f"Los principales estadísticos de las columnas categóricas para el {categoria} son: ")
    #     display(dataframe_filtrado.describe(include = "O").T)
        
    #     print("\n ..................... \n")
    #     print(f"Los principales estadísticos de las columnas numéricas para el {categoria} son: ")
    #     display(dataframe_filtrado.describe().T)


def separar_dataframe(df):
    return df.select_dtypes(include=np.number), df.select_dtypes(include="O")

def plot_numericas(df, tamanio):
    cols_numericas=df.columns

    nfilas=math.ceil(len(cols_numericas) /2)  #es para que si da 4,5, se quede con 5 y no con 4 porque de normal redondea hacia abajo
    fig, axes = plt.subplots(nrows= nfilas, ncols= 2, figsize= tamanio)
    axes=axes.flat
    for indice,col in enumerate(cols_numericas):
        sns.histplot(x=col, data= df, ax= axes[indice], bins=50)
        axes[indice].set_title(col)
        axes[indice].set_xlabel("")


    if len(cols_numericas) %2 != 0:
        fig.delaxes(axes[-1])
    else:
        pass

    plt.tight_layout()
    
def plot_categoricas(df, tamanio, paleta="mako"):
    cols_categoricas=df.columns

    nfilas=math.ceil(len(cols_categoricas) /2)  #es para que si da 4,5, se quede con 5 y no con 4 porque de normal redondea hacia abajo
    fig, axes = plt.subplots(nrows= nfilas, ncols= 2, figsize= tamanio)
    axes=axes.flat
    for indice,col in enumerate(cols_categoricas):
        sns.countplot(x=col, data= df, ax= axes[indice], palette=paleta, order= df[col].value_counts().index)
        axes[indice].set_title(col)
        axes[indice].set_xlabel("")
        axes[indice].tick_params(rotation=90)


    if len(cols_categoricas) %2 != 0:
        fig.delaxes(axes[-1])
    else:
        pass

    plt.tight_layout()    


def relacion_dependiente_categoricas(df, variable_dependiente, tamanio=(15,8), paleta="mako"):
    df_cat= f.separar_dataframe(df)[1]
    cols_categoricas=df_cat.columns
    nfilas=math.ceil(len(cols_categoricas) /2)  #es para que si da 4,5, se quede con 5 y no con 4 porque de normal redondea hacia abajo
    fig, axes = plt.subplots(nrows= nfilas, ncols= 2, figsize= tamanio)
    axes=axes.flat

    for indice, col in enumerate(cols_categoricas):
        datos_agrupados= df.groupby(col)[variable_dependiente].mean().reset_index().sort_values(variable_dependiente, ascending=False)
        sns.barplot(x=col ,y=variable_dependiente, data=datos_agrupados, palette=paleta, ax=axes[indice])
        axes[indice].tick_params(rotation=90)
        axes[indice].set_title(f"Relación entre {col} y {variable_dependiente}")
        axes[indice].set_xlabel("")
    plt.tight_layout()



def relacion_dependiente_numericas(df, variable_dependiente, tamanio=(15,8), paleta="mako"):
    df_numericas= f.separar_dataframe(df)[0]
    cols_numericas=df_numericas.columns
    nfilas=math.ceil(len(cols_numericas) /2)  #es para que si da 4,5, se quede con 5 y no con 4 porque de normal redondea hacia abajo
    fig, axes = plt.subplots(nrows= nfilas, ncols= 2, figsize= tamanio)
    axes=axes.flat

    for indice, col in enumerate(cols_numericas):
        if col == variable_dependiente:
            fig.delaxes(axes[indice])
        else:
            sns.scatterplot(x=col ,y=variable_dependiente, data=df_numericas, palette=paleta, ax=axes[indice])

            axes[indice].set_title(f"Relación entre {col} y {variable_dependiente}")
            axes[indice].set_xlabel("")
    plt.tight_layout()


def matriz_correlacion(df,tamanio=(10,7)):
    matriz_corr=df.corr(numeric_only=True)
    plt.figure(figsize=tamanio)
    mascara=np.triu(np.ones_like(matriz_corr, dtype=np.bool_))
    sns.heatmap(matriz_corr, annot=True, vmin=-1, vmax=1, mask=mascara)
    plt.tight_layout()



#OUTLIERS


def detectar_outliers(df, tamanio=(15,8), color="orange"):
    df_numericas=f.separar_dataframe(df)[0]
    num_filas=math.ceil(len(df_numericas.columns)/2)
    fig, axes = plt.subplots(nrows=num_filas,ncols=2, figsize=tamanio)
    axes= axes.flat
    for indice, col in enumerate(df_numericas.columns):
        sns.boxplot(x= col, data= df_numericas, ax=axes[indice], color=color)
        axes[indice].set_title(f"Outliers de {col}")
        axes[indice].set_xlabel("")
    if len(df_numericas.columns) %2 != 0:
        fig.delaxes(axes[-1])
    else:
        pass
    

    plt.tight_layout()




def identificar_outliers_iqr(df, k=1.5):
    df_num=df.select_dtypes(include=np.number)
    dicc_outliers={}
    for columna in df_num.columns:
        Q1, Q3= np.nanpercentile(df[columna], (25, 75))
        iqr= Q3 - Q1
        limite_superior= Q3 + (iqr * k)
        limite_inferior= Q1 - (iqr * k)

        condicion_sup= df[columna] > limite_superior
        condicion_inf= df[columna] < limite_inferior
        df_outliers= df[condicion_inf | condicion_sup]
        print(f"La columna {columna.upper()} tiene {df_outliers.shape[0]} outliers")
        if not df_outliers.empty:
            dicc_outliers[columna]= df_outliers

    return dicc_outliers    



def plot_outliers_univariados(df, tipo_grafica, tamanio,bins=20, whis=1.5):
    df_num=df.select_dtypes(include=np.number)

    fig, axes = plt.subplots(nrows= math.ceil(len(df_num.columns)/ 2), ncols=2, figsize= tamanio)
    axes=axes.flat

    for indice, columna in enumerate(df_num.columns):
        if tipo_grafica.lower() =="h":
            sns.histplot(x= columna, data= df, ax= axes[indice], bins= bins)
        elif tipo_grafica.lower() == "b":
            sns.boxplot(x= columna, data= df, ax= axes[indice], whis=whis, flierprops={"markersize" : 4, "markerfacecolor": "red"})
        else:
            print("No has elegido una grafica correcta elige h o b")
        axes[indice].set_title(f"Distribución columna {columna}")
        axes[indice].set_xlabel("")

    if len(df_num.columns) % 2 !=0:
        fig.delaxes(axes[-1])

    plt.tight_layout()



def identificar_outliers_zscore(df, limite_desviaciones =3):
    df_num=df.select_dtypes(include=np.number)
    dicc_outliers={}
    for columna in df_num.columns:
        condicion_zscore= abs(zscore(df[columna])) >= limite_desviaciones
        df_outliers= df[condicion_zscore]
        print(f"La cantidad de outliers para la {columna.upper()} es de {df_outliers.shape[0]} outliers")
        if not df_outliers.empty:
            dicc_outliers[columna]= df_outliers
    return dicc_outliers
