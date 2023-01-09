import dash
from dash import callback, html, dcc, Input, Output, dash_table
import plotly.express as px
import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from funciones import *


#app = Dash(__name__)
dash.register_page(__name__, name='Analisis de Componentes Principales PCA', order=1)

df = pd.read_csv("../iris.csv")


#########################################
# Se hace la estandarziacion de los datos
dfNN = df.dropna()
columnasObj = dfNN.select_dtypes(include=['object']).columns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler  
Estandarizar = StandardScaler()                                  # Se instancia el objeto StandardScaler o MinMaxScaler 
NuevaMatriz = dfNN.drop(columns=columnasObj)    # Se quitan las variables nominales
MEstandarizada = Estandarizar.fit_transform(NuevaMatriz)         # Se calculan la media y desviación para cada variable, y se escalan los datos
pca = PCA(n_components=None)     # pca=PCA(n_components=None), pca=PCA(.85)
pca.fit(MEstandarizada)          # Se obtiene los componentes
df_eigenvectores = pd.DataFrame(pca.components_)
df_eigenvectores.columns = NuevaMatriz.columns
Varianza = pca.explained_variance_ratio_
df_eigenvalores = pd.DataFrame(Varianza, index=NuevaMatriz.columns).T
df_eigenvalores
print(type(df_eigenvectores), str(df_eigenvectores.shape))
print(np.cumsum(pca.explained_variance_ratio_))


## Funciones

###########################################
matriz_correlacion = html.Div(children=[
    html.H2("Matriz de Correlacion"),
    dcc.Graph(figure = hacer_matriz_correlacion(df))

])

eigenvectores = html.Div(children=[
    html.H2("Eigenvectores"),
    generate_table(df_eigenvectores)
    
])

eigenvalores = html.Div(children=[
    html.H2("Eigenvalores"),
    generate_table(df_eigenvalores)
    
])

varianza_acumulada = html.Div(children=[
    html.H2("Varianza acumulada"),
    dcc.Graph(figure=graficar_varianza(Varianza, np.cumsum(pca.explained_variance_ratio_))),
    dcc.Slider(0,len(Varianza), step=1,
            value=0,
            marks={str(num): str(num) for num in range(len(Varianza)+1)},
            id='varianza-slider'),
    html.Div(id='slider-output', children=[]),

    
])



layout = html.Div(children=[
    html.H1("Analisis de Componentes Principales PCA"),
    matriz_correlacion,
    eigenvectores,
    eigenvalores,
    varianza_acumulada,
    

])


@callback(
    Output('slider-output', 'children'), 
    Input('varianza-slider', 'value')
    )
def update_variables(valor):
    return "Componente: {}".format(NuevaMatriz.columns.to_list()[:valor])
    
