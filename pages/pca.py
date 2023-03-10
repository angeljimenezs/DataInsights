import dash
from dash import callback, html, dcc, Input, Output, State, dash_table
import plotly.express as px
import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler  

from funciones import *


#app = Dash(__name__)
dash.register_page(__name__, name='PCA', order=2)


layout = dbc.Container(children=[], id='pca-layout')

@callback(
    Output('pca-layout', 'children'),
    Input('main-data', 'data')
)
def comprobar_data(data):
    if data:
        df = pd.DataFrame(data)
        return desplegar_layout(df)
    return dbc.Container([
            dbc.Alert("¡Suba un archivo válido en Home!", color="danger")])


def desplegar_layout(df):
    dfNN = df.dropna()
    columnasObj = dfNN.select_dtypes(include=['object']).columns
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
    #print(type(df_eigenvectores), str(df_eigenvectores.shape))
    #print(np.cumsum(pca.explained_variance_ratio_))

    children=[
        html.H1("Análisis de Componentes Principales PCA"),
        matriz_correlacion(df),
        eigenvectores(df_eigenvectores),
        eigenvalores(df_eigenvalores),
        varianza_acumulada(Varianza),
    ]
    return children



def matriz_correlacion(df):
    return html.Div(children=[
        html.H2("Matriz de Correlación"),
        dcc.Graph(figure = hacer_matriz_correlacion(df))
    ], 
    className='bg-succes'
)


def eigenvectores(df_eigenvectores):
    return html.Div(children=[
        html.H2("Eigenvectores"),
        generate_table(df_eigenvectores)  
])
def eigenvalores(df_eigenvalores):
    return html.Div(children=[
        html.H2("Eigenvalores"),
        generate_table(df_eigenvalores)      
])

def varianza_acumulada(Varianza):
    return html.Div(children=[
    html.H2("Varianza y selección de características"),
    dcc.Graph(figure=graficar_varianza(Varianza, np.cumsum(Varianza))),
    html.H4('Número de variables'),
    dcc.Slider(0,len(Varianza), step=1,
            value=0,
            marks={str(num): str(num) for num in range(len(Varianza)+1)},
            id='varianza-slider'),
    html.P(id='slider-output', className='fs-5', children=[]),
])

@callback(
    Output('slider-output', 'children'),
    State('main-data', 'data'), 
    Input('varianza-slider', 'value')
    )
def update_variables(data, valor):
    if data:
        df = pd.DataFrame(data)
        dfNN = df.dropna()
        columnasObj = dfNN.select_dtypes(include=['object']).columns
        NuevaMatriz = dfNN.drop(columns=columnasObj)    # Se quitan las variables nominales
        return "Componentes importantes: {}".format(NuevaMatriz.columns.to_list()[:valor])


