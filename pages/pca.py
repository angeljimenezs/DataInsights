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
dash.register_page(__name__, name='Análisis de Componentes Principales PCA', order=1)


layout = dbc.Container(children=[], id='pca-layout')

@callback(
    Output('pca-layout', 'children'),
    Input('datitos', 'data')
)
def comprobar_data(data):
    if data:
        df = pd.DataFrame(data)
        return desplegar_layout(df)
        #return dash_table.DataTable(df.to_dict('records'), [{"name": i, "id": i} for i in df.columns])
    return 'No hay datos'


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
        html.H2("Matriz de Correlacion"),
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
    html.H2("Varianza acumulada"),
    dcc.Graph(figure=graficar_varianza(Varianza, np.cumsum(Varianza))),
    dcc.Slider(0,len(Varianza), step=1,
            value=0,
            marks={str(num): str(num) for num in range(len(Varianza)+1)},
            id='varianza-slider'),
    html.Div(id='slider-output', children=[]),
])

@callback(
    Output('slider-output', 'children'),
    State('datitos', 'data'), 
    Input('varianza-slider', 'value')
    )
def update_variables(data, valor):
    if data:
        df = pd.DataFrame(data)
        dfNN = df.dropna()
        columnasObj = dfNN.select_dtypes(include=['object']).columns
        NuevaMatriz = dfNN.drop(columns=columnasObj)    # Se quitan las variables nominales
        return "Componente: {}".format(NuevaMatriz.columns.to_list()[:valor])


"""
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

], className='bg-succes')

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
    html.H1("Análisis de Componentes Principales PCA"),
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

"""    
