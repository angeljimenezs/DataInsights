from dash import Dash, html, dcc, Input, Output, dash_table
import plotly.express as px
import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np
from funciones import *


app = Dash(__name__)

df = pd.read_csv("iris.csv")


def generate_table(dataframe, max_rows=10):
    return html.Table([
        html.Thead(
            html.Tr([html.Th(col) for col in dataframe.columns])
        ),
        html.Tbody([
            html.Tr([
                html.Td(dataframe.iloc[i][col]) for col in dataframe.columns
            ]) for i in range(min(len(dataframe), max_rows))
        ])
    ])


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



app.layout = html.Div(children=[
    html.H1("Analisis de Componentes Principales PCA"),
    matriz_correlacion,
    eigenvectores,
    eigenvalores
    

])

if __name__ == '__main__':
    app.run_server(debug=True)