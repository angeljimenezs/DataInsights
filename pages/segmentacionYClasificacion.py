from gc import callbacks
import dash
from dash import callback, html, dcc, Input, Output, dash_table
import plotly.express as px
import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np
from funciones import *
# Bibiliotecas para la estandarizacion
from sklearn.preprocessing import StandardScaler, MinMaxScaler  
# Bibliotecas para kmeans
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
from kneed import KneeLocator

dash.register_page(__name__, name='Segmentación y Clasificación', order=3)
df = pd.read_csv("../iris.csv")

#### a

########## Modelo 1 Segementacion
###### Algoritmo K-means

### Limpiar los datos
dfNN = df.dropna()
columnasObj = dfNN.select_dtypes(include=['object']).columns
dfNN = dfNN.drop(columns=columnasObj)

estandarizar = StandardScaler()                         # Se instancia el objeto StandardScaler o MinMaxScaler 
MEstandarizada = estandarizar.fit_transform(dfNN)   # Se calculan la media y desviación y se escalan los datos
pd.DataFrame(MEstandarizada)

SSE = []
for i in range(2, 10):
    km = KMeans(n_clusters=i, random_state=0)
    km.fit(MEstandarizada)
    SSE.append(km.inertia_)


###############################

def hacer_grafica_elbow(SSE):
    fig = px.line(x=range(2,10), y=SSE, markers=True)
    kl = KneeLocator(range(2,10), SSE, curve="convex", direction="decreasing")
    print(kl.elbow)
    fig.add_vline(x=kl.elbow, line_width=3, line_dash="dash", line_color="green")
    return fig



#--------- Elementos del layout
matriz_correlacion = html.Div(children=[
    html.H2("Matriz de Correlacion"),
    dcc.Graph(figure = hacer_matriz_correlacion(df))

])

grafica_codo = html.Div(children=[
    html.H2("Grafica codo"),
    dcc.Graph(figure=hacer_grafica_elbow(SSE))
])

kmeans_deslizador = html.Div(children=[
    html.H2("K-Means Deslizador"),
    dcc.Slider(2,9,step=1,value=2, id='kmeans-slider'),
    html.Div(id='kmeans-tabla', children=[]),
    html.Div(id='kmeans-clusters'),
    html.Div(id='kmeans-centroids'),
    dcc.Graph(id='kmeans-3d'),


])

layout = html.Div(children=[
    dcc.Store(id='kmeans-data', data=[], storage_type='memory'), 
    matriz_correlacion,
    grafica_codo,
    kmeans_deslizador


])

@callback(
    Output('kmeans-data', 'data'),
    Input('kmeans-slider', 'value')
)
def update_kmeans_data(slider_valor):
    MParticional = KMeans(n_clusters=slider_valor, random_state=0).fit(MEstandarizada)
    MParticional.predict(MEstandarizada)
    MParticional.labels_
    dfCopia = df.copy()
    dfCopia['clusterP'] = MParticional.labels_
    return dfCopia.to_dict('records')


@callback(
    Output('kmeans-tabla', 'children'),
    Input('kmeans-data', 'data')
)
def insert_data_table(data):
    #print(data)
    dff = pd.DataFrame(data)
    return generate_table(dff)

@callback(
    Output('kmeans-clusters', 'children'),
    Input('kmeans-data', 'data')
)
def insert_clusters_table(data):
    dff = pd.DataFrame(data)
    return generate_table(dff.groupby(['clusterP'])['clusterP'].count().to_frame().T)
    #return dash_table.DataTable(dff.to_dict('records'), [{"name": i, "id": i} for i in dff.columns])

@callback(
    Output('kmeans-centroids', 'children'),
    Input('kmeans-data', 'data')
)
def insert_centroids_table(data):
    dff = pd.DataFrame(data)
    return generate_table(dff.groupby('clusterP').mean().reset_index())

@callback(
    Output('kmeans-3d', 'figure'),
    Input('kmeans-data', 'data')
)
def insert_3d_graph(data):
    dff = pd.DataFrame(data)
    fig = px.scatter_3d(dff, x=dff.columns[0], y=dff.columns[1], z=dff.columns[2],
              color='clusterP')
    return fig