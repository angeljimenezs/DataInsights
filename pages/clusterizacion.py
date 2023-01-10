from gc import callbacks
import dash
from dash import callback, html, dcc, Input, Output, dash_table, State
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

dash.register_page(__name__, name='Clustering', order=4)

layout = dbc.Container(children=[], id='kmeans-layout')

@callback(
    Output('kmeans-layout', 'children'),
    Input('datitos', 'data')
)
def comprobar_data(data):
    if data:
        df = pd.DataFrame(data)
        return desplegar_layout(df)
    return dbc.Container([
            dbc.Alert("¡Suba un archivo válido en Home!", color="danger")])


def desplegar_layout(df):
    children=[
        dcc.Store(id='kmeans-data', data=[], storage_type='memory'),
        html.H1('Clusterización por K-means'),
        matriz_correlacion(df),
        codo_element(df),
        kmeans_deslizador(df)
        ]
    return children


def matriz_correlacion(df):
    return html.Div(children=[
    html.H2("Matriz de Correlación"),
    dcc.Graph(figure = hacer_matriz_correlacion(df))
])


def codo_element(df):
    # Procesamiento
    dfNN = df.dropna()
    columnasObj = dfNN.select_dtypes(include=['object']).columns
    dfNN = dfNN.drop(columns=columnasObj)

    estandarizar = StandardScaler()                         # Se instancia el objeto StandardScaler o MinMaxScaler 
    MEstandarizada = estandarizar.fit_transform(dfNN)   # Se calculan la media y desviación y se escalan los datos

    SSE = []
    for i in range(2, 10):
        km = KMeans(n_clusters=i, random_state=0, n_init='auto')
        km.fit(MEstandarizada)
        SSE.append(km.inertia_)
    kl = KneeLocator(range(2,10), SSE, curve="convex", direction="decreasing")
    
    return html.Div(children=[
    html.H2("Cálculo de codo"),
    html.H6('Valor óptimo de k = {}'.format(kl.elbow)),
    dcc.Graph(figure=hacer_grafica_elbow(SSE, kl))
])


def kmeans_deslizador(df=None):
    return html.Div(children=[
    html.H2("Número de clústers k"),
    dcc.Slider(2,9,step=1,value=2, id='kmeans-slider'),
    html.Div(id='kmeans-clusters'),
    html.Div(id='kmeans-centroids'),
    dcc.Graph(id='kmeans-3d'),
    #---- Download---------------------------
    html.Button("Download CSV", id="btn_csv", className='btn btn-info'),
    dcc.Download(id='download-dataframe-csv'),
    #--------------------------------------
    html.Div(id='kmeans-tabla', children=[]),

])


def hacer_grafica_elbow(SSE, kl):
    fig = px.line(x=range(2,10), y=SSE, markers=True)
    print('Codo en: ',str(kl.elbow))
    fig.add_vline(x=kl.elbow, line_width=3, line_dash="dash", line_color="green")
    # Cambiar el color del grid zero
    fig = formato_grafica(fig)
    
    return fig

#-------------------- Callbacks ------------------------

@callback(
    Output('kmeans-data', 'data'),
    Input('kmeans-slider', 'value'),
    State('datitos', 'data'),
    #State('std-data', 'data')
)
def update_kmeans_data(slider_valor, data):
    #print(type(slider_valor), type(data))
    if data:
        df = pd.DataFrame(data)
        # Procesamiento
        dfNN = df.dropna()
        columnasObj = dfNN.select_dtypes(include=['object']).columns
        dfNN = dfNN.drop(columns=columnasObj)
        estandarizar = StandardScaler()                         # Se instancia el objeto StandardScaler o MinMaxScaler 
        MEstandarizada = estandarizar.fit_transform(dfNN)   # Se calculan la media y desviación y se escalan los datos

        #MEstandarizada = pd.DataFrame(stdData)
        MParticional = KMeans(n_clusters=slider_valor, random_state=0, n_init='auto').fit(MEstandarizada)
        MParticional.predict(MEstandarizada)
        MParticional.labels_
        dfCopia = df.copy()
        dfCopia['clusterP'] = MParticional.labels_
        return dfCopia.to_dict('records')
    return None


@callback(
    Output('kmeans-tabla', 'children'),
    Input('kmeans-data', 'data'),
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


#--------- Download callback
@callback(
    Output("download-dataframe-csv", "data"),
    Input("btn_csv", "n_clicks"),
    State('kmeans-data', 'data'),
    prevent_initial_call=True,
)
def func(n_clicks, data):
    dff = pd.DataFrame(data)
    return dcc.send_data_frame(dff.to_csv, "clusteredData.csv")




'''
df = pd.read_csv("../iris.csv")
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
'''