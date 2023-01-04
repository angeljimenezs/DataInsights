import dash
from dash import callback, html, dcc, Input, Output, dash_table
import plotly.express as px
import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np
from funciones import *

#app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
dash.register_page(__name__, path='/', order=0)
df = pd.read_csv("../iris.csv")
 

descripcion_datos = html.Div([
    html.H2("Paso 1: Descripción de la estructura de los datos"),
    html.H3("1) Forma (dimensiones) del DataFrame"),
    html.P(str(df.shape)),
    html.H3("2) Tipos de datos (variables)"),
     ## Aqui deberia de haber uuna tabla con los tipos de datos
    #dash_table.DataTable(df.dtypes.to_frame().T.to_dict('records'), 
    #[{"name": i, "id": i} for i in df.columns]),

])

datos_faltantes = html.Div([
    html.H2("Paso 2: Identificación de datos faltantes"),
    ## Aqui deberia de haber uuna tabla con los valores faltante
    generate_table(df.isnull().sum().to_frame().T),
])


valores_atipicos =html.Div([
    html.H2("Paso 3: Detección de valores atípicos"),
    html.H3("Histograma de valores"),
    html.Div([
        dcc.Dropdown(
            df.columns,
            df.columns[0],
            id='histogram-column'
        ),
        dcc.Graph(id='histogram-graph'),

    ]),

    html.H3("Resumen estadístico de variables numéricas"),
    #Aqui va una tabla de los valores
    generate_table(df.describe().reset_index()),

    html.H3("Diagramas de caja para detectar posibles valores atípicos"),
    # Diagrama de caja
    html.Div([
        dcc.Dropdown(
            df.columns,
            df.columns[0],
            id='boxplot-column-x'
        ),
        dcc.Dropdown(
            df.columns,
            df.columns[1],
            id='boxplot-column-y'
        ),
        dcc.Graph(id='boxplot-graph')
    ]),

])

layout = dbc.Container(children=[
    html.H1(children='Analisis explotario de datos'),
    descripcion_datos,
    datos_faltantes,
    valores_atipicos,

    html.H2("Paso 4: Identificación de relaciones entre pares variables"),
    html.H3("Matriz de correlación"),
    # Aqui va un heatmap de correlacion
    dcc.Graph(figure = hacer_matriz_correlacion(df) )
])


@callback(
    Output('boxplot-graph', 'figure'),
    Input('boxplot-column-x', 'value'))
def actualizar_boxplot(boxplot_column_x):
    fig = px.box(df, y = boxplot_column_x)
    return fig

@callback(
    Output('histogram-graph', 'figure'),
    Input('histogram-column', 'value'))
def actualizar_histograma(histogram_column):
    fig = px.histogram(df[[histogram_column]])
    return fig
