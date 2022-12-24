# Run this app with `python app.py` and
# visit http://127.0.0.1:8050/ in your web browser.

from dash import Dash, html, dcc, Input, Output
import plotly.express as px
import pandas as pd
import numpy as np
import funciones

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css'] 


app = Dash(__name__)

# assume you have a "long-form" data frame
# see https://plotly.com/python/px-arguments/ for more options
df = pd.read_csv("heart_failure.csv")


#fig = px.bar(df, x="Fruit", y="Amount", color="City", barmode="group")
#fig = px.scatter(df, x="sepallength", y="sepalwidth", color="class")


def hacer_matriz_correlacion(data):
    fig = px.imshow(data.corr(), text_auto=True, color_continuous_scale='RdBu_r', aspect="auto")
    return fig


app.layout = html.Div(children=[
    html.H1(children='Analisis explotario de datos'),

    html.H2("Paso 1: Descripción de la estructura de los datos"),
    
    html.H3("1) Forma (dimensiones) del DataFrame"),
    html.P(str(df.shape)),
    html.H3("2) Tipos de datos (variables)"),
    html.P(str(df.dtypes)),

    html.H2("Paso 2: Identificación de datos faltantes"),
    ## Aqui deberia de haber uuna tabla con los valores faltante
    html.P(str(df.isnull().sum())),

    html.H2("Paso 3: Detección de valores atípicos"),
    html.H3("Histograma de valores"),
    # Aqui va una grafica interactiva
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
    html.P(str(df.describe())),
    html.H3("Diagramas de caja para detectar posibles valores atípicos"),
    # Diagrama de caja

    html.H2("Paso 4: Identificación de relaciones entre pares variables"),
    html.H3("Matriz de correlación"),
    # Aqui va un heatmap de correlacion
    dcc.Graph(figure = hacer_matriz_correlacion(df) ),




#    dcc.Graph(
#        id='example-graph',
#        figure=fig
#    ),
    html.H1("Esta es una prueba de H1"),

    html.H1("Histogramas"),
    dcc.Graph(id='histo1', figure=funciones.hacer_histo(df))
])


@app.callback(
    Output('histogram-graph', 'figure'),
    Input('histogram-column', 'value'))
def actualizar_histograma(histogram_column):
    fig = px.histogram(df[[histogram_column]])
    return fig



if __name__ == '__main__':
    app.run_server(debug=True)