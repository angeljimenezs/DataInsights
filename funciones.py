import base64
import datetime
import io

from dash import html, dcc, dash_table
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from kneed import KneeLocator


def hacer_matriz_correlacion(data):
    fig = px.imshow(data.corr(numeric_only=True), text_auto=True, color_continuous_scale='RdBu_r', aspect="auto")
    return fig

def generate_table(dataframe, max_rows=10):
    return html.Div([ 
    html.Table([
            html.Thead(
                html.Tr([html.Th(col) for col in dataframe.columns])
            ),
            html.Tbody([
                html.Tr([
                    html.Td(str(dataframe.iloc[i][col])) for col in dataframe.columns
                ]) for i in range(min(len(dataframe), max_rows))
            ])
        ], className='table table-hover')
    ], className='table-responsive')

def graficar_varianza(varianza, varianza_acum):
    # Create traces
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=np.arange(1, len(varianza)+1), y=varianza, 
        mode='lines+markers', name='varianza'))
    fig.add_trace(go.Scatter(x=np.arange(1, len(varianza_acum)+1), y=varianza_acum, 
        mode='lines+markers', name='varianza_acumulada'))
    
    fig.update_layout(
        xaxis_title="No de variables", yaxis_title="Porcentaje de importancia"
    )

    fig = formato_grafica(fig)

    return fig

#### Dar formato a grafica
def formato_grafica(fig):
     # Cambiando el fondo
    fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
    # Cambiar el color del grid zero
    fig.update_xaxes(zeroline=True, zerolinewidth=2, zerolinecolor='Black')
    fig.update_yaxes(zeroline=True, zerolinewidth=2, zerolinecolor='Black')
    
    # Cambiar el color del grid
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
    
    return fig
    
############ Funcion de upload
def parse_contents(contents, filename, date):
    content_type, content_string = contents.split(',')

    decoded = base64.b64decode(content_string)
    try:
        if 'csv' in filename:
            # Assume that the user uploaded a CSV file
            df = pd.read_csv(
                io.StringIO(decoded.decode('utf-8')))
        elif 'xls' in filename:
            # Assume that the user uploaded an excel file
            df = pd.read_excel(io.BytesIO(decoded))
    except Exception as e:
        print(e)
        return html.Div([
            'There was an error processing this file.'
        ])

    return html.Div([
        html.H5(filename),
        html.H6(datetime.datetime.fromtimestamp(date)),
        html.Button(id="submit-button", children="Create Graph"),
        html.Hr(),
        dcc.Store(id='stored-data', data=df.to_dict('records')),

        html.Hr(),  # horizontal line

        # For debugging, display the raw contents provided by the web browser
        html.Div('Raw Content'),
        html.Pre(contents[0:200] + '...', style={
            'whiteSpace': 'pre-wrap',
            'wordBreak': 'break-all'
        })
    ])
    
