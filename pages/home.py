import dash
from dash import callback, html, dcc, Input, Output, State, dash_table
import plotly.express as px
import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np
from funciones import *


dash.register_page(__name__, name ='Home', path='/', order=0)

def dataframe_from_upload(contents, filename):
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
        return {}
    return df.to_dict('records')

upload_element = dcc.Upload(
    id='upload-data',
    children=html.Div([
        'Arrastre y suelte o ',
        html.A('Seleccione archivo')
    ]),
    style={
    'width': '100%',
    'height': '60px',
    'lineHeight': '60px',
    'borderWidth': '1px',
    'borderStyle': 'dashed',
    'borderRadius': '5px',
    'textAlign': 'center',
    'margin': '10px'
    },
    multiple=False
)


layout = html.Div([
    html.H1('¡Bienvenido a DataInsights!', className='display-4'),
    html.P('DataInsights le permite poder analizar sus propios datos a tráves 5 distintos algoritmos.'),
    html.H5('¡Empezemos! Por favor suba un archivo en formato .csv o .xls', className='text-center'),
    upload_element,
    html.Div(id='output-datatable'),
    html.Div(id='uploading-output')
])

@callback(
    Output('main-data', 'data'),
    Input('upload-data', 'contents'),
    State('upload-data', 'filename'),
)
def store_data(list_of_contents, list_of_names):
    if list_of_contents is not None:
        dat = dataframe_from_upload(list_of_contents, list_of_names)
        #print(type(dat))
        return dat

@callback(
    Output('uploading-output', 'children'),
    Input('main-data', 'data'),
    State('upload-data', 'filename'),
)
def update_uploading_output(data, filename):
    print(type(data))
    print(filename)
    if data is not None:
        return html.Div([
            dbc.Alert("¡Archivo subido!", color="success"),
            html.H5(filename)])
    
