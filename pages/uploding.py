import dash
from dash import callback, html, dcc, Input, Output, State, dash_table
import plotly.express as px
import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np
from funciones import *


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
        'Drag and Drop or ',
        html.A('Select Files')
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


dash.register_page(__name__)

layout = html.Div([
        upload_element,
        html.Div(id='output-datatable'),
])

@callback(
    Output('datitos', 'data'),
    Input('upload-data', 'contents'),
    State('upload-data', 'filename'),
    State('upload-data', 'last_modified')
)
def store_data(list_of_contents, list_of_names, list_of_dates):
    if list_of_contents is not None:
        dat = dataframe_from_upload(list_of_contents, list_of_names)
        print(dat)
        return dat


"""
callback(
    Output('output-datatable', 'children'),
    Input('upload-data', 'contents'),
    State('upload-data', 'filename'),
    State('upload-data', 'last_modified'))
def update_output(list_of_contents, list_of_names, list_of_dates):
    if list_of_contents is not None:
        children = [ parse_contents(list_of_contents, list_of_names, list_of_dates)]
        return children
"""