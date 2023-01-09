from dash import html
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
   
    layout = dict(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
    # Create traces
    fig = go.Figure(layout=layout)
    fig.add_trace(go.Scatter(x=np.arange(len(varianza)), y=varianza, 
        mode='lines+markers', name='varianza'))
    fig.add_trace(go.Scatter(x=np.arange(len(varianza_acum)), y=varianza_acum, 
        mode='lines+markers', name='varianza_acumulada'))
    
    # Cambiar el color del grid zero
    fig.update_xaxes(zeroline=True, zerolinewidth=2, zerolinecolor='Black')
    fig.update_yaxes(zeroline=True, zerolinewidth=2, zerolinecolor='Black')

    # Cambiar el color del grid
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
    
    return fig
