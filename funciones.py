from dash import html
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

def hacer_matriz_correlacion(data):
    fig = px.imshow(data.corr(numeric_only=True), text_auto=True, color_continuous_scale='RdBu_r', aspect="auto")
    return fig

def generate_table(dataframe, max_rows=10):
    return html.Table([
        html.Thead(
            html.Tr([html.Th(col) for col in dataframe.columns])
        ),
        html.Tbody([
            html.Tr([
                html.Td(str(dataframe.iloc[i][col])) for col in dataframe.columns
            ]) for i in range(min(len(dataframe), max_rows))
        ])
    ], className='table table-hover table-responsive')