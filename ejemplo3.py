import dash 
import dash_bootstrap_components as dbc
from dash import html, dcc
app = dash.Dash(
    external_stylesheets=[dbc.themes.BOOTSTRAP],
        meta_tags=[
            {"name": "viewport", "content": "width=device-width, initial-scale=1"}
            ]
)

app.layout = dbc.Container(children=[
    html.H1('Hola Bootstrap', className='text-center')
])

if __name__ == "__main__":
    app.run_server(debug=True)