import dash 
import dash_bootstrap_components as dbc
from dash import html, dcc
app = dash.Dash(
    external_stylesheets=[dbc.themes.BOOTSTRAP],
        meta_tags=[
            {"name": "viewport", "content": "width=device-width, initial-scale=1"}
            ]
)

app.layout = html.Div([dbc.Container(children=[
        html.Div([
            html.H1('Hola Bootstrap', className='text-center', style={"color":"white"})
        ], style={"background-color": "blue" }, className='rounded'),

        html.P("""Hola a todos sksdfklsdfdfdfldsA As dfdfdfd\n SDsdsds Asasdsdf Perefeddffdgbhnhf""")
    ])
], style={'background-color':'gray'})

if __name__ == "__main__":
    app.run_server(debug=True)