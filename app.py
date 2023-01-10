import base64
import datetime
import io

import dash 
from dash import html, dcc, Input, Output, State
import dash_bootstrap_components as dbc
from funciones import parse_contents

# Instanciamos dash con paginas multiples y temas de bootstrap
app = dash.Dash(__name__, use_pages=True, external_stylesheets=[dbc.themes.SPACELAB], meta_tags=[
            {"name": "viewport", "content": "width=device-width, initial-scale=1"}
            ])


SIDEBAR_STYLE = {
    "position": "fixed",
    "top": 0,
    "left": 0,
    "bottom": 0,
    "width": "16rem",
    "padding": "2rem 1rem",
    "background-color": "#000000",
}



# Barra lateral
sidebar =html.Div(
    [
        html.H2("DataInsights", style={'color': 'black'}),
        html.Hr(),
        dbc.Nav(
                [
                    dbc.NavLink(
                        [
                            html.Div(page["name"], className="ms-2"),
                        ],
                        href=page["path"],
                        active="exact",
                    )
                    for page in dash.page_registry.values()
                ],
                vertical=True,
                pills=True,
                
        )

    ],
    #className='position-fixed'
    #style={'position':'fixed'} 
    #style=SIDEBAR_STYLE,
)

# Layout para todas las paginas
app.layout = dbc.Container([
    dcc.Store(id='datitos'),
    dbc.Row(
        [
            dbc.Col(
                [
                    sidebar
                ],
                xs=4, sm=4, md=4, lg=2, xl=2, xxl=2, class_name='sticky-sidebar',
                #class_name='col-3 px-1 bg-dark position-fixed" id="sticky-sidebar', 
                #class_name='col-3 sticky-sidebar',
                style={'position':'sticky'}
                ),

            dbc.Col(
                [
                    dash.page_container
                ],
                xs=8, sm=8, md=8, lg=10, xl=10, xxl=10
                #class_name='col-9'
                )
        ]
    )
], fluid=True)



if __name__ == "__main__":
    app.run(debug=True)