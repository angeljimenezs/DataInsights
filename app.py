import dash 
from dash import html, dcc, Input, Output, State
import dash_bootstrap_components as dbc

# Instanciamos dash con paginas multiples y temas de bootstrap
app = dash.Dash(__name__, use_pages=True, external_stylesheets=[dbc.themes.SPACELAB], meta_tags=[
            {"name": "viewport", "content": "width=device-width, initial-scale=1"}
            ])
"""
######## Nav bar comeinzo
PLOTLY_LOGO = "https://images.plot.ly/logo/new-branding/plotly-logomark.png"
search_bar = dbc.Row(
    [
        dbc.Col(dbc.Input(type="search", placeholder="Search")),
        dbc.Col(
            dbc.Button(
                "Search", color="primary", className="ms-2", n_clicks=0
            ),
            width="auto",
        ),
    ],
    className="g-0 ms-auto flex-nowrap mt-3 mt-md-0",
    align="center",
)

navbar = dbc.Navbar(
    dbc.Container(
        [
            html.A(
                # Use row and col to control vertical alignment of logo / brand
                dbc.Row(
                    [
                        dbc.Col(html.Img(src=PLOTLY_LOGO, height="30px")),
                        dbc.Col(dbc.NavbarBrand("Navbar", className="ms-2")),
                    ],
                    align="center",
                    className="g-0",
                ),
                href="https://plotly.com",
                style={"textDecoration": "none"},
            ),
            dbc.NavbarToggler(id="navbar-toggler", n_clicks=0),
            dbc.Collapse(
                search_bar,
                id="navbar-collapse",
                is_open=False,
                navbar=True,
            ),
        ]
    ),
    color="dark",
    dark=True,
)


# add callback for toggling the collapse on small screens
@app.callback(
    Output("navbar-collapse", "is_open"),
    [Input("navbar-toggler", "n_clicks")],
    [State("navbar-collapse", "is_open")],
)
def toggle_navbar_collapse(n, is_open):
    if n:
        return not is_open
    return is_open

##### Navbar final
"""

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