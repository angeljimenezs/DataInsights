import dash 
from dash import html, dcc
import dash_bootstrap_components as dbc

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
    "background-color": "#f8f9fa",
}


# Barra lateral
sidebar =html.Div(
    [
        html.Div([
            html.Div([html.Img(src='assets/data.svg', alt='imagen', style={'height':'10%', 'width':'10%', "fill": 'red'})], ),
            html.H2("DataInsights"),
        ]),
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
    style=SIDEBAR_STYLE,
)

# Layout para todas las paginas
app.layout = dbc.Container([
    dbc.Row(
        [
            dbc.Col(
                [
                    sidebar
                ], xs=4, sm=4, md=4, lg=2, xl=2, xxl=2),

            dbc.Col(
                [
                    dash.page_container
                ], xs=8, sm=8, md=8, lg=10, xl=10, xxl=10)
        ]
    )
], fluid=False)


if __name__ == "__main__":
    app.run(debug=True)