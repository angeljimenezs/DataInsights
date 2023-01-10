import dash
from dash import callback, html, dcc, Input, Output, State
import plotly.express as px
import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np
from funciones import generate_table

from sklearn import model_selection
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.tree import export_text

#app = Dash(__name__)
dash.register_page(__name__, name='Arbol de decision pronostico', order=2)
df = pd.read_csv("../iris.csv")

#---------------------------Data cleaning--------------------------
columnasObj = df.select_dtypes(include=['object']).columns
columnasObj

MDatos = df.drop(columns = columnasObj)
MDatos = MDatos.dropna()
PronosticoAD = DecisionTreeRegressor(max_depth=3, random_state=0)

# --------------------------------------------------------




informacion_datos = html.Div(children=[
    html.H2("Tipos de datos"),
    generate_table(df.dtypes.to_frame().T)

])

descripcion_datos = html.Div(children=[
    html.H2("Descripcion de los datos"),
    generate_table(df.describe().reset_index())
])

seleccion = html.Div(children=[
    html.H2("Seleccion de varible a predecir X"),
    dcc.RadioItems(MDatos.columns, id='var-selector'),
    html.Button(id='submit-button-state', n_clicks=0, children='Submit', className='btn-primary'),
    html.Div(),
    dcc.Textarea(id='texto', readOnly=True, style={'width': '100%', 'height': 200}),
    dcc.Clipboard(target_id="texto"),

   

])

predictor_element = html.Div(
    html.Div(children=
    [
        html.H4('Prediccion'),
        html.Div(id="resultado-pred"),
        html.Div([
                    dcc.Input(
                        id="input_{}".format(_),
                        type="number",
                        placeholder="Inserte valor {}".format(_),
                        ) for _ in range(len(MDatos.columns)-1)
                ], id='predictor-div'),
        html.Button(id='boton-predictor', n_clicks=0, children='Predecir', className='btn-primary')
    ] 
), id='armani', hidden=True)


layout = html.Div(children=[
    dcc.Store(id='memory'),

    html.H1(children=u'Árbol de decisión (Pronóstico)'),
    informacion_datos,
    descripcion_datos,
    seleccion,
    predictor_element
])


@callback(
    Output('texto','value'),
    Output('armani', 'hidden'),
    Input('submit-button-state','n_clicks'),
    State('var-selector', 'value')
    )
def update_output(n_clicks, valor):

    if valor and n_clicks:
        # Haciendo visible Div y Desabilitando al valor seleccionado

        # Aplicacion del algoritmo
        colX = MDatos.columns.to_list()
        colX.remove(valor)
        X = np.array(MDatos[colX])
        Y = np.array(MDatos[valor])
        X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, 
                                                                    test_size = 0.2, 
                                                                    random_state = 0, 
                                                                    shuffle = True)
        PronosticoAD.fit(X_train, Y_train)
        Reporte = export_text(PronosticoAD, feature_names = colX),
        
        print(type(Reporte))
        
        return Reporte[0], False
    return dash.no_update, dash.no_update


@callback(
    Output('resultado-pred', 'children'),
    Input('boton-predictor', 'n_clicks'),
    State('var-selector', 'value'),
    [State('input_{}'.format(_), 'value') for _ in range(len(MDatos.columns)-1)])
def prediccion(n_clicks,  *children):
    if n_clicks:
        print(children)
        print(PronosticoAD)
        if None in children[1:]:
            return "Valores mal insertados"
        return PronosticoAD.predict([children[1:]])
    return ""

'''
def prediccion(n_clicks,  *children):
    if n_clicks:
        lista_a = MDatos.columns.to_list().copy()#
        lista_b = list(children)
        lista_b.pop(0)
        indiceDe = lista_a.index(children[0])
        lista_b.pop(indiceDe)
        print(type(children))
        print(children)
        print(lista_b)
        if None in lista_b:
            return "Valores mal insertados"
        return PronosticoAD.predict([lista_b])
    return ""
'''