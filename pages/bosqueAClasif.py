import dash
from dash import callback, html, dcc, Input, Output, State, ALL
import plotly.express as px
import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np
from funciones import generate_table

from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

from sklearn.ensemble import RandomForestClassifier

dash.register_page(__name__, name='Bosque Aleatorio de Clasificacion', order=5)

layout = dbc.Container(children=[], id='bac-layout')

@callback(
    Output('bac-layout', 'children'),
    Input('main-data', 'data')
)
def comprobar_data(data):
    if data:
        df = pd.DataFrame(data)
        return desplegar_layout(df)
    return dbc.Container([
            dbc.Alert("¡Suba un archivo válido en Home!", color="danger")])


def desplegar_layout(df):
    # Limpiando datos
    MDatos = df.dropna()
    # MDatos = MDatos.drop(columns = columnasObj)
    children=[
        html.H1(children=u'Bosque Aleatorio (Clasificación)'),
        informacion_datos(df),
        descripcion_datos(df),
        seleccion(MDatos),
        predictor_element(MDatos)
    ]
    return children


def informacion_datos(df):
    return html.Div(children=[
        html.H2("Tipos de datos"),
        generate_table(df.dtypes.to_frame().T)
    ])

def descripcion_datos(df):
    return html.Div(children=[
        html.H2("Descripción de los datos"),
        generate_table(df.describe().reset_index())
    ])


def seleccion(MDatos):
    #Seleccionando las columas que no sus valores unicos no sean mayores a 10 elementos
    columnasOpcion = [columna for columna in MDatos if MDatos[columna].unique().size < 10] 

    return html.Div(children=[
        html.H2("Selección de varible a predecir X"),
        dcc.RadioItems(columnasOpcion, id='var-selector-2'),
        html.Button(id='submit-button-state-2', n_clicks=0, children='Entrenar', className='btn btn-info'),
        html.Div(),
        #dcc.Textarea(id='texto-2', readOnly=True, style={'width': '100%', 'height': 200}),
        #dcc.Clipboard(target_id="texto-2"),
        html.Div([
            dbc.Alert("¡Bosque Aleatorio de clasificación listo!", color='success')    
        ], id='ready-alert-2', hidden=True)
        
    ])

def predictor_element(MDatos):
    return html.Div(
    html.Div(children=[
            html.H4('Predicción'),
            html.Div(id="resultado-pred-2"),
            html.Div([
                        dcc.Input(
                            id={
                                'type' : 'dynamic-input-2',
                                'index' : _ },
                            type="number",
                            placeholder="Inserte valor {}".format(_),
                            ) for _ in range(len(MDatos.columns)-1)
                    ], id='predictor-div'),
            html.Button(id='boton-predictor-2', n_clicks=0, children='Predecir', className='btn btn-info')
        ] 
    ), id='armani-2', hidden=True)




#### Callbacks ##################################
@callback(
    Output('ready-alert-2','hidden'),
    Output('armani-2', 'hidden'),
    Input('submit-button-state-2','n_clicks'),
    State('var-selector-2', 'value'),
    State('main-data', 'data')
    )
def update_output(n_clicks, variable, data):
    
    if data:
        df = pd.DataFrame(data)
        if variable and n_clicks:
            # Aplicandp el algoritmo
            MDatos = df.dropna()

            ClasificacionBA = RandomForestClassifier(max_depth=3, random_state=0)

            # Aplicacion del algoritmo
            colX = MDatos.columns.to_list()
            colX.remove(variable)
            X = np.array(MDatos[colX])
            Y = np.array(MDatos[variable])
            X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, 
                                                                        test_size = 0.2, 
                                                                        random_state = 0, 
                                                                        shuffle = True)
            ClasificacionBA.fit(X_train, Y_train)
            #Reporte = export_text(ClasificacionBA, feature_names = colX),
            
            #print(type(Reporte))
            
            #return Reporte[0], False
            return False, False
    return dash.no_update, dash.no_update


# Usando pattern-matching callbacks
@callback(
    Output('resultado-pred-2', 'children'),
    Input('boton-predictor-2', 'n_clicks'),
    State('var-selector-2', 'value'),
    State('main-data', 'data'),
    State({'type': 'dynamic-input-2', 'index': ALL}, 'value')
)
def prediccion(n_clicks, selection, data, values):
    print('Pattern-marching',values)
    if n_clicks:
        if None in values:
            return "Valores no validos"
        if data:
            df = pd.DataFrame(data)
            #Limpiando data
            #columnasObj = df.select_dtypes(include=['object']).columns
            #MDatos = df.drop(columns = columnasObj)
            MDatos = df.dropna()
            ClasificacionBA = RandomForestClassifier(max_depth=3, random_state=0)
            # Aplicacion del algoritmo
            colX = MDatos.columns.to_list()
            colX.remove(selection)
            X = np.array(MDatos[colX])
            Y = np.array(MDatos[selection])
            X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, 
                                                                        test_size = 0.2, 
                                                                        random_state = 0, 
                                                                        shuffle = True)
            ClasificacionBA.fit(X_train, Y_train)
            return ClasificacionBA.predict([values])
    return dash.no_update
