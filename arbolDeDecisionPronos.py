from dash import Dash, html, dcc, Input, Output, State
import plotly.express as px
import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np

from sklearn import model_selection
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.tree import export_text

app = Dash(__name__)
df = pd.read_csv("iris.csv")

#---------------------------1001173946--------------------------
columnasObj = df.select_dtypes(include=['object']).columns
columnasObj

MDatos = df.drop(columns = columnasObj)
MDatos = MDatos.dropna()

# --------------------------------------------------------
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
    ])

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
    dcc.Checklist(MDatos.columns, []),
    dcc.RadioItems(MDatos.columns, id='var-selector'),
    html.Button(id='submit-button-state', n_clicks=0, children='Submit'),
    html.Div(),
    dcc.Textarea(id='texto', readOnly=True, style={'width': '100%', 'height': 200}),
    dcc.Clipboard(
        target_id="texto")
])


app.layout = html.Div(children=[
    html.H1(children='Arbol de decisión (Pronostico)'),
    informacion_datos,
    descripcion_datos,
    seleccion,
])

@app.callback(
    Output('texto','value'),
    Input('submit-button-state','n_clicks'),
    State('var-selector', 'value'))
def update_output(n_clicks, valor):
    if valor and n_clicks:

        # Aplicacion del algoritmo
        colX = MDatos.columns.to_list()
        colX.remove(valor)
        X = np.array(MDatos[colX])
        Y = np.array(MDatos[valor])
        X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, 
                                                                    test_size = 0.2, 
                                                                    random_state = 0, 
                                                                    shuffle = True)
        PronosticoAD = DecisionTreeRegressor(max_depth=3, random_state=0)
        PronosticoAD.fit(X_train, Y_train)
        Reporte = export_text(PronosticoAD, feature_names = colX)
        return Reporte
        #return "Hola {} y {}".format(valor, colX)

if __name__ == '__main__':
    app.run_server(debug=True)