import math
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def hacer_histo(data):
    columnas = data.select_dtypes(include=np.number).columns.tolist()
    
    fig = make_subplots(rows=math.ceil(len(columnas)/2), cols=2)
    
    par=False
    i=1
    for col in columnas:
        if not par:
            trace = go.Histogram(x=data[col])
            fig.append_trace(trace, i, 1)
            par = True
        else:
            trace = go.Histogram(x=data[col])
            fig.append_trace(trace, i, 2)
            par = False
            i+=1

    return fig