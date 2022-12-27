import math
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

def hacer_matriz_correlacion(data):
    fig = px.imshow(data.corr(numeric_only=True), text_auto=True, color_continuous_scale='RdBu_r', aspect="auto")
    return fig