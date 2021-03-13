import pandas as pd
import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.express as px
import numpy as np

scorecard_df = pd.read_csv("scorecard.csv")
rf_fpr = np.loadtxt("ROC_rf_fpr")
rf_tpr = np.loadtxt("ROC_rf_tpr")
with open("ROC_rf_auc.txt", mode='r') as f:
    rf_auc = float(f.read())

# Choose dashboard format
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
# Define figures
fig = px.bar(scorecard_df, x="Prediction Percentile", y="Fraud Volume")
fig2 = px.area(x=rf_fpr, y=rf_tpr, title=f'ROC Curve (AUC={rf_auc:.4f})',
               labels=dict(x='False Positive Rate', y='True Positive Rate'))

app.layout = html.Div(children=[
    html.H1(children='Dashboard'),

    html.Div(children='''
        Dash: A web application framework for Python.
    '''),

    dcc.Graph(
        id='RF_Frauds_Detected',
        figure=fig
    ),

    dcc.Graph(
        id='RF_ROC',
        figure=fig2
    ),

])


if __name__ == '__main__':
    app.run_server(debug=True)
