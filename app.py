# -*- coding: utf-8 -*-
# Run this app with `python app.py` and
# visit http://127.0.0.1:8050/ in your web browser.
##: This fixes the dash import issue
"""
Simple module that monkey patches pkg_resources.get_distribution used by dash
to determine the version of Flask-Compress which is not available with a
flask_compress.__version__ attribute. Known to work with dash==1.16.3 and
PyInstaller==3.6.
"""

import sys
from collections import namedtuple

import pkg_resources

IS_FROZEN = hasattr(sys, '_MEIPASS')

# backup true function
_true_get_distribution = pkg_resources.get_distribution
# create small placeholder for the dash call
# _flask_compress_version = parse_version(get_distribution("flask-compress").version)
_Dist = namedtuple('_Dist', ['version'])

def _get_distribution(dist):
    if IS_FROZEN and dist == 'flask-compress':
        return _Dist('1.5.0')
    else:
        return _true_get_distribution(dist)

# monkey patch the function so it can work once frozen and pkg_resources is of
# no help
pkg_resources.get_distribution = _get_distribution
##: End fix dash import issue
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Output, Input
import pandas as pd
#import plotly.express as px
from threading import Thread
import plotly
import plotly.graph_objs as go
from db import DataBase
from transcript_db import TranscriptDataBase
import base64
from pandas import DataFrame
import sqlite3
from sqlite3 import Error
import AudioIntensity
from AudioIntensity import tool
import numpy as np
from plotly import subplots
from plotly.subplots import make_subplots
import os
from AudioIntensity import tool

fig = go.Figure()
emotions = ['calm','happy','sad', 'angry', 'fearful', 'disgust', 'surprised']

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
#database connection
app = dash.Dash(__name__)


DB = DataBase()
t_DB = TranscriptDataBase()

app.layout = html.Div(
    [
        html.Div(
            html.H1('LISA')
        ),
        html.Br(),
        dcc.Tabs(id='score-tabs',value='emotion-score',children=[
            dcc.Tab(label='Emotional Intensity',value='emotion-score',children=[
                #SCORE GRAPH
                html.Div(id='emotion-graph',
                    style={
                        'width':'49%',
                        'display':'inline-block'
                    }),
                #TRANSCRIPTS
                html.Div(id='trans-block',
                    style={
                        'width':'49%',
                        'display':'inline-block',
                        'overflowY':'scroll',
                        'height':500
                    }),
                dcc.Checklist(id='emotion-list',
                    options=[
                        {'label': ' Calm ','value':'calm'},
                        {'label': ' Happy ','value':'happy'},
                        {'label': ' Angry ','value':'angry'},
                        {'label': ' Fearful ','value':'fearful'},
                        {'label': ' Disgust ','value':'disgust'},
                        {'label': ' Surprised ','value':'surprised'}
                    ],
                    value=['angry', 'fearful', 'disgust', 'surprised', 'happy', 'calm']
                ),
                dcc.RadioItems(id='plotPref',
                    options=[
                        {'label':'scatter plot', 'value':'markers'},
                        {'label':'lines', 'value':'lines'},
                        {'label':'both', 'value':'markers+lines'}
                    ],
                    value='markers+lines'
                )
            ]),
            dcc.Tab(label='Dramatic Tension', value = 'dt-score',children=[
                html.Div(id='dt-graph')
            ]),
            dcc.Tab(label='Word Buckets', value = 'word-count',children=[
                html.Div(id='buckets')
            ])
        ]),

        dcc.Interval(
            id='graph-update',
            interval=1000,
            n_intervals=0
        )


    ]
)

@app.callback(
    Output('emotion-graph', 'children'),
    [Input('score-tabs','value'),Input('emotion-list','value'),Input('graph-update','n_intervals'),Input('plotPref','value')]
)

def update_emotion_scatter(tab,list,n,plotPref):
    DB.sql_connection('mydatabase.db')
    df = DB.get_dataframe()
    fig.data = []
    for col in list:
        fig.add_trace(go.Scatter(
            name=col,
            x=df.index,
            y=df[col].values,
            mode=plotPref
        ))

    fig.update_layout(
        title='Emotional Intensity',
        yaxis_title='Score',
        xaxis={'autorange':True,'title':'Speech Interval (5 seconds)'},
        autosize=True
    )

    if (tab == 'emotion-score'):
        return dcc.Graph(id='emotion-intensity', figure=fig)
    if(tab== 'dt-score'):
        return dcc.Graph(id='dramatic-tension',figure={})


#TRANSCRIPT
@app.callback(
     Output('trans-block','children'),
     Input('graph-update','n_intervals')
)
def update_transcript(n):
    t_DB.sql_connection('transcript_datebase.db')

    trans = t_DB.get_dataframe()

    result = []

    i = 0
    length = len(trans['speaker']) - 1
    while i < length:
        #print(i)
        temp = "Speaker " + str(trans['speaker'][i]) + ": " + str(trans['word'][i]) + " "
        while (trans['speaker'][i] == trans['speaker'][i + 1] and i < length - 2):
            temp += str(trans['word'][i+1]) + " "
            i += 1
        i += 1
        #print(temp)
        result.append(html.P(temp))


    result = tuple(result)

    return html.Blockquote(id='transcript',cite='Speaker',children=result)




if __name__ == '__main__':
    Tool = tool('placeholder')
    Thread(target = app.run_server, kwargs={'debug': True}).start()
    Thread(target = Tool.start_recording).start()
    # app.run_server(debug=True)
