from plotly.subplots import make_subplots
import plotly.express as px
import plotly.graph_objects as go
import plotly

import pandas as pd
import numpy  as  np

from subprocess import call
from pathlib import Path
from pprint import pprint
import sys

import white_theme

QUAL_COLORS = plotly.colors.qualitative.G10 + plotly.colors.qualitative.Dark24 + plotly.colors.qualitative.Light24

def heatmap(*compare, labels=None, title=None):
    fig = make_subplots(rows=1, cols=len(compare), subplot_titles=labels)
    for i, c in enumerate(compare):
        fig.add_trace(go.Heatmap(z=c, zmin=0, zmax=1), row=1, col=i + 1)
    fig.update_layout(title_text=title)
    save(fig, f'{title.replace(" ", "_")}_heatmap', w=1080, h=920)
    return fig

def save(fig, name, w=1080, h=920, dirn='plots'):
    fig.show()
    fig.write_image(f'{dirn}/{name}.svg', width=w, height=h)
    fig.write_image(f'{dirn}/{name}.png', width=w, height=h)
    fig.write_html(f'{dirn}/{name}.html')
    fig.write_json(f'{dirn}/{name}.json')
    call(f'rsvg-convert -f pdf -o {dirn}/{name}.pdf {dirn}/{name}.svg', shell=True)

def loss(path, y='loss', yaxis_title=None,                           figure_title='compare_loss', yrange=None, dirn='plots'):
    df = pd.read_csv(path)
    fig = px.line(df, y=y, color_discrete_sequence=QUAL_COLORS)
    fig.update_traces(line=dict(width=3))
    if yaxis_title is None:
        yaxis_title = y.replace('_', ' ').title()
    fig.update_layout(font_size=32, xaxis_title=f'Steps', yaxis_title=yaxis_title)
    if yrange is not None:
        fig.update_yaxes(range=yrange)
    fig.update_xaxes(type='log', tickfont=dict(size=24))
    fig.update_yaxes(type='log', tickfont=dict(size=24))
    fig.update_layout(legend=dict(yanchor='top', y=1.1,
                                  xanchor='center', x=0.5,
                                  orientation='h',
                                  font_size=24),
                      legend_title_text='Models')
    save(fig, figure_title, w=1080, h=920, dirn=dirn)
    return df, fig

def get_latest(parent, name='metrics.csv'):
    ''' Courtesy of SO: https://stackoverflow.com/questions/39327032/how-to-get-the-latest-file-in-a-folder'''
    parent = Path(parent)
    paths  = parent.glob('*')
    return max(paths, key=lambda p: p.stat().st_ctime) / name

def main(*args):
    latest = get_latest('csv_data/hockey/', name='metrics.csv')

    dirn = 'plots'
    Path(dirn).mkdir(exist_ok=True)
    loss(latest, y='train_loss', yaxis_title='Training Loss', dirn=dirn)

if __name__ == '__main__':
    main(sys.argv[1:])