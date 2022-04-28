import time

from turtle import back
from plotly.subplots import make_subplots
import plotly.express as px
import plotly.graph_objects as go
import plotly

import pandas as pd
import numpy as np

from subprocess import call
from pathlib import Path
from pprint import pprint
import sys

import white_theme

QUAL_COLORS = plotly.colors.qualitative.G10 + \
    plotly.colors.qualitative.Dark24 + plotly.colors.qualitative.Light24

DIVERGING = [[0.0,                "rgb(165,0,38)"],
             [0.1111111111111111, "rgb(215,48,39)"],
             [0.2222222222222222, "rgb(244,109,67)"],
             [0.3333333333333333, "rgb(253,174,97)"],
             [0.4444444444444444, "rgb(196, 196, 196)"],
             [0.5555555555555556, "rgb(224,243,248)"],
             [0.6666666666666666, "rgb(171,217,233)"],
             [0.7777777777777778, "rgb(116,173,209)"],
             [0.8888888888888888, "rgb(69,117,180)"],
             [1.0,                "rgb(49,54,149)"]]


def heatmap(*compare, labels=None, title=None, zmin=-1, zmax=1, horizontal=False,
            formatter=None, w=1080, h=1080,
            y_axis='', x_axis=''):
    if horizontal:
        rows = len(compare)
        cols = 1
    else:
        rows = 1
        cols = len(compare)
    fig = make_subplots(rows=rows, cols=cols, subplot_titles=labels)
    for i, c in enumerate(compare):
        if horizontal:
            c = c.T
            row = i + 1
            col = 1
        else:
            row = 1
            col = i + 1
        fig.add_trace(go.Heatmap(z=c, zmin=zmin, zmax=zmax,
                      colorscale=DIVERGING), row=row, col=col)
    fig.update_layout(title_text=title)
    fig.update_layout(xaxis_title=x_axis, yaxis_title=y_axis)
    if formatter is not None:
        formatter(fig)
    title = title.replace('/', '-').replace(' ', '_').replace(',', '-')
    save(fig, f'{title}_heatmap', w=w, h=h)
    return fig


def save(fig, name, w=1080, h=920, dirn='plots'):
    fig.show()
    fig.write_image(f'{dirn}/{name}.svg', width=w, height=h)
    fig.write_image(f'{dirn}/{name}.png', width=w, height=h)
    fig.write_html(f'{dirn}/{name}.html')
    fig.write_json(f'{dirn}/{name}.json')
    call(
        f'rsvg-convert -f pdf -o {dirn}/{name}.pdf {dirn}/{name}.svg', shell=True
    )


def loss(path, non_rolling_y=['loss'], rolling_y=['loss'], rolling_length=50, yaxis_title=None, figure_title='compare_loss', yrange=None, plot_type='log', dirn='plots'):
    df = pd.read_csv(path)

    x = df['step']

    # get rolling averages too :)
    for y_column in rolling_y:
        df[f'{y_column}_rolling'] = df[y_column].rolling(rolling_length).mean()

    y = non_rolling_y
    y.extend([f'{y_column}_rolling' for y_column in rolling_y])

    # fill NaN
    df = df.fillna(method='bfill')

    fig = px.line(df, x=x, y=y, color_discrete_sequence=QUAL_COLORS)
    fig.update_traces(line=dict(width=3))
    if yaxis_title is None:
        yaxis_title = y.replace('_', ' ').title()
    fig.update_layout(font_size=32, xaxis_title=f'Steps',
                      yaxis_title=yaxis_title)
    if yrange is not None:
        fig.update_yaxes(range=yrange)

    fig.update_xaxes(
        type=plot_type,
        tickfont=dict(size=24)
    )
    fig.update_yaxes(
        type=plot_type,
        tickfont=dict(size=24)
    )
    fig.update_layout(
        legend=dict(
            yanchor='top', y=1.1,
            xanchor='center', x=0.5,
            orientation='h',
            font_size=24
        ),
        legend_title_text='Model Performance'
    )
    save(fig, figure_title, w=1080, h=920, dirn=dirn)
    return df, fig


def get_latest(parent, name='metrics.csv'):
    ''' Courtesy of SO: https://stackoverflow.com/questions/39327032/how-to-get-the-latest-file-in-a-folder'''
    parent = Path(parent)
    paths = parent.glob('*')
    return max(paths, key=lambda p: p.stat().st_ctime) / name


def main(*args):
    plot_type = "log"
    if args[0]:
        plot_type = args[0][0]

    latest = get_latest('csv_data/hockey/', name='metrics.csv')

    dirn = 'plots'
    Path(dirn).mkdir(exist_ok=True)
    loss(
        latest,
        non_rolling_y=[
            'val_loss',
            # 'train_loss',
            # 'lr',
            # 'train_accuracy',
            # 'epoch',
            # 'val_accuracy',
        ],
        rolling_y=[
            'train_loss',
            # 'train_accuracy'
        ],
        rolling_length=196,
        yaxis_title='Loss',
        # yaxis_title='Accuracy',
        plot_type=plot_type,
        dirn=dirn,
        figure_title=f"loss {time.time()}"
    )


if __name__ == '__main__':
    main(sys.argv[1:])
