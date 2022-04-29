from plot import save

import json
import re

import plotly.express

# technique for salvaging data from html


def read_from_html():
    name = "version310"
    with open(f"{name}.html") as f:
        html = f.read()
    call_arg_str = re.findall(r'Plotly\.newPlot\((.*)\)', html[0:])[1]
    call_args = json.loads(f'[{call_arg_str}]')
    plotly_json = {'data': call_args[1], 'layout': call_args[2]}
    fig = plotly.io.from_json(json.dumps(plotly_json))

    # make linear
    fig.update_xaxes(
        type='linear',
        tickfont=dict(size=24)
    )
    fig.update_yaxes(
        type='linear',
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

    save(fig, name, w=1080, h=920, dirn='plots')


if __name__ == '__main__':
    read_from_html()
