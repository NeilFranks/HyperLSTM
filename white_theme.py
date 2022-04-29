import plotly.graph_objects as go
import plotly.io as pio

plotly_template = pio.templates["plotly_white"]

pio.templates["plotly_white_custom"] = pio.templates["plotly_white"]
pio.templates["plotly_white_custom"].update({
    'layout' : {
        'paper_bgcolor': 'white',
        'plot_bgcolor': 'white'
    }})
pio.templates.default = "plotly_white_custom"
