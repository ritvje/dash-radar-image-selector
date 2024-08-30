import pandas as pd
from dash import Dash, dcc, html, Input, Output
import plotly.express as px


app = Dash(__name__)

df = pd.read_csv(
    "metars_EFHK_20210501_20211130.csv",
    parse_dates=[
        0,
    ],
)


@app.callback(
    Output("data", "figure"),
    Input("xvar-dropdown", "value"),
    Input("yvar-dropdown", "value"),
    Input("cvar-dropdown", "value"),
)
def update_figure(x_var, y_var, c_var):
    fig = px.scatter(df, x=x_var, y=y_var, color=c_var, hover_name="remarks")
    fig.update_layout(transition_duration=1000)
    return fig


app.layout = html.Div(
    [
        html.Div([html.H1("METAR data", style={"textAlign": "center"})]),
        html.Div(
            [
                html.Div([], style={"width": "10%"}),
                html.Div(
                    [
                        html.Label("X Variable"),
                        html.Br(),
                        dcc.Dropdown(df.columns, "wind_direction", id="xvar-dropdown"),
                    ],
                    style={"width": "20%"},
                ),
                html.Div([], style={"width": "10%"}),
                html.Div(
                    [
                        html.Label("Y Variable"),
                        html.Br(),
                        dcc.Dropdown(df.columns, "wind_speed", id="yvar-dropdown"),
                    ],
                    style={"width": "20%"},
                ),
                html.Div([], style={"width": "10%"}),
                html.Div(
                    [
                        html.Label("Color Variable"),
                        html.Br(),
                        dcc.Dropdown(df.columns, "wind_gust", id="cvar-dropdown"),
                    ],
                    style={"width": "20%"},
                ),
                html.Div([], style={"width": "10%"}),
            ],
            style={"display": "flex", "flex-direction": "horizontal"},
        ),
        dcc.Graph(id="data"),
    ]
)

if __name__ == "__main__":
    app.run_server(debug=True)
