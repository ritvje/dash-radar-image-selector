"""Dash app for selecting & saving masks of areas in radar scans."""
import base64
import io
import json
from datetime import datetime
from pathlib import Path

import arrow
import dash
import dash_bootstrap_components as dbc
import dash_mantine_components as dmc
import h5py
import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pyart
from dash import Input, Output, State, dcc, html
from dash.dependencies import Input, Output
from plotly import express as px
from plotly import graph_objects as go
from scipy import ndimage
from skimage import draw

import config as cfg
from radar_plotting import dash_utils, plot_utils

# Update field names in pyart to access fields not specified in
# original code
# This is probably a really dirty hack, but it works!
pyart.aux_io.odim_h5.ODIM_H5_FIELD_NAMES = plot_utils.PYART_FIELDS


def path_to_indices(path):
    """From SVG path to numpy array of coordinates, each row being a (row, col) point"""
    indices_str = [
        el.replace("M", "").replace("Z", "").split(",") for el in path.split("L")
    ]
    return np.array(indices_str, dtype=float)


def path_to_mask(path, shape):
    """From SVG path to a boolean array where all pixels enclosed by the path
    are True, and the other pixels are False.
    """
    cols, rows = path_to_indices(path).T
    cols = cols * 1e3 / cfg.RANGE_RESOLUTION
    rr, cc = draw.polygon(rows, cols)

    # Range is now values in kilometers, so convert to pixels
    # print(cc)
    cc = np.rint(cc).astype(int)
    rr = np.rint(rr).astype(int)

    # Fix any values outside figure
    rr[rr >= shape[0]] = shape[0] - 1
    cc[cc >= shape[1]] = shape[1] - 1

    mask = np.zeros(shape, dtype=bool)
    mask[rr, cc] = True
    mask = ndimage.binary_fill_holes(mask)
    return mask


def empty_bscan():
    fig = px.imshow(
        np.zeros((360, 500)),
        labels=dict(
            x="Range [km]",
            y="Ray",
        ),
        color_continuous_scale="viridis",
        zmax=0,
        zmin=0,
    )
    fig.update_layout(template=None)
    fig.update_xaxes(showgrid=False, showticklabels=False, zeroline=False)
    fig.update_yaxes(showgrid=False, showticklabels=False, zeroline=False)

    return fig


app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.layout = html.Div(
    [
        html.Div(
            [html.H2("Select areas from radar image", style={"textAlign": "center"})]
        ),
        # First row
        html.Div(
            [
                html.Div([], style={"width": "2%"}),
                html.Div(
                    [
                        html.Label("Directory format"),
                        html.Br(),
                        dbc.Input(
                            type="text",
                            value=cfg.DEFAULT_PATH_FORMAT,
                            id="rootpath-input",
                        ),
                    ],
                    style={"width": "20%"},
                ),
                html.Div([], style={"width": "1%"}),
                html.Div(
                    [
                        html.Label("Filename format"),
                        html.Br(),
                        dbc.Input(
                            type="text",
                            value=cfg.DEFAULT_FILE_FORMAT,
                            id="filepath-input",
                        ),
                    ],
                    style={"width": "15%"},
                ),
                html.Div([], style={"width": "1%"}),
                html.Div(
                    [
                        html.Label("DD-MM-YYYY"),
                        html.Br(),
                        dmc.DatePicker(
                            id="date-input",
                            value=datetime.now().date(),
                            maxDate=datetime.now().date(),
                            inputFormat="DD-MM-YYYY",
                            # label="DD-MM-YYYY",
                        ),
                    ],
                    style={"width": "7%"},
                ),
                html.Div([], style={"width": "1%"}),
                html.Div(
                    [
                        html.Label("HH:MM"),
                        html.Br(),
                        dbc.Input(
                            id="time-input", value="12:00", type="time", step=300
                        ),
                    ],
                    style={"width": "7%"},
                ),
                html.Div([], style={"width": "1%"}),
                html.Div(
                    [
                        html.Label("Radar site"),
                        html.Br(),
                        dcc.Dropdown(id="site-input", options=cfg.RADAR_OPTIONS),
                    ],
                    style={"width": "10%"},
                ),
                html.Div([], style={"width": "1%"}),
                html.Div(
                    [
                        html.Br(),
                        dbc.Button("Load file", id="loadfile-button"),
                        html.Span(
                            id="loadfile-output",
                            style={
                                "verticalAlign": "middle",
                                "width": "100%",
                                "padding-left": "10px",
                            },
                        ),
                    ],
                    style={"width": "30%"},
                ),
            ],
            style={"display": "flex", "flex-direction": "horizontal"},
        ),
        # Second row
        html.Div(
            [
                html.Div([], style={"width": "2%"}),
                html.Div(
                    [
                        html.Label("Radar variable"),
                        html.Br(),
                        dcc.Dropdown(value="DBZH", id="var-dropdown"),
                    ],
                    style={"width": "20%"},
                ),
                html.Div([], style={"width": "1%"}),
                html.Div(
                    [
                        html.Label("Elevation angle [°]"),
                        html.Br(),
                        dcc.Dropdown(value="dataset1", id="elev-dropdown"),
                    ],
                    style={"width": "10%"},
                ),
                html.Div([], style={"width": "67%"}),
            ],
            style={"display": "flex", "flex-direction": "horizontal"},
        ),
        # Third row
        html.Div(
            [
                html.Div([], style={"width": "2%"}),
                html.Div(
                    [
                        html.Label("Output path"),
                        html.Br(),
                        dbc.Input(
                            type="text",
                            value=cfg.DEFAUL_OUTPUT_PATH,
                            id="outpath-input",
                        ),
                    ],
                    style={"width": "20%"},
                ),
                html.Div([], style={"width": "1%"}),
                html.Div(
                    [
                        html.Br(),
                        dbc.Button("Save mask", id="savemask-button"),
                        html.Span(
                            id="savemask-output",
                            style={
                                "verticalAlign": "middle",
                                "width": "100%",
                                "padding-left": "10px",
                            },
                        ),
                    ],
                    style={"width": "40%"},
                ),
                html.Div([], style={"width": "37%"}),
            ],
            style={"display": "flex", "flex-direction": "horizontal"},
        ),
        # Title for figures
        html.Div([html.H4(id="radar-title", style={"textAlign": "center"})]),
        # Figure
        html.Div(
            [
                dcc.Graph(
                    id="graph-bscan",
                    config=cfg.BSCAN_GRAPH_CONFIG,
                    figure=empty_bscan(),
                    style={"height": "70vh"},
                ),
            ],
            style={"width": "40%", "display": "inline-block", "padding": "0 0"},
        ),
        html.Div(
            [
                dcc.Graph(id="graph-histogram", style={"height": "70vh"}),
                # html.Br(),
                # html.Div([], style={"width": "5%"}),
            ],
            style={"width": "20%", "display": "inline-block", "padding": "0 0"},
        ),
        html.Div(
            [
                dcc.Graph(id="graph-ppi", config={"doubleClick": "reset"}),
            ],
            style={"width": "40%", "display": "inline-block", "padding": "0 0"},
        ),
        # dcc.Store stores the intermediate value
        dcc.Store(id="bscan-image"),
        dcc.Store(id="bscan-mask"),
        dcc.Store(id="cur-mask-dataset"),
        dcc.Store(id="full-filepath"),
    ]
)


@app.callback(
    Output("loadfile-output", "children"),
    Output("full-filepath", "data"),
    Input("loadfile-button", "n_clicks"),
    State("rootpath-input", "value"),
    State("filepath-input", "value"),
    State("date-input", "value"),
    State("time-input", "value"),
    State("site-input", "value"),
    prevent_initial_call=True,
)
def generate_filename(n_clicks, path_format, file_format, date, time, radar):
    date = arrow.get(date + " " + time).datetime
    path = Path(date.strftime(path_format).format(radar=radar))
    file = Path(date.strftime(file_format).format(radar=radar))
    fullpath = path / file
    if fullpath.exists():
        return "OK!", json.dumps(str(fullpath))
    else:
        return f"File {fullpath} doesn't exist!", None


@app.callback(
    Output("graph-histogram", "figure"),
    Output("bscan-mask", "data"),
    Input("graph-bscan", "relayoutData"),
    Input("bscan-image", "data"),
    Input("var-dropdown", "value"),
    prevent_initial_call=True,
)
def on_new_annotation(relayout_data, json_arr, qty):
    # Try to unpack array from JSON list
    try:
        arr = np.array(json.loads(json_arr))
    except:
        return dash.no_update

    if "shapes" in relayout_data:
        # Get mask from shapes
        MASK = np.zeros_like(arr).astype(bool)
        for shape in relayout_data["shapes"]:
            mask = path_to_mask(shape["path"], arr.shape)
            MASK = MASK | mask

        # Histogram figure
        hist = px.histogram(
            arr[MASK],
            labels={
                "value": qty,
                "y": "Count",
            },
            color=None,
        )

        return hist, json.dumps(MASK.tolist())
    else:
        return dash.no_update


@app.callback(
    Output("elev-dropdown", "options"),
    Output("var-dropdown", "options"),
    Input("full-filepath", "data"),
    prevent_initial_call=True,
)
def populate_lists(json_fullpath):
    if json_fullpath is None:
        return dash.no_update

    elevations = []
    qty_list = []

    def get_variables(name, node):
        if name.endswith("/where") and "elangle" in node.attrs.keys():
            elevations.append(
                {"label": node.attrs["elangle"].item(), "value": name.split("/")[0]}
            )

        if name.endswith("/what") and "quantity" in node.attrs.keys():
            if node.attrs["quantity"].decode() not in qty_list:
                qty_list.append(node.attrs["quantity"].decode())

    fullpath = Path(json.loads(json_fullpath))
    with h5py.File(fullpath, "r") as file:
        file.visititems(get_variables)

    return elevations, qty_list


def plot_bscan(radar, arr, palette, zmin, zmax, colorname, tickformat):
    # Create figure
    fig = px.imshow(
        arr.filled(),
        x=radar.range["data"] * 1e-3,
        aspect="auto",
        color_continuous_scale=palette,
        zmin=zmin,
        zmax=zmax,
        labels=dict(
            x="Range [km]",
            y="Ray",
            color=colorname,
        ),
    )
    # Colorbar settings
    fig.update_layout(
        coloraxis_colorbar=dict(
            thicknessmode="pixels",
            thickness=30,
            lenmode="fraction",
            len=0.8,
            xpad=0,
            title=dict(
                side="right",
            ),
            tickformat=tickformat,
        )
    )
    fig.update_yaxes(automargin=True)
    fig.update_xaxes(automargin=True)

    # Default annotation method
    fig.update_layout(dragmode="drawclosedpath")
    return fig


def plot_ppi(radar, qty, field_name):
    # PPI figure as static image
    fig_ppi = dash_utils.plot_onepanel_ppi(
        radar,
        qty,
        field_name,
        radar.range["data"].max() * 1e-3,
    )

    # Save to the in-memory file object
    buf = io.BytesIO()
    fig_ppi.savefig(buf, dpi=600, format="png", bbox_inches="tight")
    plt.close(fig_ppi)

    # Encode to html elements
    data = base64.b64encode(buf.getbuffer()).decode("utf-8")

    img_width = 800
    img_height = 700

    fig_ppi = go.Figure()
    fig_ppi.add_trace(
        go.Scatter(
            x=[0, img_width], y=[img_height, 0], mode="markers", marker_opacity=0
        )
    )
    # Add image as background
    scale_factor = 0.2
    fig_ppi.add_layout_image(
        dict(
            x=0,
            y=0,
            sizex=img_width * scale_factor,
            sizey=img_height * scale_factor,
            xref="x",
            yref="y",
            opacity=1.0,
            layer="above",
            visible=True,
            sizing="contain",
            source="data:image/png;base64,{}".format(data),
        )
    )
    fig_ppi.update_xaxes(
        showgrid=False,
        range=(0, img_width * scale_factor),
        showline=False,
        zeroline=False,
        visible=False,
    )
    fig_ppi.update_yaxes(
        range=(img_height * scale_factor, 0),
        showgrid=False,
        scaleanchor="x",
        showline=False,
        zeroline=False,
        visible=False,
    )
    fig_ppi.update_layout(
        height=img_height,
        width=img_width,
        template="none",
    )
    return fig_ppi


@app.callback(
    Output("graph-bscan", "figure"),
    Output("bscan-image", "data"),
    Output("cur-mask-dataset", "data"),
    Output("graph-bscan", "relayoutData"),
    Output("radar-title", "children"),
    Input("full-filepath", "data"),
    Input("var-dropdown", "value"),
    Input("elev-dropdown", "value"),
    State("graph-bscan", "relayoutData"),
    State("cur-mask-dataset", "data"),
    prevent_initial_call=True,
)
def create_figures(json_fullpath, qty, dataset, relayoutData, prev_mask_dataset):
    if json_fullpath is None:
        return dash.no_update

    fullpath = Path(json.loads(json_fullpath))
    field_name = plot_utils.PYART_FIELDS[qty]

    radar = pyart.aux_io.read_odim_h5(
        fullpath,
        include_datasets=[
            dataset,
        ],
        include_fields=[
            field_name,
        ],
    )

    arr = radar.get_field(0, field_name)
    arr.set_fill_value(np.nan)

    cmap, norm = plot_utils.get_colormap(qty)
    norm = mpl.colors.Normalize(
        vmin=plot_utils.QTY_RANGES[qty][0], vmax=plot_utils.QTY_RANGES[qty][1]
    )
    palette = dash_utils.cmap_to_RGB(cmap, norm)

    # Create Bscan figure
    fig = plot_bscan(
        radar,
        arr,
        palette,
        plot_utils.QTY_RANGES[qty][0],
        plot_utils.QTY_RANGES[qty][1],
        plot_utils.COLORBAR_TITLES[qty],
        plot_utils.QTY_FORMATS[qty].split(":")[1][:-1],
    )

    # Get existing annotations if elevation did not change
    if prev_mask_dataset is not None and (json.loads(prev_mask_dataset) == dataset):
        fig.plotly_relayout(relayoutData)
    else:
        relayoutData = {}

    title = (
        f'{radar.metadata["source"].split(",")[2].split(":")[1]} '
        f'{datetime.strptime(fullpath.stem.split("_")[0], "%Y%m%d%H%M")} '
        f"{dataset} {qty}"
    )

    return (
        fig,
        json.dumps(arr.filled().tolist()),
        json.dumps(dataset),
        relayoutData,
        title,
    )


@app.callback(
    Output("graph-ppi", "figure"),
    Input("full-filepath", "data"),
    Input("var-dropdown", "value"),
    Input("elev-dropdown", "value"),
    prevent_initial_call=True,
)
def create_ppi_figure(
    json_fullpath,
    qty,
    dataset,
):
    if json_fullpath is None:
        return dash.no_update

    fullpath = Path(json.loads(json_fullpath))
    field_name = plot_utils.PYART_FIELDS[qty]

    radar = pyart.aux_io.read_odim_h5(
        fullpath,
        include_datasets=[
            dataset,
        ],
        include_fields=[
            field_name,
        ],
    )

    # PPI figure as static image
    fig_ppi = plot_ppi(radar, qty, field_name)

    # title = (
    #     f'{radar.metadata["source"].split(",")[2].split(":")[1]} '
    #     f'{datetime.strptime(fullpath.stem.split("_")[0], "%Y%m%d%H%M")} '
    #     f"{dataset} {qty}"
    # )

    return fig_ppi


@app.callback(
    Output("savemask-output", "children"),
    Input("savemask-button", "n_clicks"),
    State("bscan-mask", "data"),
    State("outpath-input", "value"),
    State("full-filepath", "data"),
    State("elev-dropdown", "value"),
    prevent_initial_call=True,
)
def write_mask_to_hdf5(n_clicks, json_mask, outpath, json_filepath, dataset):
    if json_mask is None:
        return "No mask selected!"

    mask = np.array(json.loads(json_mask))

    radarpath = Path(json.loads(json_filepath))
    outname = radarpath.stem + "_mask" + radarpath.suffix
    outfile = Path(outpath) / outname

    orig = h5py.File(radarpath, "r")
    new = h5py.File(outfile, "a")

    # Copy attributes from original file
    dset = new.require_group(dataset)

    # Copy attributes from the original file
    if "where" in dset.keys():
        del dset["where"]
    orig.copy("%s/where" % dataset, new["/%s" % dataset])
    if "what" in dset.keys():
        del dset["what"]
    orig.copy("%s/what" % dataset, new["/%s" % dataset])
    if "where" in new.keys():
        del new["where"]
    orig.copy("where", new["/"])
    if "what" in new.keys():
        del new["what"]
    orig.copy("what", new["/"])
    orig.close()

    # Save mask as dataset
    group = dset.require_group("data1")

    # Delete existing mask
    if "data" in group.keys():
        del group["data"]

    ds = group.create_dataset("data", data=mask.astype(bool))
    # Some attributes to allow easier handling as an image later on
    ds.attrs["CLASS"] = np.string_("IMAGE")
    ds.attrs["IMAGE_VERSION"] = np.string_("1.2")

    # Remember to close
    new.close()

    return f"Saved to {outfile.name}!"


if __name__ == "__main__":
    app.run_server(debug=True)
