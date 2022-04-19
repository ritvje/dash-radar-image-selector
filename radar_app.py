import numpy as np
import json
import dash
from pathlib import Path
from dash import Dash, dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
import plotly.express as px
import matplotlib as mpl
from dash.dependencies import Input, Output
from skimage import draw
from scipy import ndimage
import h5py
import pyart

from radar_plotting import plot_utils, dash_utils

config = {
    "modeBarButtonsToAdd": [
        "drawline",
        "drawopenpath",
        "drawclosedpath",
        "drawcircle",
        "drawrect",
        "eraseshape",
    ]
}


def path_to_indices(path):
    """From SVG path to numpy array of coordinates, each row being a (row, col) point"""
    indices_str = [
        el.replace("M", "").replace("Z", "").split(",") for el in path.split("L")
    ]
    return np.rint(np.array(indices_str, dtype=float)).astype(int)


def path_to_mask(path, shape):
    """From SVG path to a boolean array where all pixels enclosed by the path
    are True, and the other pixels are False.
    """
    cols, rows = path_to_indices(path).T
    rr, cc = draw.polygon(rows, cols)

    # Fix any values outside figure
    rr[rr >= shape[0]] = shape[0] - 1
    cc[cc >= shape[1]] = shape[1] - 1

    mask = np.zeros(shape, dtype=np.bool)
    mask[rr, cc] = True
    mask = ndimage.binary_fill_holes(mask)
    return mask


def empty_bscan():
    fig = px.imshow(
        np.zeros((360, 500)),
        labels=dict(
            x="Range",
            y="Azimuth",
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
        html.Div(
            [
                html.Div([], style={"width": "5%"}),
                html.Div(
                    [
                        html.Label("Directory"),
                        html.Br(),
                        dbc.Input(type="text", value=".", id="rootpath-input"),
                    ],
                    style={"width": "20%"},
                ),
                html.Div([], style={"width": "5%"}),
                html.Div(
                    [
                        html.Label("Select file"),
                        html.Br(),
                        dcc.Dropdown(id="radar-input"),
                    ],
                    style={"width": "20%"},
                ),
                html.Div([], style={"width": "50%"}),
            ],
            style={"display": "flex", "flex-direction": "horizontal"},
        ),
        html.Div(
            [
                html.Div([], style={"width": "5%"}),
                html.Div(
                    [
                        html.Label("Radar variable"),
                        html.Br(),
                        dcc.Dropdown(value="DBZH", id="var-dropdown"),
                    ],
                    style={"width": "20%"},
                ),
                html.Div([], style={"width": "5%"}),
                html.Div(
                    [
                        html.Label("Elevation angle [°]"),
                        html.Br(),
                        dcc.Dropdown(value="dataset1", id="elev-dropdown"),
                    ],
                    style={"width": "20%"},
                ),
                html.Div([], style={"width": "50%"}),
            ],
            style={"display": "flex", "flex-direction": "horizontal"},
        ),
        html.Div(
            [
                html.Div([], style={"width": "5%"}),
                html.Div(
                    [
                        html.Label("Output path"),
                        html.Br(),
                        dbc.Input(type="text", value=".", id="outpath-input"),
                    ],
                    style={"width": "20%"},
                ),
                html.Div([], style={"width": "5%"}),
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
                html.Div([], style={"width": "10%"}),
            ],
            style={"display": "flex", "flex-direction": "horizontal"},
        ),
        html.Div(
            [
                dcc.Graph(
                    id="graph-bscan",
                    config=config,
                    figure=empty_bscan(),
                    style={"height": "70vh"},
                ),
            ],
            style={"width": "40%", "display": "inline-block", "padding": "0 0"},
        ),
        # html.Div(
        #     [
        #         html.Label("PPI image"),
        #         # dcc.Graph(id="graph-histogram", figure=fig_hist),
        #     ],
        #     style={"width": "40%", "display": "inline-block", "padding": "0 0"},
        # ),
        html.Div(
            [
                dcc.Graph(id="graph-histogram", style={"height": "70vh"}),
            ],
            style={"width": "20%", "display": "inline-block", "padding": "0 0"},
        ),
        # dcc.Store stores the intermediate value
        dcc.Store(id="bscan-image"),
        dcc.Store(id="bscan-mask"),
        dcc.Store(id="cur-mask-dataset"),
    ]
)


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
    Output("radar-input", "options"),
    Input("rootpath-input", "value"),
    prevent_initial_call=False,
)
def get_filelist(directory):
    path = Path(directory)
    file_list = [str(f) for f in path.glob("*.h5")]
    return file_list


@app.callback(
    Output("elev-dropdown", "options"),
    Output("var-dropdown", "options"),
    Input("rootpath-input", "value"),
    Input("radar-input", "value"),
    prevent_initial_call=True,
)
def populate_lists(path, file):
    elevations = []
    qty_list = []

    def get_elevation_angles(name, node):
        if name.endswith("/where") and "elangle" in node.attrs.keys():
            elevations.append(
                {"label": node.attrs["elangle"][0], "value": name.split("/")[0]}
            )

    def get_variables(name, node):
        if name.endswith("/what") and "quantity" in node.attrs.keys():
            if node.attrs["quantity"].decode() not in qty_list:
                qty_list.append(node.attrs["quantity"].decode())

    with h5py.File(Path(path) / file, "r") as file:
        # Elevation angles
        file.visititems(get_elevation_angles)
        # Radar variables
        file.visititems(get_variables)

    return elevations, qty_list


@app.callback(
    Output("graph-bscan", "figure"),
    Output("bscan-image", "data"),
    Output("cur-mask-dataset", "data"),
    Output("graph-bscan", "relayoutData"),
    Input("rootpath-input", "value"),
    Input("radar-input", "value"),
    Input("var-dropdown", "value"),
    Input("elev-dropdown", "value"),
    State("graph-bscan", "relayoutData"),
    State("cur-mask-dataset", "data"),
    prevent_initial_call=True,
)
def create_fig(path, file, qty, dataset, relayoutData, prev_mask_dataset):
    radar = pyart.aux_io.read_odim_h5(
        Path(path) / file,
        include_datasets=[
            dataset,
        ],
    )

    arr = radar.get_field(0, plot_utils.PYART_FIELDS_ODIM[qty])
    arr.set_fill_value(np.nan)

    cmap, norm = plot_utils.get_colormap(qty)
    norm = mpl.colors.Normalize(
        vmin=plot_utils.QTY_RANGES[qty][0], vmax=plot_utils.QTY_RANGES[qty][1]
    )
    palette = dash_utils.cmap_to_RGB(cmap, norm)

    # Create figure
    fig = px.imshow(
        arr.filled(),
        color_continuous_scale=palette,
        zmin=plot_utils.QTY_RANGES[qty][0],
        zmax=plot_utils.QTY_RANGES[qty][1],
        labels=dict(
            x="Range",
            y="Azimuth",
            color=plot_utils.COLORBAR_TITLES[qty],
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
            tickformat=plot_utils.QTY_FORMATS[qty].split(":")[1][:-1],
        )
    )
    fig.update_yaxes(automargin=True)
    fig.update_xaxes(automargin=True)

    # Default annotation method
    fig.update_layout(dragmode="drawclosedpath")

    # Get existing annotations if elevation did not change
    if prev_mask_dataset is not None and (json.loads(prev_mask_dataset) == dataset):
        fig.plotly_relayout(relayoutData)
    else:
        relayoutData = {}

    return fig, json.dumps(arr.filled().tolist()), json.dumps(dataset), relayoutData


@app.callback(
    Output("savemask-output", "children"),
    Input("savemask-button", "n_clicks"),
    State("bscan-mask", "data"),
    State("outpath-input", "value"),
    State("rootpath-input", "value"),
    State("radar-input", "value"),
    State("elev-dropdown", "value"),
    prevent_initial_call=True,
)
def write_mask_to_hdf5(n_clicks, json_mask, outpath, orig_path, filename, dataset):
    if json_mask is None:
        return "No mask selected!"

    mask = np.array(json.loads(json_mask))

    outfile = Path(filename)
    outfile = Path(outpath) / (outfile.stem + "_mask" + outfile.suffix)

    radarpath = Path(orig_path) / filename

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
