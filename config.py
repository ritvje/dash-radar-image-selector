# Configurations for the app
RADAR_OPTIONS = sorted(
    [
        {"label": "Korppoo", "value": "fikor"},
        {"label": "Ikaalinen", "value": "fiika"},
        {"label": "Vantaa", "value": "fivan"},
        {"label": "Luosto", "value": "filuo"},
        {"label": "Nurmes", "value": "finur"},
        {"label": "Anjalankoski", "value": "fianj"},
        {"label": "Petäjävesi", "value": "fipet"},
        {"label": "Utajärvi", "value": "fiuta"},
        {"label": "Kesälahti", "value": "fikes"},
        {"label": "Kuopio", "value": "fikuo"},
        {"label": "Vihti", "value": "fivih"},
        {"label": "Vimpeli", "value": "fivim"},
    ],
    key=lambda d: d["label"],
)

DEFAULT_PATH_FORMAT = "/mnt/hdf5/%Y/%m/%d/radar/polar/{radar}/"
DEFAULT_FILE_FORMAT = "%Y%m%d%H%M_radar.polar.{radar}.h5"
DEFAUL_OUTPUT_PATH = "."

BSCAN_GRAPH_CONFIG = {
    "modeBarButtonsToAdd": [
        "drawline",
        "drawopenpath",
        "drawclosedpath",
        "drawcircle",
        "drawrect",
        "eraseshape",
    ]
}
