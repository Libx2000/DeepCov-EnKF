# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import numpy as np

NAME_TO_VAR = {
    "land_sea_mask": "lsm",
    "orography": "z",
    "latitude": "lat",
    "longitude": "lon",
    "t2m": "t2m",
    "u10": "u10",
    "v10": "v10",
    "msl": "msl",
    "z": "z",
    "thetao": "thetao",
    "so": "so",
    "uo": "uo",
    "vo": "vo",
    "mld": "mlotst",
    "ssh": "zos",
    "ice_area_fraction": "siconc",
    "ice_thickness": "sithick",
    
}

VAR_TO_NAME = {v: k for k, v in NAME_TO_VAR.items()}

ATMOS_SINGLE_LEVEL_VARS = [
    "land_sea_mask",
    "orography",
    "latitude",
    "longitude",
    "t2m",
    "u10",
    "v10",
    "msl",
]
ATMOS_PRESSURE_LEVEL_VARS = [
    "z",
]

DEFAULT_PRESSURE_LEVELS = [500, 850]

OCEAN_SURFACE_VARS = [
    "ssh",
    "ice_area_fraction",
    "ice_thickness"
]

OCEAN_DEPTH_VARS = [
    "thetao",
    "so",
    "uo",
    "vo",
]

DEFAULT_OCEAN_DEPTHS = [0.5, 5, 20, 40, 60, 90, 120, 150, 200]