import netCDF4 as nc
import numpy as np


def search_coord(s_lat, s_lon, latlon):
    _dist = 10000
    _idx_res = 0
    for i in range(latlon.shape[0]):
        d = np.linalg.norm([s_lon - latlon[i, 1], s_lat - latlon[i, 0]])
        if d < _dist:
            _dist = d
            _idx_res = i
    return _idx_res
