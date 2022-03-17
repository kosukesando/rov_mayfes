#%%
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import xarray as xr
import matplotlib as mpl
from scipy.stats import circmean
from scipy.stats import circstd
from search_coord import search_coord

mpl.rcParams["figure.facecolor"] = "white"

#%%
fl = list(Path("./reunion").glob("*.nc"))
num_events = 0
num_vars_stm = 2
num_vars_aux = 3
num_nodes = xr.load_dataset(fl[0])["node"].size
ds_latlon = xr.load_dataset(fl[0])[["lat", "lon"]]

ds_list = []
_ds_stm = []
_ds_exp = []
# load ds and reject empties
for ei, f in enumerate(fl):
    ds = xr.load_dataset(f)
    if ei == 1:
        print(ds)
    if ds.time.size == 0:
        continue
    num_events += 1
    ds_list.append(ds)
    _data = ds[["Hs", "U"]]
    _ds_stm.append(_data.max(dim=("time", "node")))
    _ds_exp.append(_data.max(dim="time") / _data.max(dim=("time", "node")))
stm = xr.concat(_ds_stm, dim="event")
exp = xr.concat(_ds_exp, dim="event")
var_name_stm = ["$H_s$", "$U$"]
var_name_aux = ["$T_p$", "$D_p$", "$D_u$"]
key_name_stm = ["Hs", "U"]
key_name_aux = ["Tp", "Dp", "Du"]
unit = ["m", "m/s"]

#%%
fig, ax = plt.subplots(1, 2, figsize=(8, 3), dpi=100, facecolor="white")
for vi in range(num_vars_stm):
    ax[vi].set_title(f"{var_name_stm[vi]}")
    ax[vi].hist(stm[key_name_stm[vi]])
    ax[vi].set_xlabel(unit[vi])

#%%
fig, ax = plt.subplots(3, 2, figsize=(8, 9), dpi=100)
fig.suptitle("STM vs mean variables")

for vi in range(num_vars_stm):
    for ai in range(num_vars_aux):
        _ax = ax[ai, vi]
        _stm = stm[key_name_stm[vi]]
        _aux = []
        _key = key_name_aux[ai]
        for ds in ds_list:
            if _key == "Tp":
                _aux.append(ds[_key].data.mean())
            else:
                _aux.append(circmean(ds[_key].data / 180 * np.pi))
        _ax.scatter(_stm, _aux)
        _ax.set_xlabel(f"{var_name_stm[vi]}")
        _ax.set_ylabel(f"{var_name_aux[ai]}")
plt.savefig("./output/aux_test_0317/stm_vs_mean_aux.png")

#%%
fig, ax = plt.subplots(3, 2, figsize=(8, 9), dpi=100)
fig.suptitle("STM vs std variables")

for vi in range(num_vars_stm):
    for ai in range(num_vars_aux):
        _ax = ax[ai, vi]
        _stm = stm[key_name_stm[vi]]
        _aux = []
        _key = key_name_aux[ai]
        for ds in ds_list:
            if _key == "Tp":
                _aux.append(ds[_key].data.flatten().std())
            else:
                _aux.append(circstd(ds[_key].data / 180 * np.pi))
        _ax.scatter(_stm, _aux)
        _ax.set_xlabel(f"{var_name_stm[vi]}")
        _ax.set_ylabel(f"{var_name_aux[ai]}")
plt.savefig("./output/aux_test_0317/stm_vs_std_aux.png")

#%%
plt.scatter(ds_latlon["lat"], ds_latlon["lon"])

# %% 10x10 grid of coordinates
# pos_list = []
# res = 10
# for _lat in np.linspace(ds_latlon["lat"].min(), ds_latlon["lat"].max(), res):
#     for _lon in np.linspace(ds_latlon["lon"].min(), ds_latlon["lon"].max(), res):
#         pos_list.append(search_coord(_lat, _lon, ds_latlon.to_array()))

#%%
_x1 = []
_x2 = []
for ei, ds in enumerate(ds_list):
    if ei != 0:
        continue
    _x1.extend(ds["Dp"][:, pos_list].data.flatten() / 180 * np.pi)
    _x2.extend(ds["Tp"][:, pos_list].data.flatten())
    # _x1.extend(circmean(ds['Dp'][:, pos_list].data/180*np.pi,axis=0))
    # _x2.extend(ds['Tp'][:, pos_list].mean(dim='time'))
_x1 = np.array(_x1)
_x2 = np.array(_x2)
fig, ax = plt.subplots(figsize=(16, 12), dpi=100, subplot_kw={"projection": "polar"})
ax.scatter(_x1, _x2, s=0.1)

#%%
_x1 = []
for ei, ds in enumerate(ds_list):
    if ei != 0:
        continue
    _x1.extend(ds["Dp"].data / 180 * np.pi)
    # _x1.extend(circmean(ds['Dp'][:, pos_list].data/180*np.pi,axis=0))
_x1 = np.array(_x1)
fig, ax = plt.subplots(figsize=(8, 6), dpi=100)
for i in range(100):
    ax.plot(np.unwrap(_x1[:, i], period=np.pi))
ax.axhline(np.pi)
ax.axhline(-np.pi)
#%%
plt.plot(_x1[:, 1])

#%%
_x1 = []
for ei, ds in enumerate(ds_list):
    _x1.extend(ds["Dp"].data.flatten())
_x1 = np.array(_x1)
fig, ax = plt.subplots(figsize=(16, 12), dpi=100)
ax.hist(_x1)

#%%
fig, ax = plt.subplots()
for i in range(5):
    ax.hist(circmean(ds_list[i]["Dp"].data / 180 * np.pi, axis=0), bins=50)
# circmean(np.array([])/180*np.pi,axis=0)
# %%
