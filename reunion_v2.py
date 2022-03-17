# %% init
from numpy.random import default_rng
from statsmodels.distributions.empirical_distribution import monotone_fn_inverter
import numpy as np
from scipy.stats._continuous_distns import genpareto
import xarray as xr
import matplotlib as mpl
from scipy.stats import circmean
from scipy.stats import circstd
from scipy.stats import kendalltau
from statsmodels.distributions.empirical_distribution import ECDF
from sklearn.neighbors import KernelDensity
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from pathlib import Path
import netCDF4 as nc
import importlib
import stme
from pathlib import Path
from datetime import datetime
import argparse
from search_coord import search_coord

mpl.rcParams["figure.facecolor"] = "white"

#%% CONFIG only used for jupyter###############
thr_org = np.array([6.0, 20])
thr_gum = 1.0
now = datetime.now()
dt_string = now.strftime("%Y-%m-%d-%H%M")
dir_out = f"./output/aux_test/{thr_org[0]}m_{thr_org[1]}mps_{thr_gum}_{dt_string}_gumbel_exceedance/"
path_out = Path(dir_out)
if not path_out.exists():
    path_out.mkdir()
###############################################
def main():
    # %%Load STM & Exposure
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

    ##################################################################################################################
    # %%Generalized Pareto estimation
    N = 100
    genpar_params = np.zeros((N, 3, num_vars))
    gp = [None, None]

    for vi in range(num_vars):
        _stm_bootstrap = rng.choice(stm[:, vi], size=(N, stm[:, vi].shape[0]))
        for i in range(N):
            _stm = _stm_bootstrap[i, :]
            _stm_pot = _stm[_stm > thr_org[vi]]
            _xp, _mp, _sp = genpareto.fit(_stm_pot, floc=thr_org[vi])
            genpar_params[i, :, vi] = [_xp, _mp, _sp]
        # xp, mp, sp = np.percentile(genpar_params[:,:,vi],50.0,axis=0)
        xp, mp, sp = np.median(genpar_params[:, :, vi], axis=0)
        print(f"GENPAR{xp, mp, sp}")
        gp[vi] = genpareto(xp, mp, sp)
    par_name = ["$\\xi$", "$\\mu$", "$\\sigma$"]

    #########################################################
    fig, ax = plt.subplots(len(par_name), 2, figsize=(8, 9), dpi=100)

    for vi in range(num_vars):
        ax[0, vi].set_title(var_name[vi])
        for pi, p in enumerate(par_name):
            ax[pi, 0].set_ylabel(par_name[pi])
            ax[pi, vi].hist(genpar_params[:, pi, vi])
    plt.savefig(f"{dir_out}/Genpar_Params.png", bbox_inches="tight")
    #########################################################

    #########################################################
    fig, ax = plt.subplots(1, 2, figsize=(8, 3), dpi=100)
    fig.set_facecolor("white")
    ax[0].set_ylabel("CDF")

    for vi in range(num_vars):
        for i in range(N):
            _x = np.linspace(thr_org[vi], stm[:, vi].max(), 100)
            _xp = genpar_params[i, 0, vi]
            _mp = genpar_params[i, 1, vi]
            _sp = genpar_params[i, 2, vi]
            _y = genpareto(_xp, _mp, _sp).cdf(_x)
            ax[vi].plot(_x, _y, c="red", lw=0.2, alpha=1)

    for vi in range(num_vars):
        _x = np.linspace(thr_org[vi], stm[:, vi].max(), 100)
        _y = gp[vi].cdf(_x)
        ax[vi].plot(_x, _y, c="blue", lw=2, alpha=1)

    for vi in range(num_vars):
        _ecdf = ECDF(stm[stm[:, vi] > thr_org[vi], vi])
        _x = np.linspace(thr_org[vi], stm[:, vi].max(), 100)
        ax[vi].plot(_x, _ecdf(_x), lw=2, color="black")
        ax[vi].set_xlabel(f"{var_name[vi]}{unit[vi]}")

    plt.savefig(f"{dir_out}/Genpar_CDF.png", bbox_inches="tight")
    #########################################################

    ##################################################################################################################
    # %%Gumbel Transform
    importlib.reload(stme)
    stm_g = np.zeros(stm.shape)
    f_hat_cdf = [None, None]
    for vi in range(num_vars):
        f_hat_cdf[vi] = lambda x, idx=vi: stme._f_hat_cdf(ECDF(stm[:, idx]), gp[idx], x)
        _stm = stm[:, vi]
        stm_g[:, vi] = -np.log(-np.log(f_hat_cdf[vi](_stm)))
    # print(stm_g[:,0].argmax(), stm_g[:,1].argmax())
    print(stm_g[:, 0].max(), stm_g[:, 1].max())
    print(stm_g[:, 0].min(), stm_g[:, 1].min())

    #########################################################
    fig, ax = plt.subplots(1, 2, figsize=(7, 3), dpi=100)

    ax[0].scatter(stm[:, 0], stm[:, 1])
    ax[0].set_xlabel(f"{var_name[0]}{unit[0]}")
    ax[0].set_ylabel(f"{var_name[1]}{unit[1]}")
    ax[0].set_xlim(0, 20)
    ax[0].set_ylim(0, 60)

    ax[1].set_aspect(1)
    ax[1].scatter(stm_g[:, 0], stm_g[:, 1])
    ax[1].set_xlabel(r"$\hat H_s$")
    ax[1].set_ylabel(r"$\hat U$")
    ax[1].set_xlim(-2, 8)
    ax[1].set_ylim(-2, 8)

    plt.savefig(f"{dir_out}/Original_vs_Gumbel.png", bbox_inches="tight")
    #########################################################

    ##################################################################################################################
    # Set boolean mask
    is_e_hg = stm_g[:, 0] > thr_gum
    is_e_ug = stm_g[:, 1] > thr_gum
    is_me_hg = np.logical_and(stm_g[:, 0] > stm_g[:, 1], is_e_hg)
    is_me_ug = np.logical_and(stm_g[:, 1] > stm_g[:, 0], is_e_ug)
    is_e = np.stack((is_e_hg, is_e_ug), axis=1)
    is_me = np.stack((is_me_hg, is_me_ug), axis=1)

    ##################################################################################################################
    # %% Kendall's Tau
    tau = np.zeros(latlon.shape)
    pval = np.zeros(latlon.shape)
    for vi in range(num_vars):
        _stm = stm_g[:, vi]
        _exp = exp[:, :, vi]
        _mask = is_e[:, vi]
        for i in range(num_nodes):
            _tau, _pval = kendalltau(_stm[_mask], _exp[_mask, i])
            tau[i, vi] = _tau
            pval[i, vi] = _pval

    #########################################################
    fig, ax = plt.subplots(
        1, 2, sharey=True, figsize=(8, 3), dpi=100, facecolor="white"
    )
    # fig.supxlabel("Longitude")
    # fig.supylabel("Latitude")

    for vi in range(num_vars):
        ax[vi].set_xlabel("Longitude")
        ax[vi].set_ylabel("Latitude")
        _c = ["red" if p < 0.05 else "black" for p in pval[:, vi]]
        im = ax[vi].scatter(latlon[:, 1], latlon[:, 0], s=1, c=_c)
        ax[vi].set_title(var_name[vi])
        for i, _idx_pos in enumerate(idx_pos_list_saint_denis):
            ax[vi].scatter(
                latlon[_idx_pos, 1], latlon[_idx_pos, 0], s=50, color=pos_color[i]
            )
            ax[vi].annotate(
                f"#{i + 1}",
                (
                    latlon[_idx_pos, 1] + (i % 2 - 0.65) * 2 * 0.2,
                    latlon[_idx_pos, 0] - 0.01,
                ),
                bbox=dict(facecolor="white", edgecolor=pos_color[i]),
            )
    # for vi in range(num_vars):
    #   im = ax[vi].scatter(latlon[:,1],latlon[:,0],s=1,c=tau[:,vi])
    #   ax[vi].set_title(var_name[vi])
    #   fig.colorbar(im,ax=ax[vi])
    plt.savefig(f"{dir_out}/Kendall_Tau.png", bbox_inches="tight")
    #########################################################

    ##################################################################################################################
    # %%Gumbel replacement
    N = 1000
    stm_g_rep = np.zeros((N, *stm_g.shape))
    for i in range(N):
        _idx = rng.choice(stm_g.shape[0], size=stm_g.shape[0])
        _stm = stm_g[_idx, :]
        for vi in range(num_vars):
            _gumbel_sample = rng.gumbel(loc=0, scale=1, size=_stm[:, vi].shape[0])
            _gumbel_sample_sorted = np.sort(_gumbel_sample)
            _arg = np.argsort(_stm[:, vi])
            stm_g_rep[i, _arg, vi] = _gumbel_sample_sorted

    #########################################################
    fig, ax = plt.subplots(1, 1, figsize=(8, 6), dpi=100, facecolor="white")
    ax.scatter(stm_g_rep[:, :, 0], stm_g_rep[:, :, 1], alpha=0.1)
    ax.scatter(stm_g[:, 0], stm_g[:, 1], color="blue")
    ax.set_xlabel(r"$\hat H_s$")
    ax.set_ylabel(r"$\hat U$")
    ax.set_xlim(-3, 15)
    ax.set_ylim(-3, 15)
    plt.savefig(f"{dir_out}/Gumbel_Replacement.png", bbox_inches="tight")
    #########################################################

    ##################################################################################################################
    # %% Estimate conditional model parameters
    lb = [0, None, -5, 0]
    ub = [1, 1, 5, 5]
    N = stm_g_rep.shape[0]
    params_uc = np.zeros((N, 4, num_vars))
    for i in range(N):
        for vi in range(num_vars):
            a0 = np.random.uniform(low=lb[0], high=ub[0])
            b0 = np.random.uniform(low=-1, high=ub[1])
            m0 = np.random.uniform(low=-1, high=1)
            s0 = np.random.uniform(low=lb[3], high=1)
            # m0 = np.random.uniform(low=lb[2], high=ub[2])
            # s0 = np.random.uniform(low=lb[3], high=ub[3])
            evt_mask = stm_g_rep[i, :, vi] > thr_gum
            var_mask = np.full((stm_g_rep[i].shape[1]), True)
            var_mask[vi] = False

            def func(x):
                return stme.cost(
                    x, stm_g_rep[i, evt_mask, vi], stm_g_rep[i, evt_mask, var_mask]
                )

            optres = minimize(
                func,
                np.array([a0, b0, m0, s0]),
                # method='trust-constr',
                bounds=((lb[0], ub[0]), (lb[1], ub[1]), (lb[2], ub[2]), (lb[3], ub[3])),
            )
            _param = optres.x
            params_uc[i, :, vi] = _param
    params_median = np.median(params_uc, axis=0)

    #########################################################
    fig, ax = plt.subplots(4, 2, figsize=(4 * 2, 3 * 4), dpi=100)
    fig.tight_layout()
    ax[0, 0].set_ylabel("a")
    ax[1, 0].set_ylabel("b")
    ax[2, 0].set_ylabel("$\mu$")
    ax[3, 0].set_ylabel("$\sigma$")

    ax[3, 0].set_xlabel(var_name[0])
    ax[3, 1].set_xlabel(var_name[1])

    for vi in range(num_vars):
        ax[0, vi].hist(params_uc[:, 0, vi])
        ax[1, vi].hist(params_uc[:, 1, vi])
        ax[2, vi].hist(params_uc[:, 2, vi])
        ax[3, vi].hist(params_uc[:, 3, vi])
    plt.savefig(f"{dir_out}/Conmul_Estimates.png", bbox_inches="tight")
    #########################################################

    #########################################################
    fig, ax = plt.subplots(1, 2, figsize=(8, 3), dpi=100, facecolor="white")
    fig.supxlabel("$a$")
    fig.supylabel("$b$")
    params_ml = np.zeros((4, num_vars))
    for vi in range(num_vars):
        # _kde = KernelDensity(kernel="gaussian", bandwidth=0.2).fit(params_uc[:, 0:2, vi])
        # _x = np.linspace(lb[0], ub[0], 100)
        # _y = np.linspace(-1.5, ub[1], 100)
        # _x_grid, _y_grid = np.meshgrid(_x, _y)
        # _z_grid = np.reshape(np.stack((_x_grid, _y_grid), axis=2), (10000, 2))
        # _z = _kde.score_samples(_z_grid).reshape(100, 100)
        # _90_level = np.percentile(_z.flatten(), 90)
        # ax[vi].contour(
        #     _x_grid, _y_grid, _z, levels=[_90_level], colors="black", linestyles="solid"
        # )
        ax[vi].scatter(
            params_uc[:, 0, vi],
            params_uc[:, 1, vi],
            s=5,
            alpha=0.5,
            label="Generated samples",
        )
        # ax[vi].scatter(params[0,vi], params[1,vi],s=30,label='Original data')
        # ax[vi].legend()
        ax[vi].set_title(var_name[vi])
    plt.savefig(f"{dir_out}/ab_Estimates.png", bbox_inches="tight")
    #########################################################

    # #########################################################
    # fig, ax = plt.subplots(1, 1)
    # fig.set_size_inches(10, 10)
    # ax.set_aspect(1)
    # ax.scatter(stm_g[:, 0], stm_g[:, 1], s=10, label='original')
    # ax.axvline(thr_gum, color='black')
    # ax.axhline(thr_gum, color='black')

    # ax.set_xlabel(r'$\hat H_s$')
    # ax.set_ylabel(r'$\hat U$')
    # # ax.set_title(f'a:{}')
    # ax.set_xlim(-2, 8)
    # ax.set_ylim(-2, 8)

    # x_h = np.linspace(thr_gum, 8, 10)
    # y_u = np.linspace(thr_gum, 8, 10)
    # for i in range(100):
    #     a_h, b_h, mu_h, sg_h = params_uc[i, :, 0]
    #     a_u, b_u, mu_u, sg_u = params_uc[i, :, 1]

    #     y_h = x_h*a_h + (x_h**b_h)*mu_h
    #     ax.plot(x_h, y_h, alpha=0.1, color='orange')

    #     x_u = y_u*a_u + (y_u**b_u)*mu_u
    #     ax.plot(x_u, y_u, alpha=0.1, color='teal')

    # a_h, b_h, mu_h, sg_h = params_median[:, 0]
    # a_u, b_u, mu_u, sg_u = params_median[:, 1]

    # y_h = x_h*a_h + (x_h**b_h)*mu_h
    # ax.plot(x_h, y_h, color='orange')

    # x_u = y_u*a_u + (y_u**b_u)*mu_u
    # ax.plot(x_u, y_u, color='teal')
    # #########################################################

    ##################################################################################################################
    # %%Calculating residuals
    residual = []
    for vi in range(num_vars):
        _stm_g_thr = stm_g[is_e[:, vi], :]
        # _mask = sample_full_g[:,vi] > thr_gum
        # _stm_g_thr = sample_full_g[_mask,:]
        var_mask = np.full((stm_g.shape[1]), True)
        var_mask[vi] = False
        _x = _stm_g_thr[:, vi]  # conditioning(extreme)
        _y = _stm_g_thr[:, var_mask][:, 0]  # conditioned
        _z = (_y - params_median[0, vi] * _x) / (_x ** params_median[1, vi])
        residual.append(_z)
    #########################################################
    # fig, ax = plt.subplots(2, 2, figsize=(12, 9) , dpi=100, facecolor="white")
    # # fig.tight_layout()
    # fig.suptitle("Residuals")
    # for vi in range(num_vars):
    #     _stm_g_thr = stm_g[is_e[:, vi], :]
    #     var_mask = np.full((stm_g.shape[1]), True)
    #     var_mask[vi] = False
    #     _x = _stm_g_thr[:, vi]  # conditioning(extreme)
    #     _y = _stm_g_thr[:, var_mask][:, 0]  # conditioned
    #     _z = (_y - params_median[0, vi] * _x) / (_x ** params_median[1, vi])
    #     ax[0, vi].scatter(np.exp(-np.exp(-_x)), _z, s=5)
    #     # ax[0,vi].set_xlabel(f"$F_\\hat{{f'{{var_name[vi]}}'}}$" + f'$({var_name_g[vi]})$')
    #     ax[0, vi].set_xlabel(f"F({var_name[vi]})")
    #     ax[0, vi].set_ylabel(f"$Z_i$")
    #     # ax[0,vi].axhline(0,color='black',ls='--')
    #     ax[1, vi].hist(_z)
    #     _tau, _p = kendalltau(_x, _z)
    #     ax[0, vi].set_title(f"Tau:{_tau:.2e}, P-val:{_p:0.2e}")
    # plt.savefig(f"{dir_out}/Residuals.png", bbox_inches = 'tight')
    #########################################################

    #########################################################
    fig, ax = plt.subplots(1, 2, figsize=(8, 3), dpi=100, facecolor="white")
    # fig.tight_layout()
    for vi in range(num_vars):
        _stm_g_thr = stm_g[is_e[:, vi], :]
        var_mask = np.full((stm_g.shape[1]), True)
        var_mask[vi] = False
        _x = _stm_g_thr[:, vi]  # conditioning(extreme)
        _y = _stm_g_thr[:, var_mask][:, 0]  # conditioned
        _z = (_y - params_median[0, vi] * _x) / (_x ** params_median[1, vi])
        ax[vi].scatter(np.exp(-np.exp(-_x)), _z, s=5)
        ax[vi].set_xlabel(f"F({var_name[vi]})")
    ax[0].set_ylabel("$Z_{-j}$")
    plt.savefig(f"{dir_out}/Residuals.png", bbox_inches="tight")
    #########################################################

    ##################################################################################################################
    # %%Sample from model
    importlib.reload(stme)
    N_sample = 1000
    u_over_h = np.count_nonzero(is_me_ug) / (
        np.count_nonzero(is_me_hg) + np.count_nonzero(is_me_ug)
    )
    N_sample_h = round(N_sample * (1 - u_over_h))
    N_sample_u = N_sample - N_sample_h
    sample_given_hg = np.zeros((N_sample_h, 2))
    sample_given_ug = np.zeros((N_sample_u, 2))

    a_h, b_h, mu_h, sg_h = params_median[:, 0]
    a_u, b_u, mu_u, sg_u = params_median[:, 1]

    for i in range(N_sample_h):
        while True:
            _sample_h = -np.log(-np.log(rng.gumbel(0, 1)))
            # _sample_h = -np.log(-np.log( f_hat_cdf[0](gp[0].rvs())))
            if _sample_h > thr_gum:
                sample_given_hg[i, 0] = _sample_h
                break
        while True:
            _u_given_h = _sample_h * a_h + (_sample_h ** b_h) * np.random.choice(
                residual[0]
            )
            if _u_given_h < _sample_h:
                sample_given_hg[i, 1] = _u_given_h
                break
    for i in range(N_sample_u):
        while True:
            _sample_u = -np.log(-np.log(rng.gumbel(0, 1)))
            # _sample_u = -np.log(-np.log( f_hat_cdf[1](gp[1].rvs())))
            if _sample_u > thr_gum:
                sample_given_ug[i, 1] = _sample_u
                break
        while True:
            _h_given_u = _sample_u * a_u + (_sample_u ** b_u) * np.random.choice(
                residual[1]
            )
            if _h_given_u < _sample_u:
                sample_given_ug[i, 0] = _h_given_u
                break

    sample_full_g = np.concatenate((sample_given_hg, sample_given_ug))
    #########################################################
    # fig, ax = plt.subplots(1, 1)
    # fig.set_size_inches(10, 10)
    # fig.set_facecolor("white")
    # ax.set_aspect(1)

    # a_h, b_h, mu_h, sg_h = params_median[:, 0]
    # a_u, b_u, mu_u, sg_u = params_median[:, 1]

    # x_h = np.linspace(thr_gum, 10, 100)
    # y_h = x_h * a_h + (x_h ** b_h) * mu_h
    # ax.plot(x_h, y_h, color="orange", label="U|H")

    # y_u = np.linspace(thr_gum, 10, 100)
    # x_u = y_u * a_u + (y_u ** b_u) * mu_u
    # ax.plot(x_u, y_u, color="teal", label="H|U")

    # ax.scatter(stm_g[:, 0], stm_g[:, 1], s=20, color="black", label="original")
    # ax.axvline(thr_gum, color="black")
    # ax.axhline(thr_gum, color="black")

    # ax.set_xlabel(r"$\hat H_s$")
    # ax.set_ylabel(r"$\hat U$")
    # ax.set_xlim(-2, 10)
    # ax.set_ylim(-2, 10)
    # ax.scatter(
    #     sample_given_hg[:, 0], sample_given_hg[:, 1], s=10, color="orange", label="U|H"
    # )
    # ax.scatter(
    #     sample_given_ug[:, 0], sample_given_ug[:, 1], s=10, color="teal", label="H|U"
    # )
    # print(sample_given_hg.max(), sample_given_ug.max())
    # print(sample_given_hg.min(), sample_given_ug.min())

    # plt.savefig(f"{dir_out}/Simulated_Conmul.png", bbox_inches = 'tight')
    #########################################################

    ##################################################################################################################
    # %%Transform back to original scale

    ppf_h = monotone_fn_inverter(
        f_hat_cdf[0],
        np.linspace(stm_h.min(), gp[0].args[1] - gp[0].args[2] / gp[0].args[0], 10000),
    )
    ppf_u = monotone_fn_inverter(
        f_hat_cdf[1],
        np.linspace(stm_u.min(), gp[1].args[1] - gp[1].args[2] / gp[1].args[0], 10000),
    )

    sample_full = np.zeros(sample_full_g.shape)
    _uni_h = np.exp(-np.exp(-sample_full_g[:, 0]))
    _uni_u = np.exp(-np.exp(-sample_full_g[:, 1]))
    thr_gum_marginal_u = ppf_u(np.exp(-np.exp(-thr_gum)))
    thr_gum_marginal_h = ppf_h(np.exp(-np.exp(-thr_gum)))
    lb_h = f_hat_cdf[0](stm_h.min())
    lb_u = f_hat_cdf[1](stm_u.min())
    ub_h = f_hat_cdf[0](gp[0].args[1] - gp[0].args[2] / gp[0].args[0])
    ub_u = f_hat_cdf[1](gp[1].args[1] - gp[1].args[2] / gp[1].args[0])
    sample_full[:, 0] = ppf_h(np.clip(_uni_h, lb_h, ub_h))
    sample_full[:, 1] = ppf_u(np.clip(_uni_u, lb_u, ub_u))
    # sample_full[:,0] = ppf_h( np.clip(_uni_h, lb_h, ub_h))
    # sample_full[:,1] = ppf_u( np.clip(_uni_u, lb_u, ub_u))
    # sample_full[:,0] = ppf_h(_uni_h)
    # sample_full[:,1] = ppf_u(_uni_u)
    sample_given_h = sample_full[0 : sample_given_hg.shape[0], :]
    sample_given_u = sample_full[sample_given_hg.shape[0] :, :]

    #########################################################
    # fig, ax = plt.subplots(figsize=(8, 6) , dpi=100)
    # fig.set_facecolor("white")
    # ax.scatter(stm[:, 0], stm[:, 1], color="black", s=10)
    # ax.scatter(sample_given_h[:, 0], sample_given_h[:, 1], color="orange", s=1)
    # ax.scatter(sample_given_u[:, 0], sample_given_u[:, 1], color="teal", s=1)
    # ax.set_xlabel(f"{var_name[0]}{unit[0]}")
    # ax.set_ylabel(f"{var_name[1]}{unit[1]}")
    # plt.savefig(f"{dir_out}/Back-transformed.png", bbox_inches = 'tight')
    #########################################################

    #########################################################
    fig, ax = plt.subplots(1, 2, figsize=(8, 3), dpi=100, facecolor="white")

    ax[0].set_aspect(1)

    a_h, b_h, mu_h, sg_h = params_median[:, 0]
    a_u, b_u, mu_u, sg_u = params_median[:, 1]

    x_h = np.linspace(thr_gum, 10, 100)
    y_h = x_h * a_h + (x_h ** b_h) * mu_h
    ax[0].plot(x_h, y_h, color="orange", label="U|H")

    y_u = np.linspace(thr_gum, 10, 100)
    x_u = y_u * a_u + (y_u ** b_u) * mu_u
    ax[0].plot(x_u, y_u, color="teal", label="H|U")

    ax[0].scatter(stm_g[:, 0], stm_g[:, 1], s=10, color="black", label="original")
    ax[0].axvline(thr_gum, color="black")
    ax[0].axhline(thr_gum, color="black")

    ax[0].set_xlabel(r"$\hat H_s$")
    ax[0].set_ylabel(r"$\hat U$")
    ax[0].set_xlim(-2, 10)
    ax[0].set_ylim(-2, 10)
    ax[0].scatter(
        sample_given_hg[:, 0], sample_given_hg[:, 1], s=1, color="orange", label="U|H"
    )
    ax[0].scatter(
        sample_given_ug[:, 0], sample_given_ug[:, 1], s=1, color="teal", label="H|U"
    )
    print(sample_given_hg.max(), sample_given_ug.max())
    print(sample_given_hg.min(), sample_given_ug.min())

    ax[1].scatter(stm[:, 0], stm[:, 1], color="black", s=10)
    ax[1].scatter(sample_given_h[:, 0], sample_given_h[:, 1], color="orange", s=1)
    ax[1].scatter(sample_given_u[:, 0], sample_given_u[:, 1], color="teal", s=1)
    ax[1].set_xlabel(f"{var_name[0]}{unit[0]}")
    ax[1].set_ylabel(f"{var_name[1]}{unit[1]}")

    res = 100
    _x = np.linspace(0, 18, res)
    _y = np.linspace(0, 60, res)
    _x_mg, _y_mg = np.meshgrid(_x, _y)
    _z_mg_sample = np.zeros((res, res))
    _z_mg = np.zeros((res, res))

    for xi in range(res):
        for yi in range(res):
            _count_sample = np.count_nonzero(
                np.logical_and(sample_full[:, 0] > _x[xi], sample_full[:, 1] > _y[yi])
            )
            _count = np.count_nonzero(
                np.logical_and(stm[:, 0] > _x[xi], stm[:, 1] > _y[yi])
            )
            _z_mg_sample[xi, yi] = _count_sample
            _z_mg[xi, yi] = _count
    return_periods = [100]
    occur_prob = 1.04
    exceedance_prob = (
        np.count_nonzero(np.logical_or(stm[:, 0] > thr_org[0], stm[:, 1] > thr_org[1]))
        / stm.shape[0]
    )

    _levels_original = [stm.shape[0] / (rp * occur_prob) for rp in return_periods]
    _levels_sample = [
        sample_full.shape[0] / (rp * occur_prob * exceedance_prob)
        for rp in return_periods
    ]
    _linestyles = ["-", "--"]
    ax[1].contour(
        _x_mg,
        _y_mg,
        _z_mg.T,
        levels=_levels_original,
        linestyles=_linestyles,
        colors="black",
    )
    ax[1].contour(
        _x_mg,
        _y_mg,
        _z_mg_sample.T,
        levels=_levels_sample,
        linestyles=_linestyles,
        colors="red",
    )

    plt.savefig(
        f"{dir_out}/Simulated_Conmul_vs_Back_Transformed.png", bbox_inches="tight"
    )
    #########################################################

    ##################################################################################################################
    # %%Exposure set sampling

    N = 477 * 10
    return_periods = [100]
    lat = latlon[:, 0]
    lon = latlon[:, 1]
    tm_h = (exp_h.T * stm_h).T
    tm_u = (exp_u.T * stm_u).T
    tm = np.stack((tm_h, tm_u), axis=2)
    rng = default_rng()

    tm_sample = np.zeros((N, num_nodes, num_vars))

    # choose random exposure set
    _idx_evt = rng.choice(exp.shape[0], size=N)
    # choose random stm set from sample_full
    _idx_smp = rng.choice(sample_full.shape[0], size=N)

    _exp_h = exp[_idx_evt, :, 0]
    _exp_u = exp[_idx_evt, :, 1]
    _stm_h = sample_full[_idx_smp, 0][np.newaxis].T
    _stm_u = sample_full[_idx_smp, 1][np.newaxis].T

    tm_sample[:, :, 0] = _exp_h * _stm_h
    tm_sample[:, :, 1] = _exp_u * _stm_u

    ##################################################################################################################
    # %%Simple Contour Return Period
    occur_prob = 1.04
    exceedance_prob = (
        np.count_nonzero(np.logical_or(stm_g[:, 0] > thr_gum, stm_g[:, 1] > thr_gum))
        / num_events
    )

    res = 100
    _x = np.linspace(0, 18, res)
    _y = np.linspace(0, 60, res)
    _x_mg, _y_mg = np.meshgrid(_x, _y)

    #########################################################
    fig, axes = plt.subplots(2, 2, figsize=(8, 6), dpi=100, facecolor="white",)
    # fig.suptitle(
    #     f"{return_period}-yr return period (Saint-Denis) Gumbel threshold = {thr_gum}",
    #     y=0.90,
    # )
    fig.supxlabel(r"$H_s$[m]")
    fig.supylabel(r"$U$[m/s]")

    for i, ax in enumerate(axes.flatten()):
        _idx_pos = idx_pos_list_saint_denis[i]
        _z_mg_sample = np.zeros((res, res))
        _z_mg = np.zeros((res, res))

        for xi in range(res):
            for yi in range(res):
                _count_sample = np.count_nonzero(
                    np.logical_and(
                        tm_sample[:, _idx_pos, 0] > _x[xi],
                        tm_sample[:, _idx_pos, 1] > _y[yi],
                    )
                )
                _count = np.count_nonzero(
                    np.logical_and(
                        tm[:, _idx_pos, 0] > _x[xi], tm[:, _idx_pos, 1] > _y[yi]
                    )
                )
                _z_mg_sample[xi, yi] = _count_sample
                _z_mg[xi, yi] = _count
        ax.scatter(
            tm[:, _idx_pos, 0],
            tm[:, _idx_pos, 1],
            s=2,
            c="black",
            label=f"Original",
            alpha=0.3,
        )
        ax.scatter(
            tm_sample[:, _idx_pos, 0],
            tm_sample[:, _idx_pos, 1],
            s=2,
            c=pos_color[i],
            label=f"Simulated",
            alpha=0.1,
        )
        _levels_original = [tm.shape[0] / (rp * occur_prob) for rp in return_periods]
        _levels_sample = [
            tm_sample.shape[0] / (rp * occur_prob * exceedance_prob)
            for rp in return_periods
        ]
        _linestyles = ["-", "--"]
        ax.contour(
            _x_mg,
            _y_mg,
            _z_mg.T,
            levels=_levels_original,
            linestyles=_linestyles,
            colors="black",
        )
        ax.contour(
            _x_mg,
            _y_mg,
            _z_mg_sample.T,
            levels=_levels_sample,
            linestyles=_linestyles,
            colors=pos_color[i],
        )
        ax.set_title(f"Coord.{i+1}")
    plt.savefig(f"{dir_out}/RV_(Saint-Denis).png", bbox_inches="tight")

    # %%output info
    with open(f"{dir_out}/info.txt", "w") as f:
        _sample_h = np.count_nonzero(stm[:, 0] > thr_org[0])
        _sample_u = np.count_nonzero(stm[:, 1] > thr_org[1])
        _sample_hg = np.count_nonzero(stm_g[:, 0] > thr_gum)
        _sample_ug = np.count_nonzero(stm_g[:, 1] > thr_gum)

        f.write(f"Marginal Threshold:\t{thr_org[0]}[m], {thr_org[1]}[m/s]\n")
        f.write(f"Marginal Sample Size:\t{_sample_h},{_sample_u}\n")
        f.write(f"Gumbel Threshold: \t{thr_gum}\n")
        f.write(f"Gumbel Sample Size:\t{_sample_hg},{_sample_ug}\n")
        f.write(
            f"Gumbel Threshold in Marginal Scale:\t{thr_gum_marginal_h:.2f}, {thr_gum_marginal_u:.2f}"
        )
    print("FINISHED")


#%%
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Optional app description")

    parser.add_argument(
        "thr_gum", type=float, help="A required integer positional argument"
    )
    parser.add_argument(
        "thr_hs", type=float, help="A required integer positional argument"
    )
    parser.add_argument(
        "thr_u10", type=float, help="A required integer positional argument"
    )
    # parser.add_argument("--thr_gum_mean", action="store_true")

    args = parser.parse_args()
    # %%Config
    thr_org = np.array([args.thr_hs, args.thr_u10])
    thr_gum = args.thr_gum
    main()
