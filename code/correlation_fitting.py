import os
import sys
import json
import time
import warnings
import numpy as np
import openpyxl
from scipy.optimize import least_squares

# ----------------------------------------------------------------------------- constants
EV = 1.602176634e-19          # J
AMU = 1.66053906660e-27       # kg
KB = 1.380649e-23             # J/K
M_W = 18.01528 * AMU          # kg, one water molecule
Y0 = 25.75e-10                # m, half width of the slit (wall plane)
SIG = 3.166e-10               # m, sigma_ff of SPC/E
EPS_K = 78.2                  # K, eps_ff / kB of SPC/E (0.650 kJ/mol)
EPS_EV = EPS_K * 8.617333262e-5
K_EPS = 6.32                  # eps_wf0 / eps_ff, to be confirmed
U_LIMIT = 100.0               # m/s, upper mean flow speed of the fitted set
RHO0 = 998.0 * SIG**3 / M_W   # reference density, dimensionless (1.0587)
T0 = 293.15 / EPS_K           # reference temperature, dimensionless (3.7487)
W_LOSS = 1.0e-9 / SIG         # 1 nm in units of sigma, scale of the loss function
NM = SIG * 1e9                # sigma in nm
N_REP, N_FOLD, N_BOOT = 5, 5, 2000

def mu_water(T):
    """Viscosity of liquid water at 0.1 MPa in Pa s (correlation for liquid water of Huber et al., J. Phys. Chem. Ref. Data 2009, 38, 101)."""
    tr = np.asarray(T, float) / 300.0
    return 1e-6 * (280.68 * tr**-1.9 + 511.45 * tr**-7.7 + 61.131 * tr**-19.6 + 0.45903 * tr**-40.0)

# ----------------------------------------------------------------------------- data
def load(path):
    wb = openpyxl.load_workbook(path, data_only=True)
    # Sheet2: complete database. The first five columns retain the original
    # fitting variables; the sixth column is the MD mean flow speed ub (m/s).
    rows = [r[:6] for r in wb['Sheet2'].iter_rows(min_row=2, values_only=True) if r[0] is not None]
    data = np.array(rows, float)
    db = data[:, :5]                              # ratio, f (eV/A), T (K), rho (kg/m3), Ls (nm)
    ub = data[:, 5]                               # actual mean flow speed from the raw database
    # Keep the original three corrections; these values agree with the updated single-factor means.
    FIX = {(0.2, 2e-5, 293.15): 1.78058, (0.1, 4e-5, 293.15): 10.45513, (0.1, 1e-5, 338.15): 2.7383}
    for i in range(min(23, len(db))):
        key = (round(db[i, 0], 3), round(db[i, 1], 7), round(db[i, 2], 2))
        if key in FIX and abs(db[i, 3] - 998) < 5:
            db[i, 4] = FIX[key]
    # Sheet1: single-factor conditions and their three repeats, mean and SD.
    # Numeric rows across the blocks follow the same [setting, ..., mean, SD] layout.
    rep = []
    for r in wb['Sheet1'].iter_rows(values_only=True):
        v = r[:6]
        if all(isinstance(c, (int, float)) for c in v):
            rep.append([v[0], v[4], v[5]])
    return db, np.array(rep, float), ub

def groups(db):
    ratio, f, T, rho, Ls = db.T
    eps_s = K_EPS * ratio                         # eps_wf / eps_ff
    rho_s = rho * SIG**3 / M_W                    # rho sigma^3 / m
    T_s = T / EPS_K                               # kB T / eps_ff
    f_s = f * (SIG * 1e10) / EPS_EV               # f sigma / eps_ff, force per atom
    L_s = Ls * 1e-9 / SIG                         # Ls / sigma
    a = 3 * f * EV / 1e-10 / M_W                  # m/s2, force on each of the three atoms
    u_ns = a * rho * Y0**2 / (3 * mu_water(T))    # mean speed of the no slip reference
    u = (1 + 3 * Ls * 1e-9 / Y0) * u_ns           # mean flow speed implied by Ls (Eq. 7 inverted)
    return np.vstack([eps_s, rho_s, T_s, f_s]).T, L_s, u

# ----------------------------------------------------------------------------- correlation
NAMES = ['a', 'b', 'c1', 'c2', 'theta', 'delta0', 'delta1', 'k']
P0 = [0.030, 0.088, 16.5, 2.9, 30.5, 0.74, 7.95, 5.0]
LB = [1e-5, 1e-5, -60, -60, -80, 0.0, -30, 0.0]
UB = [1.0, 50.0, 90, 90, 150, 3.0, 40, 60.0]

def parts(p, X):
    """Returns L0, the force group x and the offset delta."""
    a, b, c1, c2, th, d0, d1, k = p
    eps_s, rho_s, T_s, f_s = X.T
    lam_w = a * np.exp(c1 * (rho_s - RHO0) - th * (1 / T_s - 1 / T0))
    lam_a = b * eps_s**2 * np.exp(c2 * (rho_s - RHO0))
    L0 = 1.0 / (lam_w + lam_a)
    return L0, k * f_s * L0, d0 + d1 * (RHO0 - rho_s)

def corr(p, X):
    """Ls/sigma."""
    L0, x, delta = parts(p, X)
    return L0 * (1 + x**2) - delta

def fit(X, L_s, p0=P0):
    res = lambda p: (corr(p, X) - L_s) / (np.abs(L_s) + W_LOSS)
    return least_squares(res, p0, bounds=(LB, UB), x_scale='jac', max_nfev=40000).x

def scores(Ls_nm, pred_nm):
    y, q = np.asarray(Ls_nm), np.asarray(pred_nm)
    big = y >= 1.0
    out = dict(N=len(y), N_big=int(big.sum()), R2=1 - np.sum((y - q)**2) / np.sum((y - y.mean())**2), RMSE=np.sqrt(np.mean((y - q)**2)),
               MAE=np.mean(np.abs(y - q)), within25=np.mean(np.abs(q[big] - y[big]) <= 0.25 * y[big]) if big.any() else np.nan,
               MAE_small=np.mean(np.abs(q[~big] - y[~big])) if (~big).any() else np.nan)
    return {k: (float(v) if k not in ('N', 'N_big') else v) for k, v in out.items()}

def make_splits(idx, n_rep=N_REP, n_fold=N_FOLD, seed=0):
    rng = np.random.default_rng(seed); out = []
    for _ in range(n_rep):
        ii = idx.copy(); rng.shuffle(ii); out.append(np.array_split(ii, n_fold))
    return out

def main(path, out_xlsx, out_json=None):
    db, rep, ub = load(path)
    ratio, f, T, rho, Ls = db.T
    X, L_s, _ = groups(db)
    u = ub.copy()                              # use the actual mean flow speed in Sheet2
    n = len(db); single = np.arange(n) < 23
    sel = u < U_LIMIT; idx = np.where(sel)[0]
    R = {}

    # ---- fit and rounded constants
    p = fit(X[idx], L_s[idx])
    p_round = np.array([float(f'{v:.3g}') for v in p])
    pred = corr(p_round, X) * NM
    L0, xg, delta = parts(p_round, X)
    R['N_total'] = n; R['N_fit'] = len(idx); R['N_fit_single'] = int((sel & single).sum()); R['N_fit_sampled'] = int((sel & ~single).sum())
    R['p'] = p.tolist(); R['p_round'] = p_round.tolist()
    R['fit_exact'] = scores(Ls[idx], corr(p, X[idx]) * NM); R['fit_round'] = scores(Ls[idx], pred[idx])

    # ---- repeated cross validation (same folds as the machine learning benchmark)
    cv = []; cv_pred = np.full((N_REP, n), np.nan)
    for r_, folds in enumerate(make_splits(idx)):
        for kf in range(N_FOLD):
            te = folds[kf]; tr = np.concatenate([folds[j] for j in range(N_FOLD) if j != kf])
            cv_pred[r_, te] = corr(fit(X[tr], L_s[tr], p0=p), X[te]) * NM
        s = scores(Ls[idx], cv_pred[r_, idx]); cv.append([s['R2'], s['RMSE'], s['MAE'], s['within25'], s['MAE_small']])
    cv = np.array(cv); R['cv_mean'] = cv.mean(0).tolist(); R['cv_sd'] = cv.std(0).tolist()
    with warnings.catch_warnings():                       # runs outside the fitted set have no cross validated value
        warnings.simplefilter('ignore', category=RuntimeWarning)
        cv_mean_pred = np.nanmean(cv_pred, axis=0)

    # ---- bootstrap intervals of the constants
    rng = np.random.default_rng(1); boot = []
    for _ in range(N_BOOT):
        ii = rng.choice(idx, len(idx))
        try: boot.append(fit(X[ii], L_s[ii], p0=p))
        except Exception: pass
    boot = np.array(boot); lo, hi = np.percentile(boot, 2.5, axis=0), np.percentile(boot, 97.5, axis=0)
    R['boot_lo'] = lo.tolist(); R['boot_hi'] = hi.tolist(); R['N_boot'] = len(boot)

    # ---- checks
    # (1) all single factor runs held out
    tr = np.where(sel & ~single)[0]; te = np.where(sel & single)[0]
    ph = fit(X[tr], L_s[tr], p0=p); R['holdout_single'] = scores(Ls[te], corr(ph, X[te]) * NM); R['holdout_single_pred'] = (corr(ph, X[te]) * NM).tolist()
    # (2) one single factor series held out at a time (the base case stays in)
    base = single & (np.abs(ratio - 0.1) < 1e-6) & (np.abs(f - 1e-5) < 1e-9) & (np.abs(T - 293.15) < 0.1) & (np.abs(rho - 998) < 5)
    series = {
        'wall': single & (np.abs(f - 1e-5) < 1e-9) & (np.abs(T - 293.15) < 0.1) & (np.abs(rho - 998) < 5) & ~base,
        'temperature': single & (np.abs(ratio - 0.1) < 1e-6) & (np.abs(f - 1e-5) < 1e-9) & (np.abs(T - 293.15) > 0.1),
        'density': single & (np.abs(ratio - 0.1) < 1e-6) & (np.abs(f - 1e-5) < 1e-9) & (np.abs(T - 293.15) < 0.1) & (np.abs(rho - 998) > 5),
        'force, ratio 0.1': single & (np.abs(ratio - 0.1) < 1e-6) & (f > 1.5e-5),
        'force, ratio 0.2': single & (np.abs(ratio - 0.2) < 1e-6) & (f > 1.5e-5),
    }
    R['series'] = {}
    for name, m in series.items():
        tr = np.where(sel & ~m)[0]; ps = fit(X[tr], L_s[tr], p0=p); q = corr(ps, X) * NM
        key = {'wall': ratio, 'temperature': T, 'density': rho}.get(name, f * 1e5)
        o = np.where(m)[0]; o = o[np.argsort(key[o])]
        R['series'][name] = dict(setting=key[o].tolist(), MD=Ls[o].tolist(), full_fit=pred[o].tolist(), left_out=q[o].tolist(), in_set=sel[o].tolist(), u=u[o].tolist(), x=xg[o].tolist())
    # (3) runs outside the fitted set
    R['outside'] = {}
    for a_, b_ in [(100, 150), (150, 300), (300, 1e9)]:
        m = (u >= a_) & (u < b_)
        R['outside'][f'{a_}-{int(b_) if b_ < 1e8 else "inf"}'] = dict(N=int(m.sum()), median_ratio=float(np.median(pred[m] / Ls[m])), within25=float(np.mean(np.abs(pred[m] - Ls[m]) <= 0.25 * Ls[m])),
                                                                    x_min=float(xg[m].min()), x_median=float(np.median(xg[m])), x_max=float(xg[m].max()))
    R['x_fit'] = dict(min=float(xg[sel].min()), median=float(np.median(xg[sel])), max=float(xg[sel].max()), share_corr_gt10=float(np.mean(xg[sel]**2 > 0.10)), max_corr=float(1 + xg[sel].max()**2))
    R['x_rule'] = {str(t): dict(selected=int((xg < t).sum()), slow=int(((xg < t) & sel).sum()), mid=int(((xg < t) & (u >= 100) & (u < 300)).sum()), fast=int(((xg < t) & (u >= 300)).sum())) for t in (0.8, 0.9, 1.0, 1.2)}
    # (4) ranges of the fitted set and scatter of the repeats
    R['range_fit'] = dict(ratio=[float(ratio[sel].min()), float(ratio[sel].max())], eps_s=[float(X[sel, 0].min()), float(X[sel, 0].max())], rho=[float(rho[sel].min()), float(rho[sel].max())],
                          rho_s=[float(X[sel, 1].min()), float(X[sel, 1].max())], T=[float(T[sel].min()), float(T[sel].max())], T_s=[float(X[sel, 2].min()), float(X[sel, 2].max())],
                          f=[float(f[sel].min()), float(f[sel].max())], f_s=[float(X[sel, 3].min()), float(X[sel, 3].max())], Ls=[float(Ls[sel].min()), float(Ls[sel].max())], u=[float(u[sel].min()), float(u[sel].max())])
    R['u_gap'] = [float(u[sel].max()), float(u[~sel].min())]
    R['N_neg_fit'] = int((Ls[sel] < 0).sum()); R['N_small_fit'] = int((Ls[sel] < 1).sum())
    R['N_ratio_lt_012_sampled'] = int((sel & ~single & (ratio < 0.12)).sum())
    R['rep_rel_sd'] = dict(min=float(np.min(rep[:, 2] / np.abs(rep[:, 1]))), max=float(np.max((rep[:, 2] / np.abs(rep[:, 1]))[rep[:, 1] > 1])), median=float(np.median(rep[:, 2] / np.abs(rep[:, 1]))), max_abs=float(rep[:, 2].max()))
    # physical readings
    R['E_theta_kJmol'] = float(p_round[4] * EPS_K * 8.314462618e-3); R['delta0_nm'] = float(p_round[5] * NM)
    R['fL_at_x1_eV'] = float(EPS_EV / p_round[7])     # f * Ls0 at x = 1, in eV
    n_w = 998.0 / M_W                                  # molecules per m3
    R['tauL_at_x1_Nm'] = float(3 * n_w * Y0 * (EPS_EV * EV / p_round[7]))   # wall shear stress times L0 at x = 1, in N/m

    # ---- print
    print(f'runs in the fitted set: {len(idx)} of {n}  (single factor {R["N_fit_single"]}, sampled {R["N_fit_sampled"]}); speed gap {R["u_gap"][0]:.0f} to {R["u_gap"][1]:.0f} m/s')
    for nme, v, vr, l_, h_ in zip(NAMES, p, p_round, lo, hi):
        print(f'  {nme:7s} = {v:.5g}  (rounded {vr:g}; 95% interval {l_:.4g} to {h_:.4g})')
    fmt = lambda s: f'R2 {s["R2"]:.3f}, RMSE {s["RMSE"]:.2f} nm, MAE {s["MAE"]:.2f} nm, within 25% {100*s["within25"]:.0f}% (N {s["N_big"]}), MAE below 1 nm {s["MAE_small"]:.2f} nm'
    print('fit, rounded constants :', fmt(R['fit_round']))
    print('cross validation       : R2 %.3f (%.3f), RMSE %.2f (%.2f) nm, MAE %.2f (%.2f) nm, within 25%% %.0f (%.0f) %%, MAE below 1 nm %.2f (%.2f) nm' % tuple(np.ravel([[m_, s_] for m_, s_ in zip(cv.mean(0) * [1, 1, 1, 100, 1], cv.std(0) * [1, 1, 1, 100, 1])])))
    print('single factor hold out :', fmt(R['holdout_single']))
    for name, d in R['series'].items():
        print(f'  series {name:17s} setting {np.round(d["setting"], 2)} MD {np.round(d["MD"], 2)} full fit {np.round(d["full_fit"], 2)} left out {np.round(d["left_out"], 2)} in set {d["in_set"]}')
    print('outside the fitted set :', json.dumps(R['outside']))
    print('force group in the set :', json.dumps(R['x_fit']), '| rule:', json.dumps(R['x_rule']))
    print('ranges                 :', json.dumps(R['range_fit']))
    print('other                  :', {k: R[k] for k in ('N_neg_fit', 'N_small_fit', 'N_ratio_lt_012_sampled', 'rep_rel_sd', 'E_theta_kJmol', 'delta0_nm', 'fL_at_x1_eV', 'tauL_at_x1_Nm')})

    # ---- results workbook
    wb = openpyxl.Workbook(); ws = wb.active; ws.title = 'runs'
    ws.append(['run', 'group', 'ratio eps_wf/eps_wf0', 'f (eV/A per atom)', 'T (K)', 'rho (kg/m3)', 'Ls MD (nm)', 'eps*', 'rho*', 'T*', 'f*', 'Ls/sigma',
               'mean flow speed, estimated (m/s)', 'in fitted set', 'L0/sigma', 'force group x', 'delta/sigma', 'Ls correlation (nm)', 'Ls cross validated (nm)'])
    for i in range(n):
        ws.append([i + 1, 'single factor' if single[i] else 'sampled', ratio[i], f[i], T[i], rho[i], Ls[i], X[i, 0], X[i, 1], X[i, 2], X[i, 3], L_s[i], u[i],
                   'yes' if sel[i] else 'no', L0[i], xg[i], delta[i], pred[i], None if np.isnan(cv_mean_pred[i]) else cv_mean_pred[i]])
    ws2 = wb.create_sheet('constants')
    ws2.append(['constant', 'fitted', 'rounded (used in the paper)', '2.5%', '97.5%'])
    for nme, v, vr, l_, h_ in zip(NAMES, p, p_round, lo, hi): ws2.append([nme, v, vr, l_, h_])
    ws2.append([])
    for nme, v in [('rho0*', RHO0), ('T0*', T0), ('eps_wf0/eps_ff', K_EPS), ('sigma_ff (nm)', NM), ('eps_ff/kB (K)', EPS_K), ('speed limit of the fitted set (m/s)', U_LIMIT)]: ws2.append([nme, v])
    ws3 = wb.create_sheet('scores')
    ws3.append(['set', 'N', 'R2', 'RMSE (nm)', 'MAE (nm)', 'share within 25% (Ls >= 1 nm)', 'MAE (Ls < 1 nm)'])
    s = R['fit_round']; ws3.append(['fit, rounded constants', s['N'], s['R2'], s['RMSE'], s['MAE'], s['within25'], s['MAE_small']])
    ws3.append([f'{N_FOLD} fold cross validation, mean of {N_REP} repeats', len(idx)] + cv.mean(0).tolist())
    ws3.append([f'{N_FOLD} fold cross validation, sd of {N_REP} repeats', len(idx)] + cv.std(0).tolist())
    s = R['holdout_single']; ws3.append(['all single factor runs held out', s['N'], s['R2'], s['RMSE'], s['MAE'], s['within25'], s['MAE_small']])
    ws4 = wb.create_sheet('series')
    ws4.append(['series', 'setting', 'Ls MD (nm)', 'Ls correlation (nm)', 'Ls with the series left out (nm)', 'in fitted set', 'mean flow speed (m/s)', 'force group x'])
    for name, d in R['series'].items():
        for j in range(len(d['MD'])): ws4.append([name, d['setting'][j], d['MD'][j], d['full_fit'][j], d['left_out'][j], 'yes' if d['in_set'][j] else 'no', d['u'][j], d['x'][j]])
    wb.save(out_xlsx)
    if out_json:
        R['arrays'] = dict(ratio=ratio.tolist(), f=f.tolist(), T=T.tolist(), rho=rho.tolist(), Ls=Ls.tolist(), u=u.tolist(), sel=sel.tolist(), single=single.tolist(), pred=pred.tolist(),
                           cv_pred=[None if np.isnan(v) else float(v) for v in cv_mean_pred], x=xg.tolist(), L0=L0.tolist(), delta=delta.tolist(), rep=rep.tolist())
        json.dump(R, open(out_json, 'w'), indent=1)
    return R

if __name__ == '__main__':
    args = sys.argv[1:4]
    if not args:                                              # no arguments: files of this folder, output marked as a rerun
        here = os.path.dirname(os.path.abspath(__file__)); stamp = time.strftime('%Y%m%d')
        project = os.path.dirname(here)
        out_dir = os.path.join(project, 'results')
        os.makedirs(out_dir, exist_ok=True)
        args = [os.path.join(project, 'data', 'slip_length_data.xlsx'), os.path.join(out_dir, f'correlation_results_{stamp}.xlsx'), os.path.join(out_dir, f'correlation_results_{stamp}.json')]
    main(*args)
    print('saved', os.path.basename(args[1]))
