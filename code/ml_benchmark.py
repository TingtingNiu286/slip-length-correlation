import os
import sys
import json
import time
import warnings
import numpy as np
import openpyxl
from openpyxl.styles import Font, PatternFill, Alignment
from scipy.optimize import least_squares
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV, KFold
from xgboost import XGBRegressor

warnings.filterwarnings('ignore')
HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(HERE)
DEFAULT_DATA_PATH = os.path.join(PROJECT_DIR, 'data', 'slip_length_data.xlsx')
RESULTS_DIR = os.path.join(PROJECT_DIR, 'results')

# ---------------- Step 1 constants/functions reproduced locally ----------------
EV = 1.602176634e-19
AMU = 1.66053906660e-27
KB = 1.380649e-23
M_W = 18.01528 * AMU
SIG = 3.166e-10
EPS_K = 78.2
EPS_EV = EPS_K * 8.617333262e-5
K_EPS = 6.32
U_LIMIT = 100.0
RHO0 = 998.0 * SIG**3 / M_W
T0 = 293.15 / EPS_K
W_LOSS = 1.0e-9 / SIG
NM = SIG * 1e9
N_REP, N_FOLD = 5, 5
P0 = np.array([0.030, 0.088, 16.5, 2.9, 30.5, 0.74, 7.95, 5.0], dtype=float)
LB = np.array([1e-5, 1e-5, -60, -60, -80, 0.0, -30, 0.0], dtype=float)
UB = np.array([1.0, 50.0, 90, 90, 150, 3.0, 40, 60.0], dtype=float)


def load_data(path):
    wb = openpyxl.load_workbook(path, data_only=True, read_only=True)
    if 'Sheet2' not in wb.sheetnames:
        raise ValueError("原始数据工作簿中找不到 Sheet2。")
    rows = [r[:6] for r in wb['Sheet2'].iter_rows(min_row=2, values_only=True)
            if r[0] is not None]
    data = np.asarray(rows, dtype=float)
    if data.ndim != 2 or data.shape[1] < 6:
        raise ValueError('Sheet2 必须包含6列：ratio、f、T、rho、Ls、ub。')
    db = data[:, :5].copy()  # ratio, f, T, rho, Ls
    ub = data[:, 5].copy()  # actual mean flow speed, m/s

    # Preserve the three Ls corrections used by the supplied Step 1 script.
    fixes = {
        (0.2, 2e-5, 293.15): 1.78058,
        (0.1, 4e-5, 293.15): 10.45513,
        (0.1, 1e-5, 338.15): 2.7383,
    }
    for i in range(min(23, len(db))):
        key = (round(db[i, 0], 3), round(db[i, 1], 7), round(db[i, 2], 2))
        if key in fixes and abs(db[i, 3] - 998) < 5:
            db[i, 4] = fixes[key]
    wb.close()
    return db, ub


def groups(db):
    ratio, force, temp, rho, ls_nm = db.T
    eps_star = K_EPS * ratio
    rho_star = rho * SIG**3 / M_W
    temp_star = temp / EPS_K
    force_star = force * (SIG * 1e10) / EPS_EV
    ls_star = ls_nm * 1e-9 / SIG
    X = np.column_stack([eps_star, rho_star, temp_star, force_star])
    return X, ls_star


def parts(p, X):
    a, b, c1, c2, theta, delta0, delta1, k = p
    eps_star, rho_star, temp_star, force_star = X.T
    lam_w = a * np.exp(c1 * (rho_star - RHO0) - theta * (1 / temp_star - 1 / T0))
    lam_a = b * eps_star**2 * np.exp(c2 * (rho_star - RHO0))
    L0 = 1.0 / (lam_w + lam_a)
    x = k * force_star * L0
    delta = delta0 + delta1 * (RHO0 - rho_star)
    return L0, x, delta


def corr(p, X):
    L0, x, delta = parts(p, X)
    return L0 * (1 + x**2) - delta


def fit_corr(X, Ls_star, p0=P0):
    residual = lambda p: (corr(p, X) - Ls_star) / (np.abs(Ls_star) + W_LOSS)
    return least_squares(residual, p0, bounds=(LB, UB), x_scale='jac', max_nfev=40000).x


def scores(y_nm, pred_nm):
    y = np.asarray(y_nm, dtype=float)
    q = np.asarray(pred_nm, dtype=float)
    big = y >= 1.0
    out = {
        'R2': 1 - np.sum((y - q)**2) / np.sum((y - y.mean())**2),
        'RMSE': np.sqrt(np.mean((y - q)**2)),
        'MAE': np.mean(np.abs(y - q)),
        'within25': np.mean(np.abs(q[big] - y[big]) <= 0.25 * y[big]) if big.any() else np.nan,
        'MAE_small': np.mean(np.abs(q[~big] - y[~big])) if (~big).any() else np.nan,
    }
    return out


def make_splits(idx, n_rep=N_REP, n_fold=N_FOLD, seed=0):
    rng = np.random.default_rng(seed)
    result = []
    for _ in range(n_rep):
        shuffled = idx.copy()
        rng.shuffle(shuffled)
        result.append(np.array_split(shuffled, n_fold))
    return result

# ---------------- ML model definitions: original grids retained ----------------
def make_model(name, quick):
    inner = KFold(4, shuffle=True, random_state=0)
    gs = lambda est, grid: GridSearchCV(est, grid, cv=inner,
                                         scoring='neg_mean_squared_error', n_jobs=2)
    if name == 'RF':
        grid = {'max_depth': [None, 6, 10], 'min_samples_leaf': [1, 2, 4],
                'max_features': [1.0, 0.75, 0.5]}
        if quick:
            grid = {'max_depth': [6], 'min_samples_leaf': [2], 'max_features': [1.0]}
        return gs(RandomForestRegressor(n_estimators=300, random_state=0), grid)
    if name == 'XGBoost':
        grid = {'max_depth': [2, 3, 4], 'learning_rate': [0.03, 0.1],
                'min_child_weight': [1, 3]}
        if quick:
            grid = {'max_depth': [2], 'learning_rate': [0.1], 'min_child_weight': [1]}
        return gs(XGBRegressor(n_estimators=400, random_state=0, verbosity=0,
                               subsample=0.9, n_jobs=1), grid)
    if name == 'AdaBoost':
        grid = {'estimator': [DecisionTreeRegressor(max_depth=d) for d in (3, 5, 8)],
                'learning_rate': [0.1, 0.5, 1.0]}
        if quick:
            grid = {'estimator': [DecisionTreeRegressor(max_depth=3)], 'learning_rate': [1.0]}
        return gs(AdaBoostRegressor(random_state=0, n_estimators=200), grid)
    if name == 'MLP':
        grid = {'mlpregressor__hidden_layer_sizes': [(8,), (16,), (16, 16)],
                'mlpregressor__alpha': [1e-2, 1e-1, 1.0],
                'mlpregressor__activation': ['tanh', 'relu']}
        if quick:
            grid = {'mlpregressor__hidden_layer_sizes': [(16, 16)],
                    'mlpregressor__alpha': [1.0], 'mlpregressor__activation': ['tanh']}
        return gs(make_pipeline(StandardScaler(),
                                MLPRegressor(max_iter=3000, random_state=0, solver='lbfgs')), grid)
    raise ValueError(name)


def make_sr(length, seed):
    from pyoperon.sklearn import SymbolicRegressor
    return SymbolicRegressor(allowed_symbols='add,sub,mul,div,exp,square,constant,variable',
        max_length=length, max_depth=8, generations=300, population_size=1000,
        max_evaluations=600000, n_threads=1, random_state=seed,
        optimizer_iterations=10, objectives=['r2', 'length'],
        model_selection_criterion='minimum_description_length', max_time=60)

# ---------------- Excel output ----------------
def save_predictions_xlsx(path, rows, model_predictions, summaries, selection_info):
    wb = openpyxl.Workbook()
    ws0 = wb.active
    ws0.title = '说明与统计'
    ws0.append(['项目', '数值/说明'])
    for k, v in selection_info.items():
        ws0.append([k, v])
    ws0.append([])
    ws0.append(['模型', 'CV R2 mean', 'CV R2 SD', 'CV RMSE mean (nm)',
                'CV MAE mean (nm)', 'within ±25% mean', 'MAE for Ls<1 nm mean'])
    for name, s in summaries.items():
        ws0.append([name] + s['cv_mean'] + s['cv_sd'])

    for model_name, pred_info in model_predictions.items():
        ws = wb.create_sheet(model_name[:31])
        ws.append(['Excel行号(Sheet2)', '工况类别', 'eps_wf/eps_wf0', 'f (eV/A)',
                   'T (K)', 'rho (kg/m3)', 'ub实际值 (m/s)', '是否参与训练/CV',
                   'Ls实际值 (nm)', 'CV预测均值 (nm)', 'CV残差(预测-实际) (nm)',
                   'CV绝对误差 (nm)', '全筛选集训练后预测 (nm)'])
        cvp = pred_info['cv_mean']
        fullp = pred_info['full_fit']
        for i, row in enumerate(rows):
            ratio, force, temp, rho, ub, ls, selected, single = row
            cv_value = None if not selected or not np.isfinite(cvp[i]) else float(cvp[i])
            residual = None if cv_value is None else cv_value - ls
            ws.append([i + 2, '单因素' if single else '随机采样', ratio, force, temp, rho, ub,
                       '是' if selected else '否', ls, cv_value, residual,
                       None if residual is None else abs(residual),
                       float(fullp[i]) if np.isfinite(fullp[i]) else None])
        ws.freeze_panes = 'A2'
        ws.auto_filter.ref = ws.dimensions
        for cell in ws[1]:
            cell.font = Font(bold=True, color='FFFFFF')
            cell.fill = PatternFill('solid', fgColor='1F4E78')
            cell.alignment = Alignment(wrap_text=True, vertical='center')
        ws.row_dimensions[1].height = 34
        for col in ws.columns:
            letter = col[0].column_letter
            max_len = max(len(str(c.value)) if c.value is not None else 0 for c in col[:100])
            ws.column_dimensions[letter].width = min(max(max_len + 2, 12), 26)

    for cell in ws0[1]:
        cell.font = Font(bold=True, color='FFFFFF')
        cell.fill = PatternFill('solid', fgColor='1F4E78')
    ws0.column_dimensions['A'].width = 34
    ws0.column_dimensions['B'].width = 82
    ws0.freeze_panes = 'A2'
    wb.save(path)


def main(data_path, quick=False):
    db, ub = load_data(data_path)
    ratio, force, temp, rho, Ls = db.T
    X, Ls_star = groups(db)
    n = len(db)
    if len(ub) != n:
        raise ValueError('Sheet2 的 ub 数量与数据行数不一致。')

    single = np.arange(n) < 23  # original workbook convention: first 23 rows are single-factor runs
    selected = np.isfinite(ub) & (ub < U_LIMIT)
    idx = np.where(selected)[0]
    if len(idx) < N_FOLD:
        raise ValueError(f'满足实际 ub < {U_LIMIT:g} m/s 的数据不足以进行交叉验证。')

    n_rep = 1 if quick else N_REP
    splits = make_splits(idx, n_rep=N_REP)[:n_rep]
    X_ml = np.column_stack([np.log(X[:, 0]), X[:, 1], X[:, 2], X[:, 3]])
    metric_key = lambda s: [s['R2'], s['RMSE'], s['MAE'], s['within25'], s['MAE_small']]
    summaries, model_predictions = {}, {}
    context = {'seed': 0}
    rows = [(float(ratio[i]), float(force[i]), float(temp[i]), float(rho[i]), float(ub[i]),
             float(Ls[i]), bool(selected[i]), bool(single[i])) for i in range(n)]
    rows = list(rows)

    def run(name, fit_predict, final_predict):
        cv_scores = []
        cv_all = np.full((n_rep, n), np.nan)
        for rep_i, folds in enumerate(splits):
            pred = np.full(n, np.nan)
            for fold_i in range(N_FOLD):
                test_idx = folds[fold_i]
                train_idx = np.concatenate([folds[j] for j in range(N_FOLD) if j != fold_i])
                context['seed'] = 100 * rep_i + fold_i
                pred[test_idx] = fit_predict(train_idx, test_idx)
            cv_all[rep_i] = pred
            cv_scores.append(metric_key(scores(Ls[idx], pred[idx])))
        cv_scores = np.asarray(cv_scores)
        cv_mean = np.nanmean(cv_all, axis=0)

        train_hold = np.where(selected & ~single)[0]
        test_hold = np.where(selected & single)[0]
        context['seed'] = 9
        hold_pred = fit_predict(train_hold, test_hold) if len(test_hold) and len(train_hold) else np.array([])
        hold_score = scores(Ls[test_hold], hold_pred) if len(test_hold) else {'R2': np.nan}

        full_pred = final_predict(idx, np.arange(n))
        summaries[name] = {
            'cv_mean': cv_scores.mean(axis=0).tolist(),
            'cv_sd': cv_scores.std(axis=0).tolist(),
            'holdout_R2': float(hold_score['R2']),
        }
        model_predictions[name] = {'cv_mean': cv_mean, 'full_fit': full_pred, 'cv_repeats': cv_all}
        print(f'{name:30s} CV R2 {cv_scores[:,0].mean():.3f} ({cv_scores[:,0].std():.3f}); '
              f'RMSE {cv_scores[:,1].mean():.3f} nm; MAE {cv_scores[:,2].mean():.3f} nm; '
              f'within25 {100*cv_scores[:,3].mean():.1f}%; holdout-single R2 {hold_score["R2"]:.3f}', flush=True)

    for model_name in ['RF', 'XGBoost', 'AdaBoost', 'MLP']:
        def fp(train_idx, test_idx, name=model_name):
            estimator = make_model(name, quick)
            estimator.fit(X_ml[train_idx], Ls[train_idx])
            return estimator.predict(X_ml[test_idx])
        def final_fp(train_idx, test_idx, name=model_name):
            estimator = make_model(name, quick)
            estimator.fit(X_ml[train_idx], Ls[train_idx])
            return estimator.predict(X_ml[test_idx])
        run(model_name, fp, final_fp)

    def fp_nof(train_idx, test_idx):
        estimator = make_model('MLP', quick)
        estimator.fit(X_ml[train_idx][:, :3], Ls[train_idx])
        return estimator.predict(X_ml[test_idx][:, :3])
    def final_nof(train_idx, test_idx):
        estimator = make_model('MLP', quick)
        estimator.fit(X_ml[train_idx][:, :3], Ls[train_idx])
        return estimator.predict(X_ml[test_idx][:, :3])
    run('MLP_without_force', fp_nof, final_nof)

    try:
        import pyoperon  # noqa: F401
        for length in ([10] if quick else [10, 15, 20]):
            def fp_sr(train_idx, test_idx, length=length):
                reg = make_sr(length, context['seed'])
                reg.fit(X[train_idx], Ls_star[train_idx])
                return reg.predict(X[test_idx]) * NM
            def final_sr(train_idx, test_idx, length=length):
                reg = make_sr(length, 7)
                reg.fit(X[train_idx], Ls_star[train_idx])
                return reg.predict(X[test_idx]) * NM
            run(f'SymbolicRegression_{length}', fp_sr, final_sr)
    except ImportError:
        print('pyoperon is not installed: symbolic regression skipped')

    p_all = fit_corr(X[idx], Ls_star[idx])
    def fp_corr(train_idx, test_idx):
        return corr(fit_corr(X[train_idx], Ls_star[train_idx], p0=p_all), X[test_idx]) * NM
    def final_corr(train_idx, test_idx):
        return corr(p_all, X[test_idx]) * NM
    run('Friction_correlation', fp_corr, final_corr)

    stamp = time.strftime('%Y%m%d_%H%M%S')
    out_dir = RESULTS_DIR
    os.makedirs(out_dir, exist_ok=True)
    xlsx_path = os.path.join(out_dir, f'ML_actual_vs_predicted_{stamp}.xlsx')
    save_predictions_xlsx(xlsx_path, rows, model_predictions, summaries, {
        'Raw workbook': os.path.basename(data_path),
        'Speed source': 'Sheet2, column 6: ub(m/s), read directly from the raw Excel workbook',
        'Selection rule': f'actual ub < {U_LIMIT:g} m/s',
        'Total samples': n,
        'Selected samples': len(idx),
        'Excluded samples': n - len(idx),
        'Single-factor convention': 'first 23 rows of Sheet2',
        'Cross-validation': f'{N_FOLD}-fold, repeated {n_rep} time(s); predictions averaged across repeats',
        'CV prediction': 'Out-of-fold prediction; use this column to assess predictive performance',
        'Full-fit prediction': 'Model fitted on all selected rows; in-sample for selected rows and extrapolation for excluded rows; not an unbiased test prediction',
        'Slip-length correction': 'The same three single-factor Ls corrections as the supplied Step 1 script are applied.',
        'Standalone': 'All preprocessing, reduced variables, correlation and scoring functions are implemented in this file; no Step 1 script is required.',
    })
    json_path = os.path.join(out_dir, f'ML_benchmark_summary_{stamp}.json')
    with open(json_path, 'w', encoding='utf-8') as fjson:
        json.dump(summaries, fjson, indent=2)
    print('\nSaved prediction workbook:', xlsx_path)
    print('Saved benchmark summary:', json_path)
    print(f'Actual-speed selection: {len(idx)} / {n} rows satisfy ub < {U_LIMIT:g} m/s')
    return summaries


if __name__ == '__main__':
    args = [a for a in sys.argv[1:] if a != '--quick']
    quick = '--quick' in sys.argv
    data_path = args[0] if args else DEFAULT_DATA_PATH
    main(data_path, quick=quick)
