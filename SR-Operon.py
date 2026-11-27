import os
import json
import time
import warnings
import numpy as np
import openpyxl
from openpyxl.styles import Font, PatternFill, Alignment
from pyoperon.sklearn import SymbolicRegressor

warnings.filterwarnings('ignore')
HERE = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(HERE, 'slip_length_data.xlsx')

# Constants and preprocessing copied from the standalone ML benchmark.
AMU = 1.66053906660e-27
M_W = 18.01528 * AMU
SIG = 3.166e-10
EPS_K = 78.2
EPS_EV = EPS_K * 8.617333262e-5
K_EPS = 6.32
U_LIMIT = 100.0
NM = SIG * 1e9
N_REP, N_FOLD = 5, 5


def load_data(path):
    wb = openpyxl.load_workbook(path, data_only=True, read_only=True)
    if 'Sheet2' not in wb.sheetnames:
        raise ValueError('原始数据工作簿中找不到 Sheet2。')
    rows = [r[:6] for r in wb['Sheet2'].iter_rows(min_row=2, values_only=True)
            if r[0] is not None]
    data = np.asarray(rows, dtype=float)
    if data.ndim != 2 or data.shape[1] < 6:
        raise ValueError('Sheet2 必须包含6列：ratio、f、T、rho、Ls、ub。')
    db = data[:, :5].copy()
    ub = data[:, 5].copy()
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


def reduced_variables(db):
    ratio, force, temp, rho, ls_nm = db.T
    eps_star = K_EPS * ratio
    rho_star = rho * SIG**3 / M_W
    temp_star = temp / EPS_K
    force_star = force * (SIG * 1e10) / EPS_EV
    ls_star = ls_nm * 1e-9 / SIG
    X = np.column_stack([eps_star, rho_star, temp_star, force_star])
    return X, ls_star


def make_splits(idx, seed=0):
    rng = np.random.default_rng(seed)
    result = []
    for _ in range(N_REP):
        shuffled = idx.copy()
        rng.shuffle(shuffled)
        result.append(np.array_split(shuffled, N_FOLD))
    return result


def make_sr(max_length, seed):
    # Same Operon configuration as the original ML benchmark.
    return SymbolicRegressor(
        allowed_symbols='add,sub,mul,div,exp,square,constant,variable',
        max_length=max_length,
        max_depth=8,
        generations=300,
        population_size=1000,
        max_evaluations=600000,
        n_threads=1,
        random_state=seed,
        optimizer_iterations=10,
        objectives=['r2', 'length'],
        model_selection_criterion='minimum_description_length',
        max_time=60,
    )


def scores(y_nm, pred_nm):
    y = np.asarray(y_nm, dtype=float)
    q = np.asarray(pred_nm, dtype=float)
    big = y >= 1.0
    return {
        'R2': float(1 - np.sum((y - q)**2) / np.sum((y - y.mean())**2)),
        'RMSE': float(np.sqrt(np.mean((y - q)**2))),
        'MAE': float(np.mean(np.abs(y - q))),
        'within25': float(np.mean(np.abs(q[big] - y[big]) <= 0.25 * y[big])) if big.any() else float('nan'),
        'MAE_small': float(np.mean(np.abs(q[~big] - y[~big]))) if (~big).any() else float('nan'),
    }


def main():
    if not os.path.isfile(DATA_PATH):
        raise FileNotFoundError(f'找不到 {DATA_PATH}，请将 slip_length_data.xlsx 与本脚本放在同一文件夹。')

    db, ub = load_data(DATA_PATH)
    X, y_star = reduced_variables(db)
    Ls_nm = db[:, 4]
    n = len(db)
    selected = np.isfinite(ub) & (ub < U_LIMIT)
    idx = np.where(selected)[0]
    if len(idx) < N_FOLD:
        raise ValueError('满足实际 ub < 100 m/s 的数据不足以进行5折交叉验证。')

    splits = make_splits(idx)
    feature_names = ['eps_star', 'rho_star', 'T_star', 'f_star']
    all_results = {}

    print(f'总数据：{n}；实际 ub < {U_LIMIT:g} m/s：{len(idx)}')
    print('每个复杂度：5折交叉验证，重复5次；随后用全部筛选数据训练最终公式。')

    for max_length in (10, 15, 20):
        print(f'\n开始 Operon max_length={max_length}', flush=True)
        repeat_scores = []
        for repeat_i, folds in enumerate(splits):
            pred_nm = np.full(n, np.nan)
            for fold_i in range(N_FOLD):
                test_idx = np.asarray(folds[fold_i], dtype=int)
                train_idx = np.concatenate([np.asarray(folds[j], dtype=int)
                                            for j in range(N_FOLD) if j != fold_i])
                seed = 100 * repeat_i + fold_i
                model = make_sr(max_length, seed)
                model.fit(X[train_idx], y_star[train_idx])
                pred_nm[test_idx] = model.predict(X[test_idx]) * NM
                print(f'  repeat {repeat_i + 1}/{N_REP}, fold {fold_i + 1}/{N_FOLD} 完成', flush=True)
            repeat_scores.append(scores(Ls_nm[idx], pred_nm[idx]))

        # Match original ML-benchmark.py: final all-selected-data model uses seed=7.
        final_model = make_sr(max_length, 7)
        final_model.fit(X[idx], y_star[idx])
        # Use the same expression-extraction API as the original ML-benchmark.py.
        # SymbolicRegressor does not provide get_best() in this installed wrapper.
        equation = str(final_model.get_model_string(final_model.model_, 6))
        equation_complexity = None
        equation_loss = None

        metric_names = ['R2', 'RMSE', 'MAE', 'within25', 'MAE_small']
        means = {k: float(np.mean([r[k] for r in repeat_scores])) for k in metric_names}
        sds = {k: float(np.std([r[k] for r in repeat_scores])) for k in metric_names}
        all_results[str(max_length)] = {
            'max_length': max_length,
            'cv_mean': means,
            'cv_sd': sds,
            'repeat_results': repeat_scores,
            'final_equation': equation,
            'final_equation_complexity': equation_complexity,
            'final_equation_loss': equation_loss,
            'final_equation_seed': 7,
        }
        print(f"max_length={max_length}: R2={means['R2']:.4f} ± {sds['R2']:.4f}; "
              f"RMSE={means['RMSE']:.4f} nm; MAE={means['MAE']:.4f} nm; "
              f"within25={means['within25']*100:.1f}%")
        print('最终公式：', equation)

    stamp = time.strftime('%Y%m%d_%H%M%S')
    xlsx_path = os.path.join(HERE, f'Operon_results_and_equations_{stamp}.xlsx')
    txt_path = os.path.join(HERE, f'Operon_final_equations_{stamp}.txt')
    json_path = os.path.join(HERE, f'Operon_results_and_equations_{stamp}.json')

    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = 'Summary_and_equations'
    ws.append(['max_length', 'CV_R2_mean', 'CV_R2_SD', 'CV_RMSE_mean_nm', 'CV_RMSE_SD_nm',
               'CV_MAE_mean_nm', 'CV_MAE_SD_nm', 'within25_mean', 'within25_SD',
               'MAE_small_mean_nm', 'MAE_small_SD_nm', 'final_equation_complexity',
               'final_equation_loss', 'final_equation'])
    for key in ('10', '15', '20'):
        r = all_results[key]
        ws.append([r['max_length'], r['cv_mean']['R2'], r['cv_sd']['R2'],
                   r['cv_mean']['RMSE'], r['cv_sd']['RMSE'],
                   r['cv_mean']['MAE'], r['cv_sd']['MAE'],
                   r['cv_mean']['within25'], r['cv_sd']['within25'],
                   r['cv_mean']['MAE_small'], r['cv_sd']['MAE_small'],
                   r['final_equation_complexity'], r['final_equation_loss'], r['final_equation']])
    ws.freeze_panes = 'A2'
    ws.auto_filter.ref = ws.dimensions
    for cell in ws[1]:
        cell.font = Font(bold=True, color='FFFFFF')
        cell.fill = PatternFill('solid', fgColor='1F4E78')
        cell.alignment = Alignment(wrap_text=True, vertical='center')
    ws.row_dimensions[1].height = 42
    for col in ws.columns:
        letter = col[0].column_letter
        ws.column_dimensions[letter].width = 42 if letter == 'N' else 19

    cv_ws = wb.create_sheet('CV_repeats')
    cv_ws.append(['max_length', 'repeat', 'R2', 'RMSE_nm', 'MAE_nm', 'within25', 'MAE_small_nm'])
    for key in ('10', '15', '20'):
        for i, r in enumerate(all_results[key]['repeat_results'], start=1):
            cv_ws.append([int(key), i, r['R2'], r['RMSE'], r['MAE'], r['within25'], r['MAE_small']])
    cv_ws.freeze_panes = 'A2'
    cv_ws.auto_filter.ref = cv_ws.dimensions
    for cell in cv_ws[1]:
        cell.font = Font(bold=True, color='FFFFFF')
        cell.fill = PatternFill('solid', fgColor='1F4E78')

    info = wb.create_sheet('Run_info')
    info.append(['item', 'value'])
    for row in [
        ('data_file', os.path.basename(DATA_PATH)),
        ('total_samples', n),
        ('selected_samples', len(idx)),
        ('selection_rule', 'actual ub from Sheet2 column 6 < 100 m/s'),
        ('single_factor_convention', 'first 23 rows'),
        ('input_variables', ', '.join(feature_names)),
        ('target', 'Ls/sigma'),
        ('CV', '5-fold repeated 5 times; same shuffle/split seed convention as ML-benchmark.py'),
        ('Operon final model seed', 7),
        ('note', 'Final equations are fitted on all selected samples; CV metrics are out-of-fold performance.'),
    ]:
        info.append(list(row))
    info.column_dimensions['A'].width = 28
    info.column_dimensions['B'].width = 90
    wb.save(xlsx_path)

    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write(f'Data: {os.path.basename(DATA_PATH)}\n')
        f.write(f'Selected: {len(idx)} / {n}; actual ub < {U_LIMIT:g} m/s\n')
        f.write('Input: eps_star, rho_star, T_star, f_star; output: Ls/sigma\n\n')
        for key in ('10', '15', '20'):
            r = all_results[key]
            f.write(f"max_length={key}\n{r['final_equation']}\n\n")

    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump({'data_file': os.path.basename(DATA_PATH), 'n_total': n,
                   'n_selected': len(idx), 'selection': 'actual ub < 100 m/s',
                   'models': all_results}, f, ensure_ascii=False, indent=2)

    print('\n已保存：')
    print(xlsx_path)
    print(txt_path)
    print(json_path)


if __name__ == '__main__':
    main()
