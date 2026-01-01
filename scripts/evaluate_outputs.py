import os
import pandas as pd
import math
import re

GT_DIR = os.path.join('workspace', 'ss thesis', 'gt')
PRED_DIRS = {
    'no_trocr': os.path.join('outputs', 'no_trocr_v2'),
    'trocr': os.path.join('outputs', 'trocr'),
}


def normalize(s):
    if pd.isna(s):
        return ''
    s = str(s)
    s = s.strip()
    s = re.sub(r"\s+", ' ', s)
    return s


def levenshtein(a, b):
    if a == b:
        return 0
    la, lb = len(a), len(b)
    if la == 0:
        return lb
    if lb == 0:
        return la
    prev = list(range(lb + 1))
    for i, ca in enumerate(a, start=1):
        cur = [i] + [0] * lb
        for j, cb in enumerate(b, start=1):
            cost = 0 if ca == cb else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
        prev = cur
    return prev[lb]


def word_error_rate(gt, pred):
    gw = gt.split()
    pw = pred.split()
    dist = levenshtein(' '.join(gw), ' '.join(pw)) if False else levenshtein_words(gw, pw)
    denom = max(len(gw), 1)
    return dist / denom


def levenshtein_words(a_words, b_words):
    # compute edit distance on word tokens
    la, lb = len(a_words), len(b_words)
    if la == 0:
        return lb
    if lb == 0:
        return la
    dp = [[0] * (lb + 1) for _ in range(la + 1)]
    for i in range(la + 1):
        dp[i][0] = i
    for j in range(lb + 1):
        dp[0][j] = j
    for i in range(1, la + 1):
        for j in range(1, lb + 1):
            cost = 0 if a_words[i-1] == b_words[j-1] else 1
            dp[i][j] = min(dp[i-1][j] + 1, dp[i][j-1] + 1, dp[i-1][j-1] + cost)
    return dp[la][lb]


def evaluate_pair(gt_path, pred_path):
    df_gt = pd.read_excel(gt_path, header=None)
    df_pred = pd.read_excel(pred_path, header=None)

    max_rows = max(df_gt.shape[0], df_pred.shape[0])
    max_cols = max(df_gt.shape[1] if df_gt.shape[1]>0 else 0, df_pred.shape[1] if df_pred.shape[1]>0 else 0)

    # pad
    df_gt = df_gt.reindex(index=range(max_rows), columns=range(max_cols))
    df_pred = df_pred.reindex(index=range(max_rows), columns=range(max_cols))

    total_cells = 0
    exact_matches = 0
    sum_cer = 0.0
    sum_wer = 0.0
    cells_with_gt = 0

    for i in range(max_rows):
        for j in range(max_cols):
            gt_cell = normalize(df_gt.iat[i, j])
            pred_cell = normalize(df_pred.iat[i, j])
            total_cells += 1
            if gt_cell == pred_cell:
                exact_matches += 1
            # For CER/WER we consider GT cell (even if empty)
            cells_with_gt += 1
            lev = levenshtein(gt_cell, pred_cell)
            cer = lev / max(len(gt_cell), 1)
            sum_cer += cer
            # WER
            wer = levenshtein_words(gt_cell.split(), pred_cell.split()) / max(len(gt_cell.split()), 1)
            sum_wer += wer

    return {
        'total_cells': total_cells,
        'exact_matches': exact_matches,
        'cell_acc': exact_matches / total_cells if total_cells else 0.0,
        'avg_cer': sum_cer / cells_with_gt if cells_with_gt else 0.0,
        'avg_wer': sum_wer / cells_with_gt if cells_with_gt else 0.0,
    }


def evaluate_folder(gt_dir, pred_dir):
    results = {}
    files = [f for f in os.listdir(gt_dir) if f.lower().endswith('.xlsx')]
    files.sort()
    agg = {'total_cells':0, 'exact_matches':0, 'sum_cer':0.0, 'sum_wer':0.0, 'files_evaluated':0}
    per_file = {}
    for f in files:
        gt_path = os.path.join(gt_dir, f)
        pred_path = os.path.join(pred_dir, f)
        if not os.path.exists(pred_path):
            print(f'Warning: prediction missing for {f} in {pred_dir}')
            continue
        res = evaluate_pair(gt_path, pred_path)
        per_file[f] = res
        agg['total_cells'] += res['total_cells']
        agg['exact_matches'] += res['exact_matches']
        agg['sum_cer'] += res['avg_cer'] * res['total_cells']
        agg['sum_wer'] += res['avg_wer'] * res['total_cells']
        agg['files_evaluated'] += 1

    if agg['total_cells'] > 0:
        overall = {
            'cell_acc': agg['exact_matches'] / agg['total_cells'],
            'avg_cer': agg['sum_cer'] / agg['total_cells'],
            'avg_wer': agg['sum_wer'] / agg['total_cells'],
            'files_evaluated': agg['files_evaluated'],
            'total_cells': agg['total_cells']
        }
    else:
        overall = {'cell_acc':0.0,'avg_cer':0.0,'avg_wer':0.0,'files_evaluated':0,'total_cells':0}

    return overall, per_file


if __name__ == '__main__':
    print('GT dir:', GT_DIR)
    for key, pred_dir in PRED_DIRS.items():
        if not os.path.exists(pred_dir):
            print(f'Prediction folder for {key} not found: {pred_dir} (skipping)')
            continue
        print('\nEvaluating:', key)
        overall, per_file = evaluate_folder(GT_DIR, pred_dir)
        print('Files evaluated:', overall['files_evaluated'])
        print('Total cells compared:', overall['total_cells'])
        print('Cell-level exact match accuracy: {:.4f}'.format(overall['cell_acc']))
        print('Average CER (per-cell): {:.4f}'.format(overall['avg_cer']))
        print('Average WER (per-cell): {:.4f}'.format(overall['avg_wer']))
        # print top 3 per-file accuracies
        items = sorted(per_file.items(), key=lambda kv: kv[0])
        print('\nSample per-file results:')
        for fname, r in items[:5]:
            print(f'  {fname}: cells={r["total_cells"]}, acc={r["cell_acc"]:.4f}, cer={r["avg_cer"]:.4f}, wer={r["avg_wer"]:.4f}')
        # Save per-file metrics and overall summary to outputs/metrics_summary/<mode>/
        out_base = os.path.join('outputs', 'metrics_summary')
        mode_dir = os.path.join(out_base, key)
        os.makedirs(mode_dir, exist_ok=True)

        # per-file CSV
        if per_file:
            rows = []
            for fname, r in items:
                rows.append({
                    'file': fname,
                    'cells': r['total_cells'],
                    'cell_acc': r['cell_acc'],
                    'avg_cer': r['avg_cer'],
                    'avg_wer': r['avg_wer'],
                })
            df_rows = pd.DataFrame(rows)
            df_rows.to_csv(os.path.join(mode_dir, 'per_file_metrics.csv'), index=False)

        # overall metrics JSON + CSV row appended to master summary
        overall_out = {
            'mode': key,
            'files_evaluated': overall.get('files_evaluated', 0),
            'total_cells': overall.get('total_cells', 0),
            'cell_acc': overall.get('cell_acc', 0.0),
            'avg_cer': overall.get('avg_cer', 0.0),
            'avg_wer': overall.get('avg_wer', 0.0),
        }
        # write overall JSON
        try:
            import json
            with open(os.path.join(mode_dir, 'overall_metrics.json'), 'w', encoding='utf-8') as fjson:
                json.dump(overall_out, fjson, indent=2)
        except Exception as e:
            print('Warning: failed to write overall_metrics.json:', e)

        # append to master summary CSV
        master_path = os.path.join(out_base, 'summary.csv')
        df_master_row = pd.DataFrame([overall_out])
        if not os.path.exists(master_path):
            df_master_row.to_csv(master_path, index=False)
        else:
            df_master_row.to_csv(master_path, mode='a', header=False, index=False)

    print('\nDone.')
