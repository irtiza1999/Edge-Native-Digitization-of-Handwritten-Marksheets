import pandas as pd
import os
base = os.path.join('outputs','metrics_summary')
master = os.path.join(base,'summary.csv')
report_dir = os.path.join(base,'report')
if not os.path.exists(report_dir):
    os.makedirs(report_dir, exist_ok=True)
if os.path.exists(master):
    df = pd.read_csv(master)
    df_dedup = df.drop_duplicates(subset=['mode'], keep='last')
    out = os.path.join(report_dir,'deduped_summary.csv')
    df_dedup.to_csv(out, index=False)
    print('Wrote deduped summary to', out)
else:
    print('No master summary found', master)
