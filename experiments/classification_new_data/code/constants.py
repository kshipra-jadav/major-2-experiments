from pathlib import Path

DATA_PATH = Path('/home/kshipra/work/coding/major/ml experiments/experiments/classification_new_data/data')
OUTPUT_PATH = Path('/home/kshipra/work/coding/major/ml experiments/experiments/classification_new_data/output')

SENTINEL_FILE = 'sentinel-1-processed.csv'
EOS_FILE = 'eos-04-processed.csv'

X_cols_eos = ['HH-pol', 'HV-pol']
X_cols_sentinel = ['VV-pol', 'VH-pol']
y_col = ['label']
class_labels = ['Low', 'Medium', 'High', 'Very High']