import pandas as pd
import tensorflow as tf

DATA_PATH = '/home/kshipra/work/major/ml experiments/data/manually_combined.xlsx'

eos = pd.read_excel(DATA_PATH, sheet_name='all_stacked_eos')
sentinel = pd.read_excel(DATA_PATH, sheet_name='all_stacked_sentinel')
combined = pd.read_excel(DATA_PATH, sheet_name='eos_sent_combined')


from model_experiments import PredictionIntervalEstimation

model = tf.keras.Sequential([
    # Input layer
    tf.keras.Input(shape=(2, )),
    tf.keras.layers.Dense(2, activation='relu'),
    tf.keras.layers.Dense(1)
])

PredictionIntervalEstimation(eos, features=['HH', 'HV'], target='SM', satellite='EOS').run_experiment(model=model, verbose=1)