# %%
import pandas as pd
import tensorflow as tf

DATA_PATH = '/home/kshipra/work/major/ml experiments/data/manually_combined.xlsx'

eos = pd.read_excel(DATA_PATH, sheet_name='all_stacked_eos')
sentinel = pd.read_excel(DATA_PATH, sheet_name='all_stacked_sentinel')
combined = pd.read_excel(DATA_PATH, sheet_name='eos_sent_combined')

len(eos), len(sentinel), len(combined)

# %%
X_cols = ['VH', 'VV', 'HH', 'HV', 'Angle']
y_col = ['SM (Combined)']

num_features = len(X_cols)

# %%
combined = combined[combined['SM (Combined)'] < 150]

combined

# %%
combined[["HH","Angle"]].describe()

# %%
def model_to_string(model, arrow=' -> '):
    import tensorflow as tf

    def _repr_layer(layer):
        cls = layer.__class__.__name__

        # Skip input layers
        if isinstance(layer, tf.keras.layers.InputLayer):
            return None

        # Dense: show units and non-linear activation
        if cls == 'Dense':
            units = getattr(layer, 'units', None)
            act = getattr(layer, 'activation', None)
            act_name = None
            if act is not None and hasattr(act, '__name__'):
                act_name = act.__name__
            # append activation if it's not linear
            return f"{units}" + (f"({act_name})" if act_name and act_name != 'linear' else "")

        # Dropout: show rate
        if cls == 'Dropout':
            rate = getattr(layer, 'rate', None)
            return f"Dropout({rate})"

        # Activation layer (separate layer)
        if cls == 'Activation':
            act = getattr(layer, 'activation', None)
            act_name = act.__name__ if (act is not None and hasattr(act, '__name__')) else str(act)
            return f"Activation({act_name})"

        # Nested model: recurse and wrap in parentheses
        if hasattr(layer, 'layers') and isinstance(layer.layers, (list, tuple)):
            inner = model_to_string(layer, arrow)
            return f"({inner})" if inner else None

        # Generic fallback: use class name (keeps it informative)
        return cls

    parts = []
    for layer in getattr(model, 'layers', []):
        s = _repr_layer(layer)
        if s:
            parts.append(s)
    return arrow.join(parts)


# %% [markdown]
# ## Finetuning value of `r`

# %%
all_results = dict()

# %%
from model_experiments import PredictionIntervalWithTubeLoss
from tensorflow import keras
model = keras.Sequential([
    keras.Input(shape=(len(X_cols), )),
    keras.layers.Dense(8, activation='relu', kernel_initializer=keras.initializers.RandomNormal(mean=0.0, stddev=0.2)),
    keras.layers.Dense(4, activation='relu', kernel_initializer=keras.initializers.RandomNormal(mean=0.0, stddev=0.2)),
    keras.layers.Dense(2, activation='linear',
                kernel_initializer=keras.initializers.RandomNormal(mean=0.0, stddev=0.3),
                bias_initializer=keras.initializers.Constant(value=[-3,3]))
])

r_vals = [0.1, 0.3, 0.5, 0.7, 0.9]
r_results = dict()

for r in r_vals:
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=0.02,
        decay_steps=10000,
        decay_rate=0.01)
    opt = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

    param_string = model_to_string(model)

    pi = PredictionIntervalWithTubeLoss(combined, X_cols, y_col[0], satellite='EOS+Sentinel', r=r)

    results = pi.run_experiment(model=model, optimizer=opt, epochs=500, 
                                model_param_string=param_string, plot_losses=False, plot_interval=False,
                                return_preds=True)
    
    r_results[r] = results

# %%
import pandas as pd

df = pd.DataFrame([
    {
        "r": r,
        "test_PICP": vals["test_results"]["PICP"],
        "test_MPIW": vals["test_results"]["MPIW"],
        "val_PICP": vals["val_results"]["PICP"],
        "val_MPIW": vals["val_results"]["MPIW"],
    }
    for r, vals in r_results.items()
])

df

# %% [markdown]
# ## Finetuning value of `delta`

# %%
from model_experiments import PredictionIntervalWithTubeLoss
from tensorflow import keras
model = keras.Sequential([
    keras.Input(shape=(len(X_cols), )),
    keras.layers.Dense(8, activation='relu', kernel_initializer=keras.initializers.RandomNormal(mean=0.0, stddev=0.2)),
    keras.layers.Dense(4, activation='relu', kernel_initializer=keras.initializers.RandomNormal(mean=0.0, stddev=0.2)),
    keras.layers.Dense(2, activation='linear',
                kernel_initializer=keras.initializers.RandomNormal(mean=0.0, stddev=0.3),
                bias_initializer=keras.initializers.Constant(value=[-3,3]))
])


delta_vals = [0, 0.01, 0.05, 0.1, 0.2]
delta_results = dict()

for delta in delta_vals:
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=0.02,
        decay_steps=10000,
        decay_rate=0.01)
    opt = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

    param_string = model_to_string(model)

    pi = PredictionIntervalWithTubeLoss(combined, X_cols, y_col[0], satellite='EOS+Sentinel', r=0.5, delta=delta)

    results = pi.run_experiment(model=model, optimizer=opt, epochs=500, 
                                model_param_string=param_string, plot_losses=False, plot_interval=False,
                                return_preds=True)
    
    delta_results[delta] = results

# %%
import pandas as pd

df = pd.DataFrame([
    {
        "delta": delta,
        "test_PICP": vals["test_results"]["PICP"],
        "test_MPIW": vals["test_results"]["MPIW"],
        "val_PICP": vals["val_results"]["PICP"],
        "val_MPIW": vals["val_results"]["MPIW"],
    }
    for delta, vals in delta_results.items()
])

df

# %%
from model_experiments import PredictionIntervalWithTubeLoss
from tensorflow import keras

tf.keras.backend.clear_session()

model1 = tf.keras.Sequential([
    # Input layer
    tf.keras.Input(shape=(len(X_cols), )),
    tf.keras.layers.Dense(16, activation='relu', kernel_initializer=keras.initializers.RandomNormal(mean=0.0, stddev=0.2)),
    tf.keras.layers.Dropout(0.09),
    tf.keras.layers.Dense(8, activation='relu', kernel_initializer=keras.initializers.RandomNormal(mean=0.0, stddev=0.2)),
    tf.keras.layers.Dropout(0.09),
    tf.keras.layers.Dense(2, activation='linear',
                kernel_initializer=keras.initializers.RandomNormal(mean=0.0, stddev=0.3),
                bias_initializer=keras.initializers.Constant(value=[-3,3]))
])

model2 = tf.keras.Sequential([
    tf.keras.Input(shape=(len(X_cols), )),
    tf.keras.layers.Dense(8, activation='relu', kernel_initializer=keras.initializers.RandomNormal(mean=0.0, stddev=0.2)),
    tf.keras.layers.Dense(2, activation='linear',
                kernel_initializer=keras.initializers.RandomNormal(mean=0.0, stddev=0.3),
                bias_initializer=keras.initializers.Constant(value=[-3,3]))
])

model3 = tf.keras.Sequential([
    tf.keras.Input(shape=(len(X_cols), )),
    tf.keras.layers.Dense(4, activation='relu', kernel_initializer=keras.initializers.RandomNormal(mean=0.0, stddev=0.2)),
    tf.keras.layers.Dense(2, activation='linear',
                kernel_initializer=keras.initializers.RandomNormal(mean=0.0, stddev=0.3),
                bias_initializer=keras.initializers.Constant(value=[-3,3]))
])

# %% [markdown]
# ## Exp 1

# %%
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=0.02,
    decay_steps=10000,
    decay_rate=0.01)
opt = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

pi = PredictionIntervalWithTubeLoss(combined, features=X_cols, target=y_col[0])
pi.train_model(model3, optimizer=opt, num_epochs=20)
pi.plot_prediction_interval_3d()
pi.plot_losses()
pi.plot_prediction_interval()

# %% [markdown]
# ## Exp 2

# %%
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=0.02,
    decay_steps=10000,
    decay_rate=0.01)
opt = tf.keras.optimizers.Adam(learning_rate=0.0005)

pi= PredictionIntervalWithTubeLoss(combined, features=X_cols, target=y_col[0])
pi.train_model(model2, optimizer=opt, num_epochs=1000)
pi.plot_prediction_surface('VH', 'VV', 100)
pi.plot_losses()
pi.plot_prediction_interval()

# %% [markdown]
# ## Exp 3

# %%
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=0.02,
    decay_steps=10000,
    decay_rate=0.01)
opt = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

pi= PredictionIntervalWithTubeLoss(combined, features=X_cols, target=y_col[0])
pi.train_model(model1, optimizer=opt)
pi.plot_losses()
pi.plot_prediction_interval()


