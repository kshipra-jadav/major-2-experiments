import tensorflow as tf
import pandas as pd

DATA_PATH = '/home/kshipra/work/major/ml experiments/data/manually_combined.xlsx'

def load_eos_df() -> pd.DataFrame:
    eos = pd.read_excel(DATA_PATH, sheet_name='all_stacked_eos')
    print(f"Loaded EOS DF with shape: {eos.shape}")

    return eos

def load_sentinel_df() -> pd.DataFrame:
    sentinel = pd.read_excel(DATA_PATH, sheet_name='all_stacked_sentinel')
    print(f"Loaded sentinel DF with shape: {sentinel.shape}")
    
    return sentinel

def load_combined_df() -> pd.DataFrame:
    combined = pd.read_excel(DATA_PATH, sheet_name='eos_sent_combined')
    combined = combined[combined['SM (Combined)'] < 150]
    print(f"Loaded combined DF with shape: {combined.shape}")

    return combined


def model_to_string(model, arrow=' -> '):

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
