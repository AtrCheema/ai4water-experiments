"""
====================================
Neural Networks for classification
====================================
"""
from ai4water.hyperopt import Categorical
from ai4water.datasets import MtropicsLaos
from ai4water.experiments import DLClassificationExperiments

# %%

dataset = MtropicsLaos()

lookback = 5
data =    dataset.make_classification(lookback_steps=lookback)

print(data.shape)

# %%
inputs = data.columns.tolist()[0:-1]
outputs = data.columns.tolist()[-1:]

# %%
exp = DLClassificationExperiments(
    input_features=inputs,
    output_features=outputs,
    epochs=5,
    ts_args={"lookback": lookback},
    save=False
)

# %%
exp.batch_size_space = Categorical(categories=[4, 8, 12, 16, 32],
                                   name="batch_size")

# %%
exp.fit(data=data,
        include=["MLP", "CNN", "LSTM", "TFT"])