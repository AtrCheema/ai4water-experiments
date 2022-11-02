"""
===============================================
Machine learning algorithms for classification
===============================================
"""
from ai4water.datasets import MtropicsLaos
from ai4water.experiments import MLClassificationExperiments

# %%

dataset = MtropicsLaos()

data =    dataset.make_classification(lookback_steps=1)

print(data.shape)

# %%
inputs = data.columns.tolist()[0:-1]
outputs = data.columns.tolist()[-1:]

# %%
exp = MLClassificationExperiments(
    input_features=inputs,
    output_features=outputs,
    epochs=5,
    save=False
)

# %%
exp.fit(data=data)

# %%
exp.plot_cv_scores(data=data)

# %%
exp.compare_precision_recall_curves(data=data)