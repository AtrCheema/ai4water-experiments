.. _experiments:

Experiments
-----------

The basic purpose of the ``experiments`` module of ai4water is comparison. It can be used for following scenarios

1) Comparison between different machine learning algorithms for a classification or regression task.
    This can be done using ``MLRegressionExperiments`` or ``MLClassificationExperiments`` class.

2) Comparison between different neural network architectures for a classification or a regression task.
    This can be done using ``DLRegressionExperiments`` or ``DLClassificationExperiments`` classes.

3) Test a single algorithm in different scenarios e.g., by applying different transformations on a feature and compare the results
    A use case is shown with ``TransformationExperiments`` class.

4) Optimize the hyperparameters of multiple models. This can be done by setting ``run_type`` to ``optimize``
    in ``experimnnt.fit`` method.

All the classes inherit from ``Experiments`` class.