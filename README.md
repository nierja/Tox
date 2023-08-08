TensorFlow machine-learning pipeline for prediction of the toxicity of chemicals. Code used in my Master Thesis [Quantitative structure-activity relationship and machine learning](https://dspace.cuni.cz/handle/20.500.11956/181235).

<center><img src="./data/img/pipeline.jpg" width="400" /></center>


## Usage

Descriptor generation is handled by scripts `src/descriptor_generation/*.py`

Hyperparameter tuning of deep NNs: `src/DL/Tox21_tuner.py`

Tuning of GBDT, SVMs, AdaBoost: `src/ML/ML.py`
