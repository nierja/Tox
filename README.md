# Chemical Toxicity Prediction

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/nierja/tox/blob/master/LICENSE)
[![Python](https://img.shields.io/badge/python-3.6%2B-blue.svg)](https://www.python.org/downloads/release)
[![TensorFlow](https://img.shields.io/badge/tensorflow-2.0%2B-orange.svg)](https://www.tensorflow.org)
[![Keras](https://img.shields.io/badge/keras-2.3%2B-red.svg)](https://keras.io)
[![Keras Tuner](https://img.shields.io/badge/keras__tuner-1.0%2B-yellow.svg)](https://github.com/keras-team/keras-tuner)

## Overview

This project explores the application of traditional machine-learning and deep-learning for predicting molecular toxicity. It tackles the challenge of toxicity prediction by generating ~20 different molecular representations and compares their performance on a large variety of models. This code was used for my Master Thesis [Quantitative structure-activity relationship and machine learning](https://dspace.cuni.cz/handle/20.500.11956/181235).

## Features

- Utilizes TensorFlow and Keras for building and training deep neural network.
- Implements hyperparameter tuning using the [HyperBand algorithm](https://arxiv.org/abs/1603.06560)
- Supports cross-validation, ensemble modeling, various dimensionality reduction techniques and evaluation of key metrics.
- Conducts extensive cleaning and preprocessing of chemical data
- Offers a wide variety of molecular representations and models

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)

## Installation

1. Clone the repository, install all dependencies into a virtual environment and activate it:

   ```sh
   git clone https://github.com/nierja/tox.git
   cd tox
   ./lib/initialize_venv.sh
   source ./lib/TOX_GPU_VENV/bin/activate
   export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
   ```

## Usage

1. Generate desired descriptors for the training, validation and test datasets by running:

   ```python
   python3 src/descriptor_generation/generate.py
   ```

    For generating all descriptors across all targets, uncomment their names in `generate.py`.

2. Tune the hyperpadameters and train the machine learning and deep learning models:

   ```python
   python3 src/DL/Tox21_tuner.py
   python3 src/ML/ML.py
   ```

3. You can set model parameters as CLI parameters:

   ```python
   python3 src/DL/Tox21_tuner.py --target=NR-AR --NN_type=DNN --n_layers=4 --fp=ecfp4
   ```

## Resultss

This repository contains code for building a pipeline for toxicity prediction. Various 2D and 3D fingerprints are generated using RDKit and Mordred, and their suitability as molecular representations can be compared on a large variety of machine-learning and deep-learning models. 

Both our deep learning and traditional machine learning models are employed on Tox21 and Ames Mutagenicity datasets. Their performance is evaluated against recently published models for toxicity prediction using the AUC-ROC metric and, regarding certain
toxicity targets, shows improvement over these models. For further information. please see Master Thesis [Quantitative structure-activity relationship and machine learning](https://dspace.cuni.cz/handle/20.500.11956/181235).

