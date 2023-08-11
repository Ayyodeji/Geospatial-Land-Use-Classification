# Land Use and Land Cover Time Series Analysis

This repository contains a Python script for analyzing land use and land cover time series data using machine learning techniques, including Long Short-Term Memory (LSTM) models. The script processes geospatial data, extracts relevant features, trains an LSTM model, and generates predictions for land use and land cover over time.

## Table of Contents

- [Introduction](#introduction)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
- [Geospatial Data](#geospatial-data)
- [Results](#results)
- [License](#license)

## Introduction

Land use and land cover changes are critical factors in understanding environmental and ecological dynamics. This script aims to analyze time series data related to land use and land cover using geospatial information and machine learning techniques. The provided script processes the dataset, trains an LSTM model, and visualizes predictions on geospatial data.

## Prerequisites

Before using this code, make sure you have the following dependencies installed:

- Python (3.7 or higher)
- TensorFlow (2.0 or higher)
- NumPy
- pandas
- Matplotlib
- scikit-learn
- GDAL (Geospatial Data Abstraction Library)

## Installation

1. Clone this repository:

   ```bash
   git clone https://github.com/your-username/land-use-time-series.git
   cd land-use-time-series
   ```

2. Install the required Python packages:

   ```bash
   pip install GDAL scikit-learn tensorflow 
   ```

## Usage

1. Update the dataset path and other settings in the script as needed.

2. Run the script using a Python interpreter:

   ```bash
   python land_use_time_series.py
   ```

   This will execute the code and perform the following tasks:

   - Load and preprocess the time series data.
   - Train an LSTM model to predict land use and land cover changes.
   - Generate predictions for a sample geospatial image.

## Geospatial Data

The script utilizes the Geospatial Data Abstraction Library (GDAL) to work with geospatial data. It processes a sample geospatial image, extracts features, and generates predictions for land use and land cover. The predicted results are visualized using Matplotlib.

## Results

The script outputs the following:

- Plots of time series data showing feature changes over time.
- Training and validation loss and accuracy curves during model training.
- Evaluation metrics on the test dataset.
- A saved trained model file (`model.h5`).
- A predicted geospatial image with land use and land cover predictions.

## License

This project is licensed under the GNU General Public License file for details.
