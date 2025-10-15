# Leveraging Association Rules for Better Predictions and Better Explanations

## Supplementary Material

## Contents

This archive contains the following resources:

- **`sources`**: The source code is built on the basis of a GIT clone of https://github.com/crillab/pyxai
  - `sources/pyxai/pyxai/examples/RF/`: This directory contains the code related to the experiments described in the paper. The main files to be executed are `run_all_associationrules_DT`, `run_all_associationrules_RF`, `run_all_classement_DT`, and `run_all_classement_RF`.

- **`dataset`**: The datasets converted and used in our experiments.

  More specifically, for each dataset, you can find:
  - `datasets/<dataset>.csv`: The converted dataset.
  - `datasets/<dataset>.types`: A JSON file containing all the information about the features.

- **`logs`**: The outputs produced by the algorithms run in the experiments.
  - `logs/rectify/`: Results obtained with rectification using decision trees and random forests (see tables 2 and 3).
  - `logs/explanation/`: Results showing the impact of incorporating association rules on the size of abductive explanations for decision trees and random forests (see tables 4 and 5).
  - `logs/rectify/ecai_cross_classemnt.py`: Python code to generate tables with more information about rectification statistics.
    
    Execution command:
    ```bash
    python3 ecai_cross_classemnt.py --path=datajson_classementrules_RF/ --plot=datajson_classementrules_RF/
    ```
  
  - `logs/explanation/ecai_cross_association.py`: Python code to generate tables with more information about association rules and abductive explanations for decision trees and random forests.
    
    Execution command:
    ```bash
    python3 ecai_cross_association.py --path=datajson_association_RF_json/ --plot=datajson_association_RF_json/
    ```

- **`proofs.pdf`**: The proofs of the propositions provided in the paper.

- **`directory pyxai`**: A modified version of pyxai used to compute explanations and rectify models.

## Setup

- Ensure you are using a Linux OS and Python version ≥ 3.12.7
- Install Pyxai. Follow these [instructions](https://www.cril.univ-artois.fr/pyxai/documentation/installation/github/). Instead of cloning the software, please use the source provided in this archive.
- Install the required dependencies:
  ```bash
  python3 -m pip install numpy==2.0.2
  python3 -m pip install pandas==2.2.3
  python3 -m pip install scikit-learn==1.5.2
  python3 -m pip install xgboost==1.7.3
  ```
- To compile the modified version of pyxai in the pyxai directory:
  ```bash
  python3 -m pip install -e .
  ```

## How to Run the Program

- For a given dataset, the program `sources/pyxai/pyxai/examples/RF/main_(...)` implements the two approaches presented in the paper: for decision trees and random forests.

- **Program to rectify a model using classification rules:**

  The parameter `-types` allows you to set the number of rules used.
  
  For decision tree classification rules:
  ```bash
  python3 sources/pyxai/pyxai/examples/RF/main_classementrules_DT.py -dataset="../../../../../datasets/breastTumor" -types=100
  ```
  
  For random forest classification rules:
  ```bash
  python3 sources/pyxai/pyxai/examples/RF/main_classementrules_RF.py -dataset="../../../../../datasets/breastTumor" -types=100
  ```

- **Program to compute explanations using association rules:**

  The parameter `-types` allows you to set the number of rules used (100, 1000, 10000, and 100000 in the paper).
  
  For decision tree association rules:
  ```bash
  python3 sources/pyxai/pyxai/examples/RF/main_associationrules_DT.py -dataset="../../../../../datasets/contraceptive" -types=10000
  ```
  
  For random forest association rules:
  ```bash
  python3 sources/pyxai/pyxai/examples/RF/main_associationrules_RF.py -dataset="../../../../../datasets/contraceptive" -types=10000
  ```
  
  **Note:** Only the dataset name (with the directory) needs to be specified in the `-dataset` parameter.

- **To run all datasets:**

  For all decision tree association rules:
  ```bash
  python3 run_all_associationrules_DT.py
  ```
  
  For all random forest association rules:
  ```bash
  python3 run_all_associationrules_RF.py
  ```
  
  For all decision tree classification rules:
  ```bash
  python3 run_all_classement_DT.py
  ```
  
  For all random forest classification rules:
  ```bash
  python3 run_all_classement_RF.py
  ```