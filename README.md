# soil_temp

## Main files and folders:
--data - dataset folder

--test_outputs - experiment log folder

--test_outputs/build_table.ipynb - a notebook to build the experiment result table.

--boosted_forest.py - Scikit-learn-based (slow) implementation of Boosted Forest
--soil_temp_experiments.ipynb - experiments on the soil temperature dataset
--standard_datasets.ipynb - experiments on standard datasets
--test_hidden_size.ipynb - test size of the hidden layer on the Diabetes dataset

## Setup
1. Download COSMOS-UK dataset via the notebook https://github.com/masterdoors/soil_temp/blob/main/data/UK/get_dataset.ipynb. If the scipt does not work, please download the dataset manually via the link: https://catalogue.ceh.ac.uk/documents/399ed9b1-bf59-4d85-9832-ee4d29f49bfb and put all the data to the folder: data/UK/catalogue.ceh.ac.uk/datastore/eidchub/
2. Install requirements for the experiments with boosting and ARIMA:
   
   2.1. Install python 3.9
   
   2.2. Install dependencies:
   
       pip install -r requirements_boosting.txt
   
   2.3. Run the experiments (soil_temp_experiments.ipynb, standard_datasets.ipynb)
3. Install requirements for the experiments with neural networks:
   
   3.1. Install python 3.11
   
   3.2. Install dependencies:
   
       pip install -r requirements_tkan.txt
   
   3.3. Run the experiments (soil_temp_experiments.ipynb).
