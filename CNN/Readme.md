**File List**

1. task1cnn.py: Core file to execute 
2. data_shifter.py: To create dataframe from fold metadata and if user wants files to be copied, makes directory and copies data to test/train/validate dir
3. features.py: Create log mel band energy features for 40 filter banks and save as npy file
4. callrows.py: To shuffle, stack and standardize data before passing it to the model
5. evaluate.py: Uses sed eval package to produce classification accuracies for acoustic scene classification

In addition, this folder contains predicted classes for fold 1 and 3 data in a "list of dictionaries" format.

