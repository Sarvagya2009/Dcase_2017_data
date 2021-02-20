**File List**

1. task1cnn.py: core file to execute 
2. data_shifter.py: to create dataframe from fold metadata and if user wants files to be copied, makes directory and copies data to test/train/validate dir
3. features.py: create log mel band energy features for 40 filter banks and save as npy file
4. callrows.py: To shuffle, stack and standardize data before passing it to the model
