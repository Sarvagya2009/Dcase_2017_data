# Dcase_2017_data
Includes code that was made to work on the data in DCASE 2017 challenge

## This repository includes 
1) data_shifter.py: It reads data from a metadata text file, in this case, from dcase 2017 acoustic scene classification crossval fold metadata to form a dataframe with file and scene(label) as column. Then it copies the dataset from audio folder (which has all the audiofiles without labels) to 1) Train directory where it sorts files on basis of labels
                                                                                                                      2) Test directory where all folds are copied (unlabeled)
                                                                                                                          
