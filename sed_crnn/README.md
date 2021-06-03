My implementation of https://github.com/sharathadavanne/sed-crnn.

Log mel band energies for each desired fold has been calculated followed by splitting the input features into small sequences. These are then fed to CNN-RNN hybrid network with time distributed layers.
F1 score and error rate on test data are calculated. 

Running over fold 1 and 2, got metrics of:
avg er = 0.698478344115928
avg f1 = 0.5490796904522655
