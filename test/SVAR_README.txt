This folder replicates the results from the Working Paper "Uncertainty and the Real Economy: Evidence from Denmark"

The model is estimated using BEAR 4.0 and can be downloaded from the ECB website. 

The folder contains of the following files:
- data.xlsx: the data set and the specification of the exogenous block
- HistDecomp.m: calculates the historical decomposition of the model with constant parameters estimated over the period from 2000Q1-2020Q1. It is nessecary to run the main model before running this program. This is done in line 5. To replicate the historical decomposition in the paper, transform the historical decomposition in levels to first differences in excel. 
- BearSettings.m: all the settings in order to run the baseline model. 

