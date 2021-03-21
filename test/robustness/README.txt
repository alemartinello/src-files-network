This folder makes the figures in the working paper with alternative specification and identifications

The program is ran through the 'manyirf.m' file. You need the rest of the BEAR toolbox to work with the files.

Before this, please check all settings:
- Model specific settings such as time period is set in \bear_settings.
The settings file is special because it allows you to loop over different lag lenghts, model variables etc. 
- In the beginning of \bear_toolbox_maincode it is specified which variables the program must not delete between each itereation.
