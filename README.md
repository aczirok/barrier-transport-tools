# barrier-transport-tools
computational tools to (i) correct sampling artifacts in barrier transport data sets and to (ii) fit data to a simple three compartment model

model_v2.py: 
 - model definitions:
   - sampling: sampling model
   - membrane: two compartment model for a passive membrane
   - cell: three compartment model for a barrier cell layer 
 - fit : generic fitting procedure

sampling_correction.ipynb: example usage (jupyter notebook)

fit.py: 
 - helper functions (from sampling_correction.ipynb) to analyze experimental data
 - read_data: custom import function to load sample sequences

  
