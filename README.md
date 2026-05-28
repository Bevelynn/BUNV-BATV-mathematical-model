## Description of .xlsx and .csv files: 
 
### "BUNV_data.xlsx":
Required for the file "posterior_predictions_visualiser.py".  
Contains in vitro experimental data used to calibrate the mathematical model for BUNV.  
Sheets 1 - 4: Viral titre data (ED assay outcomes at each time point) from the low MOI infection experiments (four replicates).
Sheets 5 - 8: Viral titre data (ED assay outcomes at each time point) from the high MOI infection experiments (four replicates).  
Sheet 9: Cell number data from the low MOI infection experiments.
Sheet 10: Cell number data from the high MOI infection experiments.
Sheets 11 - 13: Extracellular RNA copy numbers for each genome segment from the high MOI experiments (reported as RNA copy concentrations in the PCR reaction mixture).
Sheets 14 - 16: Viral titre data (ED assay outcomes at each time point) from the viral decay experiments (three replicates).

### "BATV_data.xlsx":
Required for the file "posterior_predictions_visualiser.py".
Contains in vitro experimental data used to calibrate the mathematical model for BATV.  
Sheets 1 - 5: Viral titre data (ED assay outcomes at each time point) from the low MOI infection experiments (five replicates).
Sheets 6 - 10: Viral titre data (ED assay outcomes at each time point) from the high MOI infection experiments (five replicates).
Sheet 11: Cell number data from the infection experiments.
Sheets 12 - 14: Viral titre data (ED assay outcomes at each time point) from the viral decay experiments (three replicates).

### "BUNV_posterior.csv" and "BATV_posterior.csv":
Required for the files "posterior_predictions_visualiser.py" and "posterior_distributions_visualiser.py".
Contains lists of parameter values from the posterior distributions for BUNV and BATV, respectively.  
Column 1: Values of ln(posterior density) for the parameter set of the corresponding row.
Column 2: Values of $T_0$.  
Column 3: Values of $\beta$.    
Column 4: Values of $c$.    
Column 5: Values of $\gamma$.
Column 6: Values of $p$.    
Column 7: Values of $p_\mathrm{RNA}$ (N/A for BATV).  
Column 8: Values of $n_E$.  
Column 9: Values of $n_I$. 
Column 10: Values of $\kappa$.
Column 11: Values of $\tau_E$.
Column 12: Values of $\tau_I$.
Column 13: Values of $\omega$ (fixed for BATV).
Column 14: Values of $\alpha_{\frac{\mathrm{BHK}}{\mathrm{A549}}}$ (N/A for BATV).
  
          
## Description of .py files

### "model_functions.py":
Contains functions to simulate the mathematical model and functions to compile the data for each virus into dictionaries.

### "posterior_predictions_visualiser.py":
Uses "model_functions.py" to simulate the mathematical model for a sample of parameter sets from the posterior samples, "BUNV_posterior.csv" and "BATV_posterior.csv", and plots the prediction corresponding to the most likely parameter set, the pointwise 95% credible intervals of the predictions, and the corresponding experimental data sets, to produce Fig 3, Fig 4, Fig 5, and Fig 9.  

### "posterior_distributions_visualiser.py":
Plots the marginal posterior distributions from "BUNV_posterior.csv" and "BATV_posterior.csv", creating Fig 6.  
Obtains and plots posterior distributions for relevant parameter combinations to produce Fig 7.
Plots two-dimensional histograms of the posterior distributions for $p$ and $\beta$, creating Fig 8.  
