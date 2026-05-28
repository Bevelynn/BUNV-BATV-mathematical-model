import numpy as np
import phymcmc.plot
import model_functions
import matplotlib.pyplot as plt

color_BUNV = 'tab:blue'
color_BATV = 'tab:orange'

# Loading the posterior samples for each virus
posterior_BUNV = np.loadtxt('BUNV_posterior.csv', delimiter = ',')
posterior_BATV = np.loadtxt('BATV_posterior.csv', delimiter = ',')

# Loading data for each virus
dat_BUNV = model_functions.get_data_BUNV()
dat_BATV = model_functions.get_data_BATV()

# time 0 represents the time at which virus was innoculated into each well
max_t_inf = 175
max_t_decay = 125
gran = 10 # number of time points per hour
inc_times = np.linspace(0, 1, gran+1)
inf_times = np.linspace(1, max_t_inf, (max_t_inf-1)*gran+1)
decay_times = np.linspace(0, max_t_decay, max_t_decay*gran+1)
plotting_times = np.linspace(0, max_t_inf, max_t_inf*gran+1)


# Getting MCMC best fit for BUNV
# i.e parameter set that has the highest posterior probability
bestfit_index_BUNV = np.argmax(posterior_BUNV, axis=0).tolist()[0]
bestfit_BUNV = posterior_BUNV[bestfit_index_BUNV, :]

# putting the best parameter set into a dictionary
bfpdic_BUNV = {'V0low': 96564,'V0high': 4501208, 'V0RNA':36186159, 'V0decay': 78629, 
               's': 2, 'lambda': 0.048, 'tauN': 174, 'nN': 5, 'mu': 4.8e-08,
               'T0': bestfit_BUNV[1], 'beta': bestfit_BUNV[2], 'c': bestfit_BUNV[3], 
               'gamma': bestfit_BUNV[4], 'p': bestfit_BUNV[5], 'pRNA': bestfit_BUNV[6],
               'nE': bestfit_BUNV[7], 'nI': bestfit_BUNV[8], 'kappa': bestfit_BUNV[9],
               'tauE': bestfit_BUNV[10], 'tauI': bestfit_BUNV[11], 'w': bestfit_BUNV[12], 'alpha': bestfit_BUNV[13]}


# Getting MCMC best fit for BATV
# i.e parameter set that has the highest posterior probability
bestfit_index_BATV = np.argmax(posterior_BATV, axis=0).tolist()[0]
bestfit_BATV = posterior_BATV[bestfit_index_BATV, :]

# putting the best parameter set into a dictionary
bfpdic_BATV = {'V0low': 367076,'V0high': 26689621, 'V0decay': 333342,
               's': 2, 'lambda': 0.048, 'tauN': 174, 'nN': 5, 'mu': 4.8e-08,
               'T0': bestfit_BATV[1], 'beta': bestfit_BATV[2], 'c': bestfit_BATV[3], 
               'gamma': bestfit_BATV[4], 'p': bestfit_BATV[5], 'pRNA': bestfit_BATV[6],
               'nE': bestfit_BATV[7], 'nI': bestfit_BATV[8], 'kappa': bestfit_BATV[9],
               'tauE': bestfit_BATV[10], 'tauI': bestfit_BATV[11], 'w': bestfit_BATV[12], 'alpha': bestfit_BATV[13]}



pfit = ['T0', 'beta', 'c', 'gamma', 'p', 'pRNA', 'nE', 'nI', 'kappa', 'tauE', 'tauI', 'w', 'alpha']


#BUNV model predictions
params = model_functions.params(bfpdic_BUNV.copy(), pfit)
model = model_functions.model(dat_BUNV, params)
sol_low = model.get_solution(inc_times, inf_times-1,  bfpdic_BUNV['T0'], bfpdic_BUNV['V0low'], 0)
sol_high = model.get_solution(inc_times, inf_times-1,  bfpdic_BUNV['T0'], bfpdic_BUNV['V0high'], bfpdic_BUNV['V0RNA'])
T_low_BUNV, E_low_BUNV, I_low_BUNV, V_inc_low_SIN_A_BUNV, V_inc_low_SIN_B_BUNV, V_inc_low_RNA_BUNV, V_low_SIN_A_BUNV, V_low_SIN_B_BUNV, V_low_RNA_BUNV = sol_low
T_high_BUNV, E_high_BUNV, I_high_BUNV, V_inc_high_SIN_A_BUNV, V_inc_high_SIN_B_BUNV, V_inc_high_RNA_BUNV, V_high_SIN_A_BUNV, V_high_SIN_B_BUNV, V_high_RNA_BUNV = sol_high

V_decay_BUNV = bfpdic_BUNV['V0decay']*np.exp(-bfpdic_BUNV['c']*decay_times)

# Getting a sample of posterior predictions to calculate credible interval
# sample n = 1000 parameter sets with replacement
n = 1000
solutions = []
sample_idxs = np.random.choice(len(posterior_BUNV), n)
for idx_num, idx in enumerate(sample_idxs, 1):
    print(f'{idx_num}/{n}' )
    model.params.vector = posterior_BUNV[idx,1:]
    solutions.append( [model.get_solution(inc_times, inf_times-1, model.params.pardict['T0'], model.params.pardict['V0low'], 0), model.get_solution(inc_times, inf_times-1, model.params.pardict['T0'], model.params.pardict['V0high'], model.params.pardict['V0RNA']), model.params.pardict['V0decay']*np.exp(-model.params.pardict['c']*decay_times)]  )

Ts_low_BUNV = np.vstack( [solutions[i][0][0] for i in range(n)] )
Es_low_BUNV = np.vstack( [solutions[i][0][1] for i in range(n)] )
Is_low_BUNV = np.vstack( [solutions[i][0][2] for i in range(n)] )
Vs_inc_low_SIN_A_BUNV = np.vstack( [solutions[i][0][3] for i in range(n)] )
Vs_inc_low_SIN_B_BUNV = np.vstack( [solutions[i][0][4] for i in range(n)] )
Vs_inc_low_RNA_BUNV = np.vstack( [solutions[i][0][5] for i in range(n)] )
Vs_low_SIN_A_BUNV = np.vstack( [solutions[i][0][6] for i in range(n)] )
Vs_low_SIN_B_BUNV = np.vstack( [solutions[i][0][7] for i in range(n)] )
Vs_low_RNA_BUNV = np.vstack( [solutions[i][0][8] for i in range(n)] )

Ts_high_BUNV = np.vstack( [solutions[i][1][0] for i in range(n)] )
Es_high_BUNV = np.vstack( [solutions[i][1][1] for i in range(n)] )
Is_high_BUNV = np.vstack( [solutions[i][1][2] for i in range(n)] )
Vs_inc_high_SIN_A_BUNV = np.vstack( [solutions[i][1][3] for i in range(n)] )
Vs_inc_high_SIN_B_BUNV = np.vstack( [solutions[i][1][4] for i in range(n)] )
Vs_inc_high_RNA_BUNV = np.vstack( [solutions[i][1][5] for i in range(n)] )
Vs_high_SIN_A_BUNV = np.vstack( [solutions[i][1][6] for i in range(n)] )
Vs_high_SIN_B_BUNV = np.vstack( [solutions[i][1][7] for i in range(n)] )
Vs_high_RNA_BUNV = np.vstack( [solutions[i][1][8] for i in range(n)] )

Vs_decay_BUNV = np.vstack( [solutions[i][2] for i in range(n)] )



#BATV model predictions
params = model_functions.params(bfpdic_BATV.copy(), pfit)
model = model_functions.model(dat_BATV, params)
sol_low = model.get_solution(inc_times, inf_times-1, bfpdic_BATV['T0'], bfpdic_BATV['V0low'], 0)
sol_high = model.get_solution(inc_times, inf_times-1, bfpdic_BATV['T0'], bfpdic_BATV['V0high'], 0)
T_low_BATV, E_low_BATV, I_low_BATV, V_inc_low_SIN_A_BATV, V_inc_low_SIN_B_BATV, V_inc_low_RNA_BATV, V_low_SIN_A_BATV, V_low_SIN_B_BATV, V_low_RNA_BATV = sol_low
T_high_BATV, E_high_BATV, I_high_BATV, V_inc_high_SIN_A_BATV, V_inc_high_SIN_B_BATV, V_inc_high_RNA_BATV, V_high_SIN_A_BATV, V_high_SIN_B_BATV, V_high_RNA_BATV = sol_high

V_decay_BATV = bfpdic_BATV['V0decay']*np.exp(-bfpdic_BATV['c']*decay_times)

# Getting a sample of posterior predictions to calculate credible interval
solutions = []
sample_idxs = np.random.choice(len(posterior_BATV), n)
for idx_num, idx in enumerate(sample_idxs, 1):
    print(f'{idx_num}/{n}' )
    model.params.vector = posterior_BATV[idx,1:]
    solutions.append( [model.get_solution(inc_times, inf_times-1, model.params.pardict['T0'], model.params.pardict['V0low'], 0), model.get_solution(inc_times, inf_times-1, model.params.pardict['T0'], model.params.pardict['V0high'], 0), model.params.pardict['V0decay']*np.exp(-model.params.pardict['c']*decay_times)]  )

Ts_low_BATV = np.vstack( [solutions[i][0][0] for i in range(n)] )
Es_low_BATV = np.vstack( [solutions[i][0][1] for i in range(n)] )
Is_low_BATV = np.vstack( [solutions[i][0][2] for i in range(n)] )
Vs_inc_low_SIN_A_BATV = np.vstack( [solutions[i][0][3] for i in range(n)] )
Vs_inc_low_SIN_B_BATV = np.vstack( [solutions[i][0][4] for i in range(n)] )
Vs_inc_low_RNA_BATV = np.vstack( [solutions[i][0][5] for i in range(n)] )
Vs_low_SIN_A_BATV = np.vstack( [solutions[i][0][6] for i in range(n)] )
Vs_low_SIN_B_BATV = np.vstack( [solutions[i][0][7] for i in range(n)] )
Vs_low_RNA_BATV = np.vstack( [solutions[i][0][8] for i in range(n)] )

Ts_high_BATV = np.vstack( [solutions[i][1][0] for i in range(n)] )
Es_high_BATV = np.vstack( [solutions[i][1][1] for i in range(n)] )
Is_high_BATV = np.vstack( [solutions[i][1][2] for i in range(n)] )
Vs_inc_high_SIN_A_BATV = np.vstack( [solutions[i][1][3] for i in range(n)] )
Vs_inc_high_SIN_B_BATV = np.vstack( [solutions[i][1][4] for i in range(n)] )
Vs_inc_high_RNA_BATV = np.vstack( [solutions[i][1][5] for i in range(n)] )
Vs_high_SIN_A_BATV = np.vstack( [solutions[i][1][6] for i in range(n)] )
Vs_high_SIN_B_BATV = np.vstack( [solutions[i][1][7] for i in range(n)] )
Vs_high_RNA_BATV = np.vstack( [solutions[i][1][8] for i in range(n)] )

Vs_decay_BATV = np.vstack( [solutions[i][2] for i in range(n)] )
    


''''########## Plotting Fig 3 (row 1: BUNV low MOI, row 2: BUNV high MOI, row 3: BATV low MOI, row 4: BATV high MOI) ##########'''

colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

R1colorlow = colors[0]
R2colorlow = colors[1]
R3colorlow = colors[2]
R4colorlow = colors[3]
R5colorlow = colors[4]

R1colorhigh = colors[5]
R2colorhigh = colors[6]
R3colorhigh = colors[7]
R4colorhigh = colors[8]
R5colorhigh = colors[9]

max_cells = 3*10**6


# Posterior model predictions for T+E+I, T, E, I, VSIN_A, VSIN_B
gridfig = phymcmc.plot.grid_plot((4,3), wspace = 0.3, rwidth=6, rheight=4.5)


# Total alive cells, T+E+I
ax = gridfig.subaxes(0)

#panel label
ax.text(
    0.95, 0.95, "A1",
    transform=ax.transAxes,
    ha="right",
    va="top",
    fontsize=24,
    fontweight="bold"
)

# Plot best fit
ax.plot(plotting_times, T_low_BUNV + E_low_BUNV + I_low_BUNV, color = color_BUNV)
# Compute 95% variability
med, lb95, ub95, lb65, ub65 = np.percentile(Ts_low_BUNV + Es_low_BUNV + Is_low_BUNV,[50,2.5,100-2.5, 17.5, 100-17.5],axis=0)
# Show variability as shaded area
ax.fill_between(plotting_times,ub95,lb95,color='#'+'c'*6, zorder=-100)

ax.scatter(dat_BUNV['times_all'], dat_BUNV['cells_low_R1'], color = R1colorlow)
ax.scatter(dat_BUNV['times_all'], dat_BUNV['cells_low_R2'], color = R2colorlow)
ax.scatter(dat_BUNV['times_all'], dat_BUNV['cells_low_R3'], color = R3colorlow)
ax.scatter(dat_BUNV['times_all'], dat_BUNV['cells_low_R4'], color = R4colorlow)

ax.set_xlabel('Time (hpi)')
ax.set_xticks(np.arange(0, 175, 24))
ax.set_ylabel('Total cells/well')
ax.set_ylim(0, max_cells)


# Separate T, E, and I curves
ax = gridfig.subaxes(1)

ax.text(
    0.95, 0.95, "B1",
    transform=ax.transAxes,
    ha="right",
    va="top",
    fontsize=24,
    fontweight="bold"
)

# Uninfected cells, T
# Plot best fit
ax.plot(plotting_times, T_low_BUNV, color = color_BUNV)
# Compute 95% variability
med, lb95, ub95, lb65, ub65 = np.percentile(Ts_low_BUNV,[50,2.5,100-2.5, 17.5, 100-17.5],axis=0)
# Show variability as shaded area
ax.fill_between(plotting_times,ub95,lb95,color='#'+'c'*6,zorder=-100)

# Eclipse phase cells, E
# Plot best fit
ax.plot(plotting_times, E_low_BUNV, color = color_BUNV, ls = '--')
# Compute 95% variability
med, lb95, ub95, lb65, ub65 = np.percentile(Es_low_BUNV,[50,2.5,100-2.5, 17.5, 100-17.5],axis=0)
# Show variability as shaded area
ax.fill_between(plotting_times,ub95,lb95,color='#'+'c'*6,zorder=-100)

# Infectious cells, I
# Plot best fit
ax.plot(plotting_times, I_low_BUNV, color = color_BUNV, ls = ':')
# Compute 95% variability
med, lb95, ub95, lb65, ub65 = np.percentile(Is_low_BUNV,[50,2.5,100-2.5, 17.5, 100-17.5],axis=0)
# Show variability as shaded area
ax.fill_between(plotting_times,ub95,lb95,color='#'+'c'*6,zorder=-100)


ax.set_xlabel('Time (hpi)')
ax.set_xticks(np.arange(0, 175, 24))
ax.set_ylabel('Cells/well')
ax.set_ylim(0, max_cells)

   
# Infectious virus
ax = gridfig.subaxes(2)

ax.text(
    0.95, 0.95, "C1",
    transform=ax.transAxes,
    ha="right",
    va="top",
    fontsize=24,
    fontweight="bold"
)

# Virus titrated in A549s (units SIN)
yerr = np.vstack([np.abs(dat_BUNV['df_midsin_low_R1']['mode'][3:]-dat_BUNV['df_midsin_low_R1'][k][3:]) for k in ['lb95','ub95']])
ax.errorbar(dat_BUNV['times_R1'],dat_BUNV['df_midsin_low_R1']['mode'][3:], yerr=yerr, marker='s', markersize= 4, label='lower MOI R1', color=R1colorlow, ls = 'none')

yerr = np.vstack([np.abs(dat_BUNV['df_midsin_low_R2']['mode'][3:]-dat_BUNV['df_midsin_low_R2'][k][3:]) for k in ['lb95','ub95']])
ax.errorbar(dat_BUNV['times_R2'],dat_BUNV['df_midsin_low_R2']['mode'][3:],yerr=yerr, marker='s', markersize= 4, label='lower MOI R2', color=R2colorlow, ls = 'none')

#initial points
yerr = np.vstack([np.abs(dat_BUNV['df_midsin_low_R1']['mode'][0]-dat_BUNV['df_midsin_low_R1'][k][0]) for k in ['lb95','ub95']])
ax.errorbar([0],dat_BUNV['df_midsin_low_R1']['mode'][0],yerr=yerr,marker='s', markersize= 4, color=R1colorlow, ls = 'none', alpha = 0.2)

yerr = np.vstack([np.abs(dat_BUNV['df_midsin_low_R1']['mode'][1]-dat_BUNV['df_midsin_low_R1'][k][1]) for k in ['lb95','ub95']])
ax.errorbar([1],dat_BUNV['df_midsin_low_R1']['mode'][1],yerr=yerr,marker='s', markersize= 4, color=R1colorlow, ls = 'none')

yerr = np.vstack([np.abs(dat_BUNV['df_midsin_low_R1']['mode'][2]-dat_BUNV['df_midsin_low_R1'][k][2]) for k in ['lb95','ub95']])
ax.errorbar([1],dat_BUNV['df_midsin_low_R1']['mode'][2],yerr=yerr,marker='s', markersize= 4, color=R1colorlow, ls = 'none', alpha = 0.2)


yerr = np.vstack([np.abs(dat_BUNV['df_midsin_low_R2']['mode'][:2]-dat_BUNV['df_midsin_low_R2'][k][:2]) for k in ['lb95','ub95']])
ax.errorbar([0, 1],dat_BUNV['df_midsin_low_R2']['mode'][:2],yerr=yerr,marker='s', markersize= 4, color=R2colorlow, ls = 'none')

yerr = np.vstack([np.abs(dat_BUNV['df_midsin_low_R2']['mode'][2]-dat_BUNV['df_midsin_low_R2'][k][2]) for k in ['lb95','ub95']])
ax.errorbar([1],dat_BUNV['df_midsin_low_R2']['mode'][2],yerr=yerr,marker='s', markersize= 4, color=R2colorlow, ls = 'none', alpha = 0.2)

# Plot best fit
ax.semilogy(inc_times, V_inc_low_SIN_A_BUNV, color = color_BUNV, zorder = 100)#, label = 'Best fit')
ax.semilogy(inf_times, V_low_SIN_A_BUNV, color =color_BUNV, zorder = 100)

# during incubation period
# Compute 95% variability
med, lb95, ub95, lb65, ub65 = np.percentile(Vs_inc_low_SIN_A_BUNV,[50,2.5,100-2.5, 17.5, 100-17.5],axis=0)
# Show variability as shaded area
ax.fill_between(inc_times,ub95,lb95,color='#'+'c'*6,zorder=-100)#, label='95\% CI')

# after incubation period
# Compute 95% variability
med, lb95, ub95, lb65, ub65 = np.percentile(Vs_low_SIN_A_BUNV,[50,2.5,100-2.5, 17.5, 100-17.5],axis=0)
# Show variability as shaded area
ax.fill_between(inf_times,ub95,lb95,color='#'+'c'*6,zorder=-100)


# Virus titrated in BHKs (units SIN_BHK)
yerr = np.vstack([np.abs(dat_BUNV['df_midsin_low_R3']['mode'][2:]-dat_BUNV['df_midsin_low_R3'][k][2:]) for k in ['lb95','ub95']])
ax.errorbar(dat_BUNV['times_R34'],dat_BUNV['df_midsin_low_R3']['mode'][2:],yerr=yerr, marker='s', markersize= 4, label='lower MOI R3', color=R3colorlow, ls = 'none')
 
yerr = np.vstack([np.abs(dat_BUNV['df_midsin_low_R4']['mode'][2:]-dat_BUNV['df_midsin_low_R4'][k][2:]) for k in ['lb95','ub95']])
ax.errorbar(dat_BUNV['times_R34'],dat_BUNV['df_midsin_low_R4']['mode'][2:],yerr=yerr, marker='s', markersize= 4, label='lower MOI R4', color=R4colorlow, ls = 'none')
 
#initial points
yerr = np.vstack([np.abs(dat_BUNV['df_midsin_low_R3']['mode'][:2]-dat_BUNV['df_midsin_low_R3'][k][:2]) for k in ['lb95','ub95']])
ax.errorbar([0, 1],dat_BUNV['df_midsin_low_R3']['mode'][:2],yerr=yerr,marker='s', markersize= 4, color=R3colorlow, ls = 'none', alpha = 0.2)

yerr = np.vstack([np.abs(dat_BUNV['df_midsin_low_R4']['mode'][:2]-dat_BUNV['df_midsin_low_R4'][k][:2]) for k in ['lb95','ub95']])
ax.errorbar([0, 1],dat_BUNV['df_midsin_low_R4']['mode'][:2],yerr=yerr,marker='s', markersize= 4, color=R4colorlow, ls = 'none', alpha = 0.2)
 
# Plot best fit
ax.semilogy(inc_times, V_inc_low_SIN_B_BUNV, color = color_BUNV, zorder = 100, ls = '--')#, label = 'Best fit')
ax.semilogy(inf_times, V_low_SIN_B_BUNV, color = color_BUNV, zorder = 100, ls = '--')

# during incubation period
# Compute 95% variability
med, lb95, ub95, lb65, ub65 = np.percentile(Vs_inc_low_SIN_B_BUNV,[50,2.5,100-2.5, 17.5, 100-17.5],axis=0)
# Show variability as shaded area
ax.fill_between(inc_times,ub95,lb95,color='#'+'c'*6,zorder=-100)#, label='95\% CI')

# after incubation period
# Compute 95% variability
med, lb95, ub95, lb65, ub65 = np.percentile(Vs_low_SIN_B_BUNV,[50,2.5,100-2.5, 17.5, 100-17.5],axis=0)
# Show variability as shaded area
ax.fill_between(inf_times,ub95,lb95,color='#'+'c'*6,zorder=-100)


ax.plot([1,1], [10**0, 10**10], 'k--')

ax.set_xlabel('Time (hpi)')
ax.set_xticks(np.arange(0, 175, 24))
ax.set_ylabel('Viral titre (SIN/well)')
ax.set_ylim(1,10**10)

hand1,labs1 = ax.get_legend_handles_labels()

ax.legend(hand1,labs1,frameon=False,handlelength=1.0, loc='center left', bbox_to_anchor=(1, 0.5))


''''########## Row 2 (BUNV high MOI) ##########'''

# Total alive cells, T+E+I
ax = gridfig.subaxes(3)

ax.text(
    0.95, 0.95, "A2",
    transform=ax.transAxes,
    ha="right",
    va="top",
    fontsize=24,
    fontweight="bold"
)

# Plot best fit
ax.plot(plotting_times, T_high_BUNV + E_high_BUNV + I_high_BUNV, color = color_BUNV)
# Compute 95% variability
med, lb95, ub95, lb65, ub65 = np.percentile(Ts_high_BUNV + Es_high_BUNV + Is_high_BUNV,[50,2.5,100-2.5, 17.5, 100-17.5],axis=0)
# Show variability as shaded area
ax.fill_between(plotting_times,ub95,lb95,color='#'+'c'*6,zorder=-100)

ax.scatter(dat_BUNV['times_all'], dat_BUNV['cells_high_R1'], color = R1colorhigh)
ax.scatter(dat_BUNV['times_all'], dat_BUNV['cells_high_R2'], color = R2colorhigh)
ax.scatter(dat_BUNV['times_all'], dat_BUNV['cells_high_R3'], color = R3colorhigh)
ax.scatter(dat_BUNV['times_all'], dat_BUNV['cells_high_R4'], color = R4colorhigh)

ax.set_xlabel('Time (hpi)')
ax.set_xticks(np.arange(0, 175, 24))
ax.set_ylabel('Total cells/well')
ax.set_ylim(0, max_cells)


# Separate T, E, and I curves
ax = gridfig.subaxes(4)

ax.text(
    0.95, 0.95, "B2",
    transform=ax.transAxes,
    ha="right",
    va="top",
    fontsize=24,
    fontweight="bold"
)

# Uninfected cells, T
# Plot best fit
ax.plot(plotting_times, T_high_BUNV, color = color_BUNV)
# Compute 95% variability
med, lb95, ub95, lb65, ub65 = np.percentile(Ts_high_BUNV,[50,2.5,100-2.5, 17.5, 100-17.5],axis=0)
# Show variability as shaded area
ax.fill_between(plotting_times,ub95,lb95,color='#'+'c'*6,zorder=-100)

# Eclipse phase cells, E
# Plot best fit
ax.plot(plotting_times, E_high_BUNV, color = color_BUNV, ls = '--')
# Compute 95% variability
med, lb95, ub95, lb65, ub65 = np.percentile(Es_high_BUNV,[50,2.5,100-2.5, 17.5, 100-17.5],axis=0)
# Show variability as shaded area
ax.fill_between(plotting_times,ub95,lb95,color='#'+'c'*6,zorder=-100)

# Infectious cells, I
# Plot best fit
ax.plot(plotting_times, I_high_BUNV, color = color_BUNV, ls = ':')
# Compute 95% variability
med, lb95, ub95, lb65, ub65 = np.percentile(Is_high_BUNV,[50,2.5,100-2.5, 17.5, 100-17.5],axis=0)
# Show variability as shaded area
ax.fill_between(plotting_times,ub95,lb95,color='#'+'c'*6,zorder=-100)


ax.set_xlabel('Time (hpi)')
ax.set_xticks(np.arange(0, 175, 24))
ax.set_ylabel('Cells/well')
ax.set_ylim(0, max_cells)


# Infectious virus
ax = gridfig.subaxes(5)

ax.text(
    0.95, 0.95, "C2",
    transform=ax.transAxes,
    ha="right",
    va="top",
    fontsize=24,
    fontweight="bold"
)

# Virus titrated in A549s (units SIN)
yerr = np.vstack([np.abs(dat_BUNV['df_midsin_high_R1']['mode'][3:]-dat_BUNV['df_midsin_high_R1'][k][3:]) for k in ['lb95','ub95']])
ax.errorbar(dat_BUNV['times_R1'],dat_BUNV['df_midsin_high_R1']['mode'][3:],yerr=yerr,marker='s', markersize= 4, label='higher MOI R1', color=R1colorhigh, ls = 'none')

yerr = np.vstack([np.abs(dat_BUNV['df_midsin_high_R2']['mode'][3:]-dat_BUNV['df_midsin_high_R2'][k][3:]) for k in ['lb95','ub95']])
ax.errorbar(dat_BUNV['times_R2'],dat_BUNV['df_midsin_high_R2']['mode'][3:],yerr=yerr,marker='s', markersize= 4, label='higher MOI R2', color=R2colorhigh, ls = 'none')

#initial points
yerr = np.vstack([np.abs(dat_BUNV['df_midsin_high_R1']['mode'][:2]-dat_BUNV['df_midsin_high_R1'][k][:2]) for k in ['lb95','ub95']])
ax.errorbar([0, 1],dat_BUNV['df_midsin_high_R1']['mode'][:2],yerr=yerr,marker='s', markersize= 4, color=R1colorhigh, ls = 'none')

yerr = np.vstack([np.abs(dat_BUNV['df_midsin_high_R1']['mode'][2]-dat_BUNV['df_midsin_high_R1'][k][2]) for k in ['lb95','ub95']])
ax.errorbar([1],dat_BUNV['df_midsin_high_R1']['mode'][2],yerr=yerr,marker='s', markersize= 4, color=R1colorhigh, ls = 'none', alpha = 0.2)

yerr = np.vstack([np.abs(dat_BUNV['df_midsin_high_R2']['mode'][:2]-dat_BUNV['df_midsin_high_R2'][k][:2]) for k in ['lb95','ub95']])
ax.errorbar([0, 1],dat_BUNV['df_midsin_high_R2']['mode'][:2],yerr=yerr,marker='s', markersize= 4, color=R2colorhigh, ls = 'none')

yerr = np.vstack([np.abs(dat_BUNV['df_midsin_high_R2']['mode'][2]-dat_BUNV['df_midsin_high_R2'][k][2]) for k in ['lb95','ub95']])
ax.errorbar([1],dat_BUNV['df_midsin_high_R2']['mode'][2],yerr=yerr,marker='s', markersize= 4, color=R2colorhigh, ls = 'none', alpha = 0.2)

# Plot best fit
ax.semilogy(inc_times, V_inc_high_SIN_A_BUNV, color = color_BUNV, zorder = 100)#, label = 'Best fit')
ax.semilogy(inf_times, V_high_SIN_A_BUNV, color = color_BUNV, zorder = 100)

# during incubation period
# Compute 95% variability
med, lb95, ub95, lb65, ub65 = np.percentile(Vs_inc_high_SIN_A_BUNV,[50,2.5,100-2.5, 17.5, 100-17.5],axis=0)
# Show variability as shaded area
ax.fill_between(inc_times,ub95,lb95,color='#'+'c'*6,zorder=-100)#, label='95\% CI')

# after incubation period
# Compute 95% variability
med, lb95, ub95, lb65, ub65 = np.percentile(Vs_high_SIN_A_BUNV,[50,2.5,100-2.5, 17.5, 100-17.5],axis=0)
# Show variability as shaded area
ax.fill_between(inf_times,ub95,lb95,color='#'+'c'*6,zorder=-100)


# Virus titrated in BHKs (units SIN_BHK)    
yerr = np.vstack([np.abs(dat_BUNV['df_midsin_high_R3']['mode'][2:]-dat_BUNV['df_midsin_high_R3'][k][2:]) for k in ['lb95','ub95']])
ax.errorbar(dat_BUNV['times_R34'],dat_BUNV['df_midsin_high_R3']['mode'][2:],yerr=yerr,marker='s', markersize= 4, label='higher MOI R3', color=R3colorhigh, ls = 'none')

yerr = np.vstack([np.abs(dat_BUNV['df_midsin_high_R4']['mode'][2:]-dat_BUNV['df_midsin_high_R4'][k][2:]) for k in ['lb95','ub95']])
ax.errorbar(dat_BUNV['times_R34'],dat_BUNV['df_midsin_high_R4']['mode'][2:],yerr=yerr,marker='s', markersize= 4, label='higher MOI R4', color=R4colorhigh, ls = 'none')

#initial points
yerr = np.vstack([np.abs(dat_BUNV['df_midsin_high_R3']['mode'][:2]-dat_BUNV['df_midsin_high_R3'][k][:2]) for k in ['lb95','ub95']])
ax.errorbar([0, 1],dat_BUNV['df_midsin_high_R3']['mode'][:2],yerr=yerr,marker='s', markersize= 4, color=R3colorhigh, ls = 'none', alpha = 0.2)

yerr = np.vstack([np.abs(dat_BUNV['df_midsin_high_R4']['mode'][:2]-dat_BUNV['df_midsin_high_R4'][k][:2]) for k in ['lb95','ub95']])
ax.errorbar([0, 1],dat_BUNV['df_midsin_high_R4']['mode'][:2],yerr=yerr,marker='s', markersize= 4, color=R4colorhigh, ls = 'none', alpha = 0.2)

# Plot best fit
ax.semilogy(inc_times, V_inc_high_SIN_B_BUNV, color = color_BUNV, zorder = 100, ls = '--')#, label = 'Best fit')
ax.semilogy(inf_times, V_high_SIN_B_BUNV, color = color_BUNV, zorder = 100, ls = '--')

# during incubation period
# Compute 95% variability
med, lb95, ub95, lb65, ub65 = np.percentile(Vs_inc_high_SIN_B_BUNV,[50,2.5,100-2.5, 17.5, 100-17.5],axis=0)
# Show variability as shaded area
ax.fill_between(inc_times,ub95,lb95,color='#'+'c'*6, zorder=-100)#, label='95\% CI')

# after incubation period
# Compute 95% variability
med, lb95, ub95, lb65, ub65 = np.percentile(Vs_high_SIN_B_BUNV,[50,2.5,100-2.5, 17.5, 100-17.5],axis=0)
# Show variability as shaded area
ax.fill_between(inf_times,ub95,lb95,color='#'+'c'*6,zorder=-100)


ax.plot([1,1], [10**0, 10**10], 'k--')

ax.set_xlabel('Time (hpi)')
ax.set_xticks(np.arange(0, 175, 24))
ax.set_ylabel('Viral titre (SIN/well)')
ax.set_ylim(1,10**10)    

hand1,labs1 = ax.get_legend_handles_labels()

ax.legend(hand1,labs1,frameon=False,handlelength=1.0, loc='center left', bbox_to_anchor=(1, 0.5))


''''########## Row 3 (BATV low MOI) ##########'''


# Total alive cells, T+E+I
ax = gridfig.subaxes(6)

ax.text(
    0.95, 0.95, "A3",
    transform=ax.transAxes,
    ha="right",
    va="top",
    fontsize=24,
    fontweight="bold"
)

# Plot best fit
ax.plot(plotting_times, T_low_BATV + E_low_BATV + I_low_BATV, color = color_BATV)
# Compute 95% variability
med, lb95, ub95, lb65, ub65 = np.percentile(Ts_low_BATV + Es_low_BATV + Is_low_BATV,[50,2.5,100-2.5, 17.5, 100-17.5],axis=0)
# Show variability as shaded area
ax.fill_between(plotting_times,ub95,lb95,color='#'+'c'*6, zorder=-100)

for rep in range(1, 6):
    ax.scatter(dat_BATV['times_all'], dat_BATV['cells_low_R%s' %rep], color = colors[rep-1])


ax.set_xlabel('Time (hpi)')
ax.set_xticks(np.arange(0, 175, 24))
ax.set_ylabel('Total cells/well')
ax.set_ylim(0, max_cells)


# Separate T, E, and I curves
ax = gridfig.subaxes(7)

ax.text(
    0.95, 0.95, "B3",
    transform=ax.transAxes,
    ha="right",
    va="top",
    fontsize=24,
    fontweight="bold"
)

# Uninfected cells, T
# Plot best fit
ax.plot(plotting_times, T_low_BATV, color = color_BATV)
# Compute 95% variability
med, lb95, ub95, lb65, ub65 = np.percentile(Ts_low_BATV,[50,2.5,100-2.5, 17.5, 100-17.5],axis=0)
# Show variability as shaded area
ax.fill_between(plotting_times,ub95,lb95,color='#'+'c'*6,zorder=-100)

# Eclipse phase cells, E
# Plot best fit
ax.plot(plotting_times, E_low_BATV, color = color_BATV, ls = '--')
# Compute 95% variability
med, lb95, ub95, lb65, ub65 = np.percentile(Es_low_BATV,[50,2.5,100-2.5, 17.5, 100-17.5],axis=0)
# Show variability as shaded area
ax.fill_between(plotting_times,ub95,lb95,color='#'+'c'*6,zorder=-100)

# Infectious cells, I
# Plot best fit
ax.plot(plotting_times, I_low_BATV, color = color_BATV, ls = ':')
# Compute 95% variability
med, lb95, ub95, lb65, ub65 = np.percentile(Is_low_BATV,[50,2.5,100-2.5, 17.5, 100-17.5],axis=0)
# Show variability as shaded area
ax.fill_between(plotting_times,ub95,lb95,color='#'+'c'*6,zorder=-100)


ax.set_xlabel('Time (hpi)')
ax.set_xticks(np.arange(0, 175, 24))
ax.set_ylabel('Cells/well')
ax.set_ylim(0, max_cells)


   
# Infectious virus titrated in A549s (units SIN)
ax = gridfig.subaxes(8)

ax.text(
    0.95, 0.95, "C3",
    transform=ax.transAxes,
    ha="right",
    va="top",
    fontsize=24,
    fontweight="bold"
)

for rep in range(1, 6):
    yerr = np.vstack([np.abs(dat_BATV['df_midsin_low_R%s' %rep]['mode']-dat_BATV['df_midsin_low_R%s' %rep][k]) for k in ['lb95','ub95']])
    ax.errorbar(dat_BATV['times_low_R%s' %rep], dat_BATV['df_midsin_low_R%s' %rep]['mode'],yerr=yerr, marker='s', markersize= 4, label='lower MOI R%s' %rep, color=colors[rep-1], ls = 'none')

# Plot best fit
ax.semilogy(inc_times, V_inc_low_SIN_A_BATV, color = color_BATV, zorder = 100)#, label = 'Best fit')
ax.semilogy(inf_times, V_low_SIN_A_BATV, color = color_BATV, zorder = 100)

# during incubation period
# Compute 95% variability
med, lb95, ub95, lb65, ub65 = np.percentile(Vs_inc_low_SIN_A_BATV,[50,2.5,100-2.5, 17.5, 100-17.5],axis=0)
# Show variability as shaded area
ax.fill_between(inc_times,ub95,lb95,color='#'+'c'*6,zorder=-100)#, label='95\% CI')

# after incubation period
# Compute 95% variability
med, lb95, ub95, lb65, ub65 = np.percentile(Vs_low_SIN_A_BATV,[50,2.5,100-2.5, 17.5, 100-17.5],axis=0)
# Show variability as shaded area
ax.fill_between(inf_times,ub95,lb95,color='#'+'c'*6,zorder=-100)
# Compute 95% variability

ax.plot([1,1], [10**0, 10**10], 'k--')

ax.set_xlabel('Time (hpi)')
ax.set_xticks(np.arange(0, 175, 24))
ax.set_ylabel('Viral titre (SIN/well)')
ax.set_ylim(1,10**10)

hand1,labs1 = ax.get_legend_handles_labels()

ax.legend(hand1,labs1,frameon=False,handlelength=1.0, loc='center left', bbox_to_anchor=(1, 0.5))


''''########## Row 4 (BATV high MOI) ##########'''


# Total alive cells, T+E+I
ax = gridfig.subaxes(9)

ax.text(
    0.95, 0.95, "A4",
    transform=ax.transAxes,
    ha="right",
    va="top",
    fontsize=24,
    fontweight="bold"
)

# Plot best fit
ax.plot(plotting_times, T_high_BATV + E_high_BATV + I_high_BATV, color = color_BATV)
# Compute 95% variability
med, lb95, ub95, lb65, ub65 = np.percentile(Ts_high_BATV + Es_high_BATV + Is_high_BATV,[50,2.5,100-2.5, 17.5, 100-17.5],axis=0)
# Show variability as shaded area
ax.fill_between(plotting_times,ub95,lb95,color='#'+'c'*6,zorder=-100)

for rep in range(1, 6):
    ax.scatter(dat_BATV['times_all'], dat_BATV['cells_high_R%s' %rep], color = colors[rep+4])

ax.set_xlabel('Time (hpi)')
ax.set_xticks(np.arange(0, 175, 24))
ax.set_ylabel('Total cells/well')
ax.set_ylim(0, max_cells)


# Separate T, E, and I curves
ax = gridfig.subaxes(10)

ax.text(
    0.95, 0.95, "B4",
    transform=ax.transAxes,
    ha="right",
    va="top",
    fontsize=24,
    fontweight="bold"
)

# Uninfected cells, T
# Plot best fit
ax.plot(plotting_times, T_high_BATV, color = color_BATV)
# Compute 95% variability
med, lb95, ub95, lb65, ub65 = np.percentile(Ts_high_BATV,[50,2.5,100-2.5, 17.5, 100-17.5],axis=0)
# Show variability as shaded area
ax.fill_between(plotting_times,ub95,lb95,color='#'+'c'*6,zorder=-100)

# Eclipse phase cells, E
# Plot best fit
ax.plot(plotting_times, E_high_BATV, color = color_BATV, ls = '--')
# Compute 95% variability
med, lb95, ub95, lb65, ub65 = np.percentile(Es_high_BATV,[50,2.5,100-2.5, 17.5, 100-17.5],axis=0)
# Show variability as shaded area
ax.fill_between(plotting_times,ub95,lb95,color='#'+'c'*6,zorder=-100)

# Infectious cells, I
# Plot best fit
ax.plot(plotting_times, I_high_BATV, color = color_BATV, ls = ':')
# Compute 95% variability
med, lb95, ub95, lb65, ub65 = np.percentile(Is_high_BATV,[50,2.5,100-2.5, 17.5, 100-17.5],axis=0)
# Show variability as shaded area
ax.fill_between(plotting_times,ub95,lb95,color='#'+'c'*6,zorder=-100)

ax.set_xlabel('Time (hpi)')
ax.set_xticks(np.arange(0, 175, 24))
ax.set_ylabel('Cells/well')
ax.set_ylim(0, max_cells)


# Infectious virus titrated in A549s (units SIN)
ax = gridfig.subaxes(11)

ax.text(
    0.95, 0.95, "C4",
    transform=ax.transAxes,
    ha="right",
    va="top",
    fontsize=24,
    fontweight="bold"
)

for rep in range(1, 6):
    yerr = np.vstack([np.abs(dat_BATV['df_midsin_high_R%s' %rep]['mode']-dat_BATV['df_midsin_high_R%s' %rep][k]) for k in ['lb95','ub95']])
    ax.errorbar(dat_BATV['times_high_R%s' %rep], dat_BATV['df_midsin_high_R%s' %rep]['mode'],yerr=yerr, marker='s', markersize= 4, label='higher MOI R%s' %rep, color=colors[rep+4], ls = 'none')

# Plot best fit
ax.semilogy(inc_times, V_inc_high_SIN_A_BATV, color = color_BATV, zorder = 100)#, label = 'Best fit')
ax.semilogy(inf_times, V_high_SIN_A_BATV, color = color_BATV, zorder = 100)

# during incubation period
# Compute 95% variability
med, lb95, ub95, lb65, ub65 = np.percentile(Vs_inc_high_SIN_A_BATV,[50,2.5,100-2.5, 17.5, 100-17.5],axis=0)
# Show variability as shaded area
ax.fill_between(inc_times,ub95,lb95,color='#'+'c'*6,zorder=-100)#, label='95\% CI')

# after incubation period
# Compute 95% variability
med, lb95, ub95, lb65, ub65 = np.percentile(Vs_high_SIN_A_BATV,[50,2.5,100-2.5, 17.5, 100-17.5],axis=0)
# Show variability as shaded area
ax.fill_between(inf_times,ub95,lb95,color='#'+'c'*6,zorder=-100)

ax.plot([1,1], [10**0, 10**10], 'k--')

ax.set_xlabel('Time (hpi)')
ax.set_xticks(np.arange(0, 175, 24))
ax.set_ylabel('Viral titre (SIN/well)')
ax.set_ylim(1,10**10)    

hand1,labs1 = ax.get_legend_handles_labels()

ax.legend(hand1,labs1,frameon=False,handlelength=1.0, loc='center left', bbox_to_anchor=(1, 0.5))


gridfig.fig.savefig('Fig3.pdf', bbox_inches='tight')
    



'''' ########## Plotting Fig 4 (BUNV high MOI, V_RNA vs extracellular genome counts) ########## '''

gridfig = phymcmc.plot.grid_plot((1,1), wspace = 0.3, rwidth=6, rheight=4)

ax = gridfig.subaxes(0)

ax.scatter(dat_BUNV['times_all'], dat_BUNV['ext_RNA_S_R1'], color = 'blue', label = 'S segment RNA')
ax.scatter(dat_BUNV['times_all'], dat_BUNV['ext_RNA_S_R2'], color = 'blue')
ax.scatter(dat_BUNV['times_all'], dat_BUNV['ext_RNA_S_R3'], color = 'blue')
ax.scatter(dat_BUNV['times_all'], dat_BUNV['ext_RNA_S_R4'], color = 'blue')

ax.scatter(dat_BUNV['times_all'], dat_BUNV['ext_RNA_M_R1'], color = 'orange', label = 'M segment RNA')
ax.scatter(dat_BUNV['times_all'], dat_BUNV['ext_RNA_M_R2'], color = 'orange')
ax.scatter(dat_BUNV['times_all'], dat_BUNV['ext_RNA_M_R3'], color = 'orange')
ax.scatter(dat_BUNV['times_all'], dat_BUNV['ext_RNA_M_R4'], color = 'orange')

ax.scatter(dat_BUNV['times_all'], dat_BUNV['ext_RNA_L_R1'], color = 'green', label = 'L segment RNA')
ax.scatter(dat_BUNV['times_all'], dat_BUNV['ext_RNA_L_R2'], color = 'green')
ax.scatter(dat_BUNV['times_all'], dat_BUNV['ext_RNA_L_R3'], color = 'green')
ax.scatter(dat_BUNV['times_all'], dat_BUNV['ext_RNA_L_R4'], color = 'green')

# Plot best fit
ax.semilogy(inc_times, V_inc_high_RNA_BUNV, color = color_BUNV)
ax.semilogy(inf_times, V_high_RNA_BUNV, color = color_BUNV)

# during incubation period
# Compute 95% variability
med, lb95, ub95, lb65, ub65 = np.percentile(Vs_inc_high_RNA_BUNV, [50,2.5,100-2.5, 17.5, 100-17.5],axis=0)
#show variability as shaded area
ax.fill_between(inc_times, ub95, lb95, color='#'+'c'*6, zorder=-100)

# after incubation period
# Compute 95% variability
med, lb95, ub95, lb65, ub65 = np.percentile(Vs_high_RNA_BUNV, [50,2.5,100-2.5, 17.5, 100-17.5],axis=0)
# Show variability as shaded area
ax.fill_between(inf_times, ub95, lb95, color='#'+'c'*6,zorder=-100)

ax.plot([1,1], [10**0, 10**10], 'k--')
    
ax.set_yscale('log')
ax.set_xlabel('Time (hpi)')
ax.set_xticks(np.arange(0, 175, 24))
ax.set_ylabel('Virus (RNA copies/well)')
ax.set_ylim(1, 10**10)

ax.legend(frameon=False, handlelength=1.0, loc='center left', bbox_to_anchor=(1, 0.5))

gridfig.fig.savefig('Fig4.pdf', bbox_inches='tight')



'''' ########## Plotting Fig 5 (viral decay) ########## '''
    
R1color = 'blue'
R2color = 'orange'
R3color = 'green'

plt.figure(figsize = (11, 5))

#BUNV
ax = plt.subplot(121)

yerr = np.vstack([np.abs(dat_BUNV['df_midsin_decay_R1']['mode']-dat_BUNV['df_midsin_decay_R1'][k]) for k in ['lb95','ub95']])
plt.errorbar(dat_BUNV['times_decay'], dat_BUNV['df_midsin_decay_R1']['mode'],yerr=yerr, marker='o', markersize= 4, label='R1', color=R1color, ls = 'none')

yerr = np.vstack([np.abs(dat_BUNV['df_midsin_decay_R2']['mode']-dat_BUNV['df_midsin_decay_R2'][k]) for k in ['lb95','ub95']])
plt.errorbar(dat_BUNV['times_decay'],dat_BUNV['df_midsin_decay_R2']['mode'],yerr=yerr, marker='o', markersize= 4, label='R2', color=R2color, ls = 'none')

yerr = np.vstack([np.abs(dat_BUNV['df_midsin_decay_R3']['mode']-dat_BUNV['df_midsin_decay_R3'][k]) for k in ['lb95','ub95']])
plt.errorbar(dat_BUNV['times_decay'],dat_BUNV['df_midsin_decay_R3']['mode'],yerr=yerr, marker='o', markersize= 4, label='R3', color=R3color, ls = 'none')

#plot best fit
plt.plot(decay_times, V_decay_BUNV, color = color_BUNV)

# Compute 95% variability
med, lb95, ub95, lb65, ub65 = np.percentile(Vs_decay_BUNV,[50,2.5,100-2.5, 17.5, 100-17.5],axis=0)
# Show variability as shaded area
plt.fill_between(decay_times,ub95,lb95,color='#'+'c'*6, zorder=-100)

plt.yscale('log')
plt.xlabel('Time (hpi)')
plt.xticks(np.arange(0, 125, 24))
plt.grid(which='major',color=[0.9]*3,zorder=-10)
plt.ylabel('Viral titre (SIN/well)')
plt.ylim(5,2*10**6)
plt.title('BUNV')

#BATV
ax = plt.subplot(122)

yerr = np.vstack([np.abs(dat_BATV['df_midsin_decay_R1']['mode']-dat_BATV['df_midsin_decay_R1'][k]) for k in ['lb95','ub95']])
plt.errorbar(dat_BATV['times_decay'], dat_BATV['df_midsin_decay_R1']['mode'],yerr=yerr, marker='o', markersize= 4, label='R1', color=R1color, ls = 'none')

yerr = np.vstack([np.abs(dat_BATV['df_midsin_decay_R2']['mode']-dat_BATV['df_midsin_decay_R2'][k]) for k in ['lb95','ub95']])
plt.errorbar(dat_BATV['times_decay'],dat_BATV['df_midsin_decay_R2']['mode'],yerr=yerr, marker='o', markersize= 4, label='R2', color=R2color, ls = 'none')

yerr = np.vstack([np.abs(dat_BATV['df_midsin_decay_R3']['mode']-dat_BATV['df_midsin_decay_R3'][k]) for k in ['lb95','ub95']])
plt.errorbar(dat_BATV['times_decay'],dat_BATV['df_midsin_decay_R3']['mode'],yerr=yerr, marker='o', markersize= 4, label='R3', color=R3color, ls = 'none')

#plot best fit
plt.plot(decay_times, V_decay_BATV, color = color_BATV)

# Compute 95% variability
med, lb95, ub95, lb65, ub65 = np.percentile(Vs_decay_BATV,[50,2.5,100-2.5, 17.5, 100-17.5],axis=0)
# Show variability as shaded area
plt.fill_between(decay_times,ub95,lb95,color='#'+'c'*6, zorder=-100)

plt.yscale('log')
plt.xlabel('Time (hpi)')
plt.xticks(np.arange(0, 125, 24))
plt.grid(which='major',color=[0.9]*3,zorder=-10)
plt.ylabel('Viral titre (SIN/well)')
plt.ylim(5,2*10**6)
plt.title('BATV')

plt.savefig('Fig5.pdf')



'''' ########## Plotting Fig 9 (virus comparison across MOIs) ########## '''

T0 = 10**6
# using equal V0 for each virus, in terms of SIN
V0low = 10**5
V0high = 7*10**6


#BUNV model predictions
params = model_functions.params(bfpdic_BUNV.copy(), pfit)
model = model_functions.model(dat_BUNV, params)
sol_low = model.get_solution(inc_times, inf_times-1,  T0, V0low, 0)
sol_high = model.get_solution(inc_times, inf_times-1,  T0, V0high, 0)
T_low_BUNV, E_low_BUNV, I_low_BUNV, V_inc_low_SIN_A_BUNV, V_inc_low_SIN_B_BUNV, V_inc_low_RNA_BUNV, V_low_SIN_A_BUNV, V_low_SIN_B_BUNV, V_low_RNA_BUNV = sol_low
T_high_BUNV, E_high_BUNV, I_high_BUNV, V_inc_high_SIN_A_BUNV, V_inc_high_SIN_B_BUNV, V_inc_high_RNA_BUNV, V_high_SIN_A_BUNV, V_high_SIN_B_BUNV, V_high_RNA_BUNV = sol_high


# Getting a sample of posterior predictions to calculate credible interval
solutions = []
sample_idxs = np.random.choice(len(posterior_BUNV), n)
for idx_num, idx in enumerate(sample_idxs, 1):
    print(f'{idx_num}/{n}' )
    model.params.vector = posterior_BUNV[idx,1:]
    solutions.append( [model.get_solution(inc_times, inf_times-1, T0, V0low, 0), model.get_solution(inc_times, inf_times-1, T0, V0high, 0)]  )

Ts_low_BUNV = np.vstack( [solutions[i][0][0] for i in range(n)] )
Es_low_BUNV = np.vstack( [solutions[i][0][1] for i in range(n)] )
Is_low_BUNV = np.vstack( [solutions[i][0][2] for i in range(n)] )
Vs_inc_low_SIN_A_BUNV = np.vstack( [solutions[i][0][3] for i in range(n)] )
Vs_inc_low_SIN_B_BUNV = np.vstack( [solutions[i][0][4] for i in range(n)] )
Vs_inc_low_RNA_BUNV = np.vstack( [solutions[i][0][5] for i in range(n)] )
Vs_low_SIN_A_BUNV = np.vstack( [solutions[i][0][6] for i in range(n)] )
Vs_low_SIN_B_BUNV = np.vstack( [solutions[i][0][7] for i in range(n)] )
Vs_low_RNA_BUNV = np.vstack( [solutions[i][0][8] for i in range(n)] )

Ts_high_BUNV = np.vstack( [solutions[i][1][0] for i in range(n)] )
Es_high_BUNV = np.vstack( [solutions[i][1][1] for i in range(n)] )
Is_high_BUNV = np.vstack( [solutions[i][1][2] for i in range(n)] )
Vs_inc_high_SIN_A_BUNV = np.vstack( [solutions[i][1][3] for i in range(n)] )
Vs_inc_high_SIN_B_BUNV = np.vstack( [solutions[i][1][4] for i in range(n)] )
Vs_inc_high_RNA_BUNV = np.vstack( [solutions[i][1][5] for i in range(n)] )
Vs_high_SIN_A_BUNV = np.vstack( [solutions[i][1][6] for i in range(n)] )
Vs_high_SIN_B_BUNV = np.vstack( [solutions[i][1][7] for i in range(n)] )
Vs_high_RNA_BUNV = np.vstack( [solutions[i][1][8] for i in range(n)] )


#BATV model predictions
params = model_functions.params(bfpdic_BATV.copy(), pfit)
model = model_functions.model(dat_BATV, params)
sol_low = model.get_solution(inc_times, inf_times-1, T0, V0low, 0)
sol_high = model.get_solution(inc_times, inf_times-1, T0, V0high, 0)
T_low_BATV, E_low_BATV, I_low_BATV, V_inc_low_SIN_A_BATV, V_inc_low_SIN_B_BATV, V_inc_low_RNA_BATV, V_low_SIN_A_BATV, V_low_SIN_B_BATV, V_low_RNA_BATV = sol_low
T_high_BATV, E_high_BATV, I_high_BATV, V_inc_high_SIN_A_BATV, V_inc_high_SIN_B_BATV, V_inc_high_RNA_BATV, V_high_SIN_A_BATV, V_high_SIN_B_BATV, V_high_RNA_BATV = sol_high


# Getting a sample of posterior predictions to calculate credible interval
solutions = []
sample_idxs = np.random.choice(len(posterior_BATV), n)
for idx_num, idx in enumerate(sample_idxs, 1):
    print(f'{idx_num}/{n}' )
    model.params.vector = posterior_BATV[idx,1:]
    solutions.append( [model.get_solution(inc_times, inf_times-1, T0, V0low, 0), model.get_solution(inc_times, inf_times-1, T0, V0high, 0)]  )

Ts_low_BATV = np.vstack( [solutions[i][0][0] for i in range(n)] )
Es_low_BATV = np.vstack( [solutions[i][0][1] for i in range(n)] )
Is_low_BATV = np.vstack( [solutions[i][0][2] for i in range(n)] )
Vs_inc_low_SIN_A_BATV = np.vstack( [solutions[i][0][3] for i in range(n)] )
Vs_inc_low_SIN_B_BATV = np.vstack( [solutions[i][0][4] for i in range(n)] )
Vs_inc_low_RNA_BATV = np.vstack( [solutions[i][0][5] for i in range(n)] )
Vs_low_SIN_A_BATV = np.vstack( [solutions[i][0][6] for i in range(n)] )
Vs_low_SIN_B_BATV = np.vstack( [solutions[i][0][7] for i in range(n)] )
Vs_low_RNA_BATV = np.vstack( [solutions[i][0][8] for i in range(n)] )

Ts_high_BATV = np.vstack( [solutions[i][1][0] for i in range(n)] )
Es_high_BATV = np.vstack( [solutions[i][1][1] for i in range(n)] )
Is_high_BATV = np.vstack( [solutions[i][1][2] for i in range(n)] )
Vs_inc_high_SIN_A_BATV = np.vstack( [solutions[i][1][3] for i in range(n)] )
Vs_inc_high_SIN_B_BATV = np.vstack( [solutions[i][1][4] for i in range(n)] )
Vs_inc_high_RNA_BATV = np.vstack( [solutions[i][1][5] for i in range(n)] )
Vs_high_SIN_A_BATV = np.vstack( [solutions[i][1][6] for i in range(n)] )
Vs_high_SIN_B_BATV = np.vstack( [solutions[i][1][7] for i in range(n)] )
Vs_high_RNA_BATV = np.vstack( [solutions[i][1][8] for i in range(n)] )



max_cells = 2.5*10**6

gridfig = phymcmc.plot.grid_plot((2,2), wspace = 0.3, rwidth=6, rheight=4.5)


# Uninfected cells, T
ax = gridfig.subaxes(0)

ax.text(
    0.95, 0.95, "A",
    transform=ax.transAxes,
    ha="right",
    va="top",
    fontsize=24,
    fontweight="bold"
)

# Plot best fit
ax.plot(plotting_times, T_low_BUNV, color = color_BUNV)
ax.plot(plotting_times, T_high_BUNV, color = color_BUNV, ls = '--')
ax.plot(plotting_times, T_low_BATV, color = color_BATV)
ax.plot(plotting_times, T_high_BATV, color = color_BATV, ls = '--')

# Compute 95% variability
med, lb95, ub95, lb65, ub65 = np.percentile(Ts_low_BUNV,[50,2.5,100-2.5, 17.5, 100-17.5],axis=0)
# Show variability as shaded area
ax.fill_between(plotting_times,ub95,lb95, color=color_BUNV, zorder=-100, alpha = 0.5)

# Compute 95% variability
med, lb95, ub95, lb65, ub65 = np.percentile(Ts_high_BUNV,[50,2.5,100-2.5, 17.5, 100-17.5],axis=0)
# Show variability as shaded area
ax.fill_between(plotting_times,ub95,lb95, color=color_BUNV, zorder=-100, alpha = 0.5)

# Compute 95% variability
med, lb95, ub95, lb65, ub65 = np.percentile(Ts_low_BATV,[50,2.5,100-2.5, 17.5, 100-17.5],axis=0)
# Show variability as shaded area
ax.fill_between(plotting_times,ub95,lb95, color=color_BATV, zorder=-100, alpha = 0.5)

# Compute 95% variability
med, lb95, ub95, lb65, ub65 = np.percentile(Ts_high_BATV,[50,2.5,100-2.5, 17.5, 100-17.5],axis=0)
# Show variability as shaded area
ax.fill_between(plotting_times,ub95,lb95, color=color_BATV, zorder=-100, alpha = 0.5)


ax.set_xlabel('Time (hpi)')
ax.set_xticks(np.arange(0, 175, 24))
ax.set_ylabel('Uninfected cells')
ax.set_ylim(0, max_cells)


# Eclipse phase cells, E
ax = gridfig.subaxes(1)

ax.text(
    0.95, 0.95, "B",
    transform=ax.transAxes,
    ha="right",
    va="top",
    fontsize=24,
    fontweight="bold"
)

# Plot best fit
ax.plot(plotting_times, E_low_BUNV, color = color_BUNV, label ='BUNV low MOI')
ax.plot(plotting_times, E_high_BUNV, color = color_BUNV, ls = '--', label ='BUNV high MOI')
ax.plot(plotting_times, E_low_BATV, color = color_BATV, label ='BATV low MOI')
ax.plot(plotting_times, E_high_BATV, color = color_BATV, ls = '--', label ='BATV high MOI')

# Compute 95% variability
med, lb95, ub95, lb65, ub65 = np.percentile(Es_low_BUNV,[50,2.5,100-2.5, 17.5, 100-17.5],axis=0)
# Show variability as shaded area
ax.fill_between(plotting_times,ub95,lb95, color=color_BUNV, zorder=-100, alpha = 0.5)

# Compute 95% variability
med, lb95, ub95, lb65, ub65 = np.percentile(Es_high_BUNV,[50,2.5,100-2.5, 17.5, 100-17.5],axis=0)
# Show variability as shaded area
ax.fill_between(plotting_times,ub95,lb95, color=color_BUNV, zorder=-100, alpha = 0.5)

# Compute 95% variability
med, lb95, ub95, lb65, ub65 = np.percentile(Es_low_BATV,[50,2.5,100-2.5, 17.5, 100-17.5],axis=0)
# Show variability as shaded area
ax.fill_between(plotting_times,ub95,lb95, color=color_BATV, zorder=-100, alpha = 0.5)

# Compute 95% variability
med, lb95, ub95, lb65, ub65 = np.percentile(Es_high_BATV,[50,2.5,100-2.5, 17.5, 100-17.5],axis=0)
# Show variability as shaded area
ax.fill_between(plotting_times,ub95,lb95, color=color_BATV, zorder=-100, alpha = 0.5)


ax.set_xlabel('Time (hpi)')
ax.set_xticks(np.arange(0, 175, 24))
ax.set_ylabel('Eclipse phase cells')
ax.set_ylim(0, max_cells)

hand1,labs1 = ax.get_legend_handles_labels()

ax.legend(hand1,labs1,frameon=False,handlelength=1.0, loc='center left', bbox_to_anchor=(1, 0.5))


# Infectious cells, I
ax = gridfig.subaxes(2)

ax.text(
    0.95, 0.95, "C",
    transform=ax.transAxes,
    ha="right",
    va="top",
    fontsize=24,
    fontweight="bold"
)

# Plot best fit
ax.plot(plotting_times, I_low_BUNV, color = color_BUNV)
ax.plot(plotting_times, I_high_BUNV, color = color_BUNV, ls = '--')
ax.plot(plotting_times, I_low_BATV, color = color_BATV)
ax.plot(plotting_times, I_high_BATV, color = color_BATV, ls = '--')

# Compute 95% variability
med, lb95, ub95, lb65, ub65 = np.percentile(Is_low_BUNV,[50,2.5,100-2.5, 17.5, 100-17.5],axis=0)
# Show variability as shaded area
ax.fill_between(plotting_times,ub95,lb95, color=color_BUNV, zorder=-100, alpha = 0.5)

# Compute 95% variability
med, lb95, ub95, lb65, ub65 = np.percentile(Is_high_BUNV,[50,2.5,100-2.5, 17.5, 100-17.5],axis=0)
# Show variability as shaded area
ax.fill_between(plotting_times,ub95,lb95, color=color_BUNV, zorder=-100, alpha = 0.5)

# Compute 95% variability
med, lb95, ub95, lb65, ub65 = np.percentile(Is_low_BATV,[50,2.5,100-2.5, 17.5, 100-17.5],axis=0)
# Show variability as shaded area
ax.fill_between(plotting_times,ub95,lb95, color=color_BATV, zorder=-100, alpha = 0.5)

# Compute 95% variability
med, lb95, ub95, lb65, ub65 = np.percentile(Is_high_BATV,[50,2.5,100-2.5, 17.5, 100-17.5],axis=0)
# Show variability as shaded area
ax.fill_between(plotting_times,ub95,lb95, color=color_BATV, zorder=-100, alpha = 0.5)


ax.set_xlabel('Time (hpi)')
ax.set_xticks(np.arange(0, 175, 24))
ax.set_ylabel('Infectious cells')
ax.set_ylim(0, max_cells)



# Infectious virus (units SIN)   
ax = gridfig.subaxes(3)

ax.text(
    0.95, 0.95, "D",
    transform=ax.transAxes,
    ha="right",
    va="top",
    fontsize=24,
    fontweight="bold"
)

# Plot best fit
ax.semilogy(inc_times, V_inc_low_SIN_A_BUNV, color = color_BUNV, zorder = 100)
ax.semilogy(inf_times, V_low_SIN_A_BUNV, color =color_BUNV, zorder = 100)

ax.semilogy(inc_times, V_inc_high_SIN_A_BUNV, color = color_BUNV, zorder = 100, ls = '--')
ax.semilogy(inf_times, V_high_SIN_A_BUNV, color =color_BUNV, zorder = 100, ls = '--')

ax.semilogy(inc_times, V_inc_low_SIN_A_BATV, color = color_BATV, zorder = 100)
ax.semilogy(inf_times, V_low_SIN_A_BATV, color =color_BATV, zorder = 100)

ax.semilogy(inc_times, V_inc_high_SIN_A_BATV, color = color_BATV, zorder = 100, ls = '--')
ax.semilogy(inf_times, V_high_SIN_A_BATV, color =color_BATV, zorder = 100, ls = '--')

# during incubation period
# Compute 95% variability
med, lb95, ub95, lb65, ub65 = np.percentile(Vs_inc_low_SIN_A_BUNV,[50,2.5,100-2.5, 17.5, 100-17.5],axis=0)
# Show variability as shaded area
ax.fill_between(inc_times,ub95,lb95,color=color_BUNV,zorder=-100, alpha = 0.5)

# after incubation period
# Compute 95% variability
med, lb95, ub95, lb65, ub65 = np.percentile(Vs_low_SIN_A_BUNV,[50,2.5,100-2.5, 17.5, 100-17.5],axis=0)
# Show variability as shaded area
ax.fill_between(inf_times,ub95,lb95,color=color_BUNV,zorder=-100, alpha = 0.5)


# during incubation period
# Compute 95% variability
med, lb95, ub95, lb65, ub65 = np.percentile(Vs_inc_high_SIN_A_BUNV,[50,2.5,100-2.5, 17.5, 100-17.5],axis=0)
# Show variability as shaded area
ax.fill_between(inc_times,ub95,lb95,color=color_BUNV,zorder=-100, alpha = 0.5)

# after incubation period
# Compute 95% variability
med, lb95, ub95, lb65, ub65 = np.percentile(Vs_high_SIN_A_BUNV,[50,2.5,100-2.5, 17.5, 100-17.5],axis=0)
# Show variability as shaded area
ax.fill_between(inf_times,ub95,lb95,color=color_BUNV,zorder=-100, alpha = 0.5)


# during incubation period
# Compute 95% variability
med, lb95, ub95, lb65, ub65 = np.percentile(Vs_inc_low_SIN_A_BATV,[50,2.5,100-2.5, 17.5, 100-17.5],axis=0)
# Show variability as shaded area
ax.fill_between(inc_times,ub95,lb95,color=color_BATV,zorder=-100, alpha = 0.5)

# after incubation period
# Compute 95% variability
med, lb95, ub95, lb65, ub65 = np.percentile(Vs_low_SIN_A_BATV,[50,2.5,100-2.5, 17.5, 100-17.5],axis=0)
# Show variability as shaded area
ax.fill_between(inf_times,ub95,lb95,color=color_BATV,zorder=-100, alpha = 0.5)


# during incubation period
# Compute 95% variability
med, lb95, ub95, lb65, ub65 = np.percentile(Vs_inc_high_SIN_A_BATV,[50,2.5,100-2.5, 17.5, 100-17.5],axis=0)
# Show variability as shaded area
ax.fill_between(inc_times,ub95,lb95,color=color_BATV,zorder=-100, alpha = 0.5)

# after incubation period
# Compute 95% variability
med, lb95, ub95, lb65, ub65 = np.percentile(Vs_high_SIN_A_BATV,[50,2.5,100-2.5, 17.5, 100-17.5],axis=0)
# Show variability as shaded area
ax.fill_between(inf_times,ub95,lb95,color=color_BATV,zorder=-100, alpha = 0.5)


ax.plot([1,1], [10**0, 10**10], 'k--')

ax.set_xlabel('Time (hpi)')
ax.set_xticks(np.arange(0, 175, 24))
ax.set_ylabel('Viral titre (SIN/well)')
ax.set_ylim(1,10**10)

gridfig.fig.savefig('Fig9.pdf', bbox_inches='tight')
