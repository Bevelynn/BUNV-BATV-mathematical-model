import numpy as np
import phymcmc.plot
import matplotlib.pyplot as plt
import matplotlib.patches as Patch
from phymcmc import corner

colors = ['tab:blue', 'tab:orange']
viruses = ['BUNV', 'BATV']

# Loading the posterior samples for both viruses (excluding the first column containing the log posterior probability)
posteriors = []
for virus in viruses:
   posteriors.append(np.loadtxt('%s_posterior.csv' %virus, delimiter = ',')[:, 1:])

for virus_idx, virus in enumerate(viruses):
    print(f'number of accepted parameter sets for {virus} is {len(posteriors[virus_idx])}')


# labels for Fig 6
param_labels = ['Initial number of cells, $T_0$',
                r'Infection rate, $\beta$ (ml $\cdot$ (cell $\cdot$ h)$^{-1})$',
                r'Viral decay rate, $c$ (h$^{-1}$)',
                r'Cells infected per SIN entry, $\gamma$ (cells $\cdot$ SIN$^{-1}$)',
                r'Production rate of SIN, $p$ (SIN $\cdot$ (cell $\cdot$ h)$^{-1}$)', 
                r'Production rate of RNA, $p_\mathrm{RNA}$ (RNA $\cdot$ (cell $\cdot$ h)$^{-1}$)', 
                r'No. eclipse stages, $n_E$',
                r'No. infectious stages, $n_I$',
                r'No. eclipse stages jumped due to re-infection, $\kappa$',
                r'Average eclipse period for singly infected cells, $\tau_E$ (h)',
                r'Average time until virus-induced death, $\tau_I$ (h)',
                'Wash factor, $\omega$', 
                r'SIN in BHKs per SIN in A549s, $\alpha_{\frac{\mathrm{BHK}}{\mathrm{A549}}}$'] 

# indicators for whether we want each parameter on a linear scale and whether it is an integer
linear_true = [False]*len(param_labels)
integer_true = [False]*len(param_labels)
linear_idxs = [6, 7, 8, 9, 10]
integer_idxs = [6, 7, 8]
for idx in linear_idxs:
    linear_true[idx] = True
for idx in integer_idxs:
    integer_true[idx] = True

# medians and 95% CI of each parameter for each virus
for param_idx, param in enumerate(param_labels):
    print(param)
    for virus_idx, virus in enumerate(viruses):
        med = np.percentile(posteriors[virus_idx][:, param_idx],50)
        lb = np.percentile(posteriors[virus_idx][:, param_idx],2.5)
        ub = np.percentile(posteriors[virus_idx][:, param_idx],97.5)
        max_val = max(posteriors[virus_idx][:, param_idx])
        min_val = min(posteriors[virus_idx][:, param_idx])
        print(virus, med, f'({lb},{ub})')
    
    
''' ##### plotting Fig 6 (marginal posterior histograms) ##### '''

gridfig = phymcmc.plot.grid_plot((5,3), rwidth=6, rheight=4.5, wspace=0.3, hspace = 0.3)

for param_idx, param_label in enumerate(param_labels):
    
    ax = gridfig.subaxes(param_idx)
    
    labs = []
    for virus_idx, virus_label in enumerate(viruses):
        if not integer_true[param_idx]:
            n = phymcmc.plot.hist(ax, posteriors[virus_idx][:, param_idx], bins=50, linear=linear_true[param_idx], color=colors[virus_idx])
        else:
            n = phymcmc.plot.hist(ax, posteriors[virus_idx][:, param_idx], bins=np.arange(min(posteriors[virus_idx][:, param_idx])-0.5, max(posteriors[virus_idx][:, param_idx])+1.5, 1), linear=linear_true[param_idx], color=colors[virus_idx])
        labs.append(virus_label)
    
    if not integer_true[param_idx]:
        ax.set_ylim(-200, 3000) 
    
    #T0 x-axis labels
    if param_idx == 0:
        ax.set_xticks([8*10**5, 9*10**5, 10**6, 1.1*10**6, 1.3*10**6], minor = True)
    
    #c x-axis labels
    if param_idx == 2:
        ax.set_xticks([0.04, 0.05, 0.06, 0.07],  minor = True)
     
    #p x-axis labels
    if param_idx == 4:
        ax.set_xticks([1, 10, 100])
        
    #pRNA x-axis labels
    if param_idx == 5:
        ax.set_xticks([10, 100])
        
    #w x-axis labels
    if param_idx == 11:
        ax.set_xticks([10**-4, 10**-3])
    
    if param_idx == 2:
        ax.legend(labs,frameon=False,handlelength=1.0, loc='center left', bbox_to_anchor=(1, 0.5))
        
    ax.set_xlabel(param_label)
    
gridfig.fig.savefig('Fig6.pdf', bbox_inches='tight')


''' ##### plotting Fig 7 (posterior histograms for dervied parameters) ##### '''

# labels for Fig 7
param_labels_derived = [r'Reduction in eclipse period due to re-infection, $\kappa\tau_E/n_E$ (h)', r'Basic reproduction number, $\tau_I p \gamma \frac{\beta T_0}{\beta T_0 + cs}$']
# indicators for whether we want a linear scale
linear_true = [True, False]

# volume of supernatant (ml)
s = 2

params_derived = [np.zeros((len(posteriors[0]), 2)), np.zeros((len(posteriors[1]), 2))]
for virus_idx, virus in enumerate(viruses):
    #jump time = kappa*tauE/nE
    params_derived[virus_idx][:, 0] = posteriors[virus_idx][:, 8]*posteriors[virus_idx][:,9]/posteriors[virus_idx][:, 6]
    #basic reproduction number = tauI*p*gamma*beta*T0/(beta*T0 + c*s)
    params_derived[virus_idx][:, 1] = posteriors[virus_idx][:, 10]*posteriors[virus_idx][:, 4]*posteriors[virus_idx][:, 3]*posteriors[virus_idx][:, 1]*posteriors[virus_idx][:, 0]/(posteriors[virus_idx][:, 1]*posteriors[virus_idx][:, 0] + posteriors[virus_idx][:, 2]*s)

# medians and 95% CI of each parameter for each virus
for param_idx, param in enumerate(param_labels_derived):
    print(param)
    for virus_idx, virus in enumerate(viruses):
        med = np.percentile(params_derived[virus_idx][:, param_idx],50)
        lb = np.percentile(params_derived[virus_idx][:, param_idx],2.5)
        ub = np.percentile(params_derived[virus_idx][:, param_idx],97.5)
        print(virus, med, f'({lb},{ub})')

gridfig = phymcmc.plot.grid_plot((1,2), rwidth=6, rheight=4.5)
for param_idx, param_label in enumerate(param_labels_derived):
    ax = gridfig.subaxes(param_idx)
    labs = []
    for virus_idx, virus_label in enumerate(viruses):
        n = phymcmc.plot.hist(ax, params_derived[virus_idx][:, param_idx], bins=50, linear=linear_true[param_idx], color=colors[virus_idx])
        labs.append(virus_label)
    if param_idx==1:
        ax.set_xticks([2000,  5000], [r'$2\times 10^{3}$', r'$5\times 10^{3}$'], minor = True)
        ax.legend(labs,frameon=False,handlelength=1.0, loc='center left', bbox_to_anchor=(1, 0.5))
            
    ax.set_xlabel(param_label)
        
gridfig.fig.savefig('Fig7.pdf', bbox_inches='tight')
    

''' ##### plotting Fig 8 ##### '''

# beta, p
param_x_idx, param_y_idx = 1, 4

x1 = posteriors[0][:, param_x_idx]
y1 = posteriors[0][:, param_y_idx]

x2 = posteriors[1][:, param_x_idx]
y2 = posteriors[1][:, param_y_idx]


# Create mask for values in x2 within these bounds
mask = (x2 >= 6*10**-8 ) & (x2 <= 10**-6)

# Apply mask to both arrays
x2_filtered = x2[mask]
y2_filtered = y2[mask]


plt.figure(figsize = (5, 5))

corner.hist2d(x1, y1, plot_datapoints = False, color = colors[0], fill_contours = True)
corner.hist2d(x2_filtered, y2_filtered, plot_datapoints = False, color = colors[1], fill_contours = True)

plt.xlim(min(min(x1), min(x2_filtered)), max(max(x1), max(x2_filtered)))
plt.ylim(min(min(y1), min(y2_filtered)), 10**2)
plt.xscale('log')
plt.yscale('log')
plt.xlabel(param_labels[param_x_idx])
plt.ylabel(param_labels[param_y_idx])

BUNV_patch  = Patch.Patch(color = colors[0], alpha=0.6, label = viruses[0])
BATV_patch = Patch.Patch(color = colors[1], alpha=0.6, label = viruses[1])
plt.legend(handles=[BUNV_patch, BATV_patch])
    
plt.savefig('Fig8.pdf', bbox_inches='tight')
    
        
