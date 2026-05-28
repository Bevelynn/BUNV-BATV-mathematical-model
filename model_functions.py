import numpy as np
import phymcmc
import pandas
import scipy.integrate
import math
import midsin

class params(phymcmc.ParamStruct):
    def validate(self):
        """ Assess validity of model parameters. """
        if min(self.vector) < 0.0:
            raise ValueError(repr(self.pardict))
        
        if self.pardict['w']>1:
            raise ValueError(repr(self.pardict))
            
        if self.pardict['kappa']>self.pardict['nE']:
            raise ValueError(repr(self.pardict))
            
        if self.pardict['tauI']>500:
            raise ValueError(repr(self.pardict))
            
        if not 0<int(round(self.pardict['nE']))<61:
            raise ValueError(repr(self.pardict))
            
        if not 0<int(round(self.pardict['nI']))<61:
            raise ValueError(repr(self.pardict))
            
        # only for BUNV
        # if self.pardict['gamma']*self.pardict['V0high']>self.pardict['V0RNA']:
        #     raise ValueError(repr(self.pardict))
            
        # if self.pardict['gamma']*self.pardict['p']>self.pardict['pRNA']:
        #     raise ValueError(repr(self.pardict))
    
def my_solve_ivp(odefunc, tspan, y0, teval, method='BDF', dense_output=False, events=None, vectorized=False, **options):
    try:
        res= scipy.integrate.solve_ivp(odefunc, t_span = tspan, y0 = y0, method=method,t_eval = teval, dense_output = dense_output, events = events, vectorized = vectorized, **options)
    except:
        raise phymcmc.ODEintegrationError('** WARNING! solve_ivp raised an exception.')
    if res.success:
        return res
    raise phymcmc.ODEintegrationError('** WARNING! solve_ivp integration Error:\n %s'%repr(res))


class model(phymcmc.base_model):
    def derivative(self, t, x, inc):
        """ Return the derivative of each variable of the model. """
        T, E, I, V, V_RNA, R = (x[0], x[1:self.pdic['nE'] + 1], x[self.pdic['nE'] + 1: self.pdic['nE'] + self.pdic['nI'] + 1], x[self.pdic['nE'] + self.pdic['nI'] + 1], x[self.pdic['nE'] + self.pdic['nI'] + 2], x[self.pdic['nE'] + self.pdic['nI'] + 3])
        
        #list of the eclipse stages that will jump into E2, E3, ..., EnE if a virion enters
        if self.pdic['kappa']==0:
            jump_poss = 0
            jump_list = [0]*(self.pdic['nE']-1)
        else:
            jump_poss = 1
            # note that the length of jump_list needs to be nE-1, so this doesn't work if jump=0
            jump_list = []
            for _ in range(self.pdic['kappa']-1):
                jump_list.append(0)
            for idx in range(self.pdic['nE']-self.pdic['kappa']):
                jump_list.append(E[idx])
        
        # hazard function for natural cell death
        # if simulating the incubation, t is true time
        if inc:
            h = self.deltN**self.pdic['nN']*t**(self.pdic['nN']-1)/(math.factorial(self.pdic['nN']-1)*sum((self.deltN*t)**k/math.factorial(k) for k in range(self.pdic['nN'])))
        # if simulating after the incubation, t + 1 is the true time in hpi
        else:
            h = self.deltN**self.pdic['nN']*(t+1)**(self.pdic['nN']-1)/(math.factorial(self.pdic['nN']-1)*sum((self.deltN*(t+1))**n/math.factorial(n) for n in range(self.pdic['nN'])))
        
        
        #uninfected cells
        dT = self.pdic['lambda']*R*T - h*T - self.pdic['gamma']*T*(self.pdic['beta']/self.pdic['s'])*V
        #eclipse phase cells
        dE1 = self.pdic['lambda']*R*E[0] - h*E[0] + self.pdic['gamma']*T*(self.pdic['beta']/self.pdic['s'])*V - self.deltE*E[0] - jump_poss*self.pdic['gamma']*(self.pdic['beta']/self.pdic['s'])*V*E[0]
        dEi = self.pdic['lambda']*R*E[1:] - h*E[1:] - self.deltE*np.diff(E) - jump_poss*self.pdic['gamma']*(self.pdic['beta']/self.pdic['s'])*V*E[1:] + self.pdic['gamma']*(self.pdic['beta']/self.pdic['s'])*V*np.array(jump_list)
        #infectious cells
        dI1 = - h*I[0] + self.deltE*E[-1] - self.deltI*I[0] + jump_poss*self.pdic['gamma']*(self.pdic['beta']/self.pdic['s'])*V*np.sum(E[self.pdic['nE']-self.pdic['kappa']:])
        dIi = - h*I[1:] - self.deltI*np.diff(I)
        #infectious virus (units SIN)
        dV = self.pdic['p']*np.sum(I) - (self.pdic['c'] + (self.pdic['beta']/self.pdic['s'])*(T + np.sum(E) + np.sum(I)))*V
        #RNA copies of each segment
        dV_RNA = self.pdic['pRNA']*np.sum(I) - (self.pdic['beta']/self.pdic['s'])*(T + np.sum(E) + np.sum(I))*V_RNA
        #resources
        dR = - (self.pdic['mu']/self.pdic['s'])*(T + np.sum(E) + np.sum(I))*R
        
        return np.hstack((dT, dE1, dEi, dI1, dIi, dV, dV_RNA, dR))
    
    def get_solution(self, inc_times, inf_times, T0, V0, V0RNA):
        """ Solve the model and obtain the model-generated prediction. """
        self.pdic = self.params.pardict
    	# must be an integer
        self.pdic['nN'] = int(round(self.pdic['nN']))
        self.pdic['nE'] = int(round(self.pdic['nE']))
        self.pdic['nI'] = int(round(self.pdic['nI']))
        self.pdic['kappa'] = int(round(self.pdic['kappa']))
        self.deltN = self.pdic['nN']/self.pdic['tauN']
        self.deltE = self.pdic['nE']/self.pdic['tauE']
        self.deltI = self.pdic['nI']/self.pdic['tauI']
        
        #initial conditions at start of incubation period
        R0_inc = [T0]
        for _ in range(self.pdic['nE'] + self.pdic['nI']):
            R0_inc.append(0)
        R0_inc.append(V0)
        R0_inc.append(V0RNA)
        R0_inc.append(1)

        #solve the system of ODEs to get the output of the model up to 1 hour
        sol_inc = my_solve_ivp(self.derivative, tspan=[0, 1], y0=R0_inc, teval=inc_times, args = (True,))
        
        T_inc = sol_inc.y[0,:]
        E_inc = sum(sol_inc.y[i,:] for i in range(1, self.pdic['nE'] + 1))
        I_inc = sum(sol_inc.y[i,:] for i in range(self.pdic['nE'] + 1, self.pdic['nE'] + self.pdic['nI'] + 1))
        V_inc_SIN_A = sol_inc.y[self.pdic['nE'] + self.pdic['nI'] + 1,:]
        V_inc_SIN_B = V_inc_SIN_A*self.pdic['alpha']
        V_inc_RNA = sol_inc.y[self.pdic['nE'] + self.pdic['nI'] + 2,:]
        
        #initial conditions at start of 'infection' period
        R0 = sol_inc.y[:, -1].copy()
        #number of target cells, eclipse phase cells, infectious phase cells, and resouce level come directly from incubation simulation
        #set the amount of infectious virus and RNA in supernatant after wash
        R0[-3] = self.pdic['w']*R0[-3]
        R0[-2] = self.pdic['w']*R0[-2]
        #get the output of the model at the times post infection
        sol = my_solve_ivp(self.derivative, tspan=[0, inf_times[-1]], y0=R0, teval=inf_times, args = (False,))
        
        # variables for plotting
        # take the incubation solution without final point because we want to remove the extra 1h time point for plotting
        T = np.concatenate((T_inc[:-1], sol.y[0,:]))
        E = np.concatenate((E_inc[:-1], sum(sol.y[i,:] for i in range(1, self.pdic['nE'] + 1))))
        I = np.concatenate((I_inc[:-1], sum(sol.y[i,:] for i in range(self.pdic['nE'] + 1, self.pdic['nE'] + self.pdic['nI'] + 1))))
        # virus after incubation 
        V_SIN_A = sol.y[self.pdic['nE'] + self.pdic['nI'] + 1,:]
        V_SIN_B = V_SIN_A*self.pdic['alpha']
        V_RNA = sol.y[self.pdic['nE'] + self.pdic['nI'] + 2,:]
        
        return T, E, I, V_inc_SIN_A, V_inc_SIN_B, V_inc_RNA, V_SIN_A, V_SIN_B, V_RNA
    
    
def get_data_BUNV():
    
    fn = 'BUNV_data.xlsx'
    
    df_lowR1 = pandas.read_excel(fn,sheet_name='low MOI R1',header=0)
    df_lowR2 = pandas.read_excel(fn,sheet_name='low MOI R2',header=0)
    df_lowR3 = pandas.read_excel(fn,sheet_name='low MOI R3',header=0)
    df_lowR4 = pandas.read_excel(fn,sheet_name='low MOI R4',header=0)
    
    df_highR1 = pandas.read_excel(fn,sheet_name='high MOI R1',header=0)
    df_highR2 = pandas.read_excel(fn,sheet_name='high MOI R2',header=0)
    df_highR3 = pandas.read_excel(fn,sheet_name='high MOI R3',header=0)
    df_highR4 = pandas.read_excel(fn,sheet_name='high MOI R4',header=0)
    
    df_cell_no_low = pandas.read_excel(fn,sheet_name='low MOI cell number',header=0)
    df_cell_no_high = pandas.read_excel(fn,sheet_name='high MOI cell number',header=0)
    
    df_RNA_S = pandas.read_excel(fn,sheet_name='RNA S',header=0)
    df_RNA_M = pandas.read_excel(fn,sheet_name='RNA M',header=0)
    df_RNA_L = pandas.read_excel(fn,sheet_name='RNA L',header=0)
    
    df_decayR1 = pandas.read_excel(fn,sheet_name='decay R1',header=0)
    df_decayR2 = pandas.read_excel(fn,sheet_name='decay R2',header=0)
    df_decayR3 = pandas.read_excel(fn,sheet_name='decay R3',header=0)
    
    dat = dict(
    		#all times and units in hours, counting zero hours as the start of the incubation period
            times_decay = df_decayR1['t'].to_numpy(dtype=float),
            times_R1 = df_lowR1['t'][3:].to_numpy(dtype=float),
            times_R2 = df_lowR2['t'][3:].to_numpy(dtype=float),
            times_R34 = df_lowR3['t'][2:].to_numpy(dtype=float),
            times_all = df_cell_no_low['t'].to_numpy(dtype=float)
            )
    
    # Dilution factors for ED assay
    # dil is 1e[0,-8] for the infection experiments by steps of 1
    Dils_inf = np.array([1.e+00, 1.e-01, 1.e-02, 1.e-03, 1.e-04, 1.e-05, 1.e-06, 1.e-07, 1.e-08])
    # dil is 1e[-1,-6] for the decay experiments by steps of 1
    Dils_decay = np.array([1.e-01, 1.e-02, 1.e-03, 1.e-04, 1.e-05, 1.e-06])
    
    # Number of positive (infected) wells in the ED assay (for all time points and dilutions, in an array)
    dat['ki_low_R1'] = df_lowR1[Dils_inf].to_numpy()
    dat['ki_low_R2'] = df_lowR2[Dils_inf].to_numpy()
    dat['ki_low_R3'] = df_lowR3[Dils_inf].to_numpy()
    dat['ki_low_R4'] = df_lowR4[Dils_inf].to_numpy()
    
    dat['ki_high_R1'] = df_highR1[Dils_inf].to_numpy()
    dat['ki_high_R2'] = df_highR2[Dils_inf].to_numpy()
    dat['ki_high_R3'] = df_highR3[Dils_inf].to_numpy()
    dat['ki_high_R4'] = df_highR4[Dils_inf].to_numpy()

    dat['cells_low_R1'] = df_cell_no_low['R1'].to_numpy()
    dat['cells_low_R2'] = df_cell_no_low['R2'].to_numpy()
    dat['cells_low_R3'] = df_cell_no_low['R3'].to_numpy()
    dat['cells_low_R4'] = df_cell_no_low['R4'].to_numpy()
    
    dat['cells_high_R1'] = df_cell_no_high['R1'].to_numpy()
    dat['cells_high_R2'] = df_cell_no_high['R2'].to_numpy()
    dat['cells_high_R3'] = df_cell_no_high['R3'].to_numpy()
    dat['cells_high_R4'] = df_cell_no_high['R4'].to_numpy()
    
    dat['ki_decay_R1'] = df_decayR1[Dils_decay].to_numpy()
    dat['ki_decay_R2'] = df_decayR2[Dils_decay].to_numpy()
    dat['ki_decay_R3'] = df_decayR3[Dils_decay].to_numpy()
    
    # the RNA data values give the number of RNA copies per microlitre of reaction mix in the pcr
    # multiply by this factor to convert to copy numbers in the whole well
    # the total reaction mix was 15ul, 1.25ul of which came from the RNA extracted from the sample
    # the total supernatant was 2ml
    factor = 15/1.25*2000
    
    dat['ext_RNA_S_R1'] = df_RNA_S['R1'].to_numpy()*factor
    dat['ext_RNA_S_R2'] = df_RNA_S['R2'].to_numpy()*factor
    dat['ext_RNA_S_R3'] = df_RNA_S['R3'].to_numpy()*factor
    dat['ext_RNA_S_R4'] = df_RNA_S['R4'].to_numpy()*factor

    dat['ext_RNA_M_R1'] = df_RNA_M['R1'].to_numpy()*factor
    dat['ext_RNA_M_R2'] = df_RNA_M['R2'].to_numpy()*factor
    dat['ext_RNA_M_R3'] = df_RNA_M['R3'].to_numpy()*factor
    dat['ext_RNA_M_R4'] = df_RNA_M['R4'].to_numpy()*factor

    dat['ext_RNA_L_R1'] = df_RNA_L['R1'].to_numpy()*factor
    dat['ext_RNA_L_R2'] = df_RNA_L['R2'].to_numpy()*factor
    dat['ext_RNA_L_R3'] = df_RNA_L['R3'].to_numpy()*factor
    dat['ext_RNA_L_R4'] = df_RNA_L['R4'].to_numpy()*factor
    
    
    # getting predictions of SIN using midSIN
    
    # Inoculum volume (in mL) for each well in ED assay
    VinocA = 0.045
    VinocB = 0.05
    Vinoc_decay = 0.1
    
    # number of wells for each dilution
    n = 6
    
    ntotR1 = n*np.ones_like(dat['ki_low_R1'][0,:])
    ntotR2 = n*np.ones_like(dat['ki_low_R2'][0,:])
    ntotR34 = n*np.ones_like(dat['ki_low_R3'][0,:])
    ntotdecay = n*np.ones_like(dat['ki_decay_R1'][0,:])
    
    #volume of superatant over time in the decay assay
    sR1 = df_decayR1['volume (ml)'].to_numpy(dtype=float)
    sR2 = df_decayR2['volume (ml)'].to_numpy(dtype=float)
    sR3 = df_decayR3['volume (ml)'].to_numpy(dtype=float)
    
    replicates = ['low_R1', 'high_R1', 'low_R2', 'high_R2', 
                  'low_R3', 'high_R3', 'low_R4', 'high_R4',
                  'decay_R1', 'decay_R2', 'decay_R3']
    
    Vinocs = {'low_R1': VinocA, 'high_R1': VinocA, 'low_R2': VinocA, 'high_R2': VinocA,
              'low_R3': VinocB, 'high_R3': VinocB, 'low_R4': VinocB, 'high_R4': VinocB,
              'decay_R1': Vinoc_decay, 'decay_R2': Vinoc_decay, 'decay_R3': Vinoc_decay}
    
    Dils = {'low_R1': Dils_inf, 'high_R1': Dils_inf, 'low_R2': Dils_inf, 'high_R2': Dils_inf,
              'low_R3': Dils_inf, 'high_R3': Dils_inf, 'low_R4': Dils_inf, 'high_R4': Dils_inf,
              'decay_R1': Dils_decay, 'decay_R2': Dils_decay, 'decay_R3': Dils_decay}
    
    ns = {'low_R1': ntotR1, 'high_R1': ntotR1, 'low_R2': ntotR2, 'high_R2': ntotR2,
          'low_R3': ntotR34, 'high_R3': ntotR34, 'low_R4': ntotR34, 'high_R4': ntotR34,
          'decay_R1': ntotdecay, 'decay_R2': ntotdecay, 'decay_R3': ntotdecay}
    
    sup_vols = {'low_R1': 2, 'high_R1': 2, 'low_R2': 2, 'high_R2': 2,
          'low_R3': 2, 'high_R3': 2, 'low_R4': 2, 'high_R4': 2,
          'decay_R1': sR1, 'decay_R2': sR2, 'decay_R3': sR3}
    
    for replicate in replicates:
        sinass = [] # To hold the full midSIN assay output
        tmp = [] # To temporarily hold midSIN mode, mean, bounds
    
        for row in dat['ki_%s' %replicate]:
            #Keep the midSIN assay outcome for each row
            sinass.append(midsin.Assay(Vinocs[replicate], Dils[replicate].max(), 0.1, row, ns[replicate]))
            #Keep the key values for each midSIN assay
            tmp.append( sup_vols[replicate]*10**np.array([sinass[-1].pack['mode'], sinass[-1].pack['mean']]+sinass[-1].pack['bounds']) )
    
        dat['df_midsin_%s' %replicate] = pandas.DataFrame(data=np.vstack(tmp), columns=['mode','mean','lb68','ub68','lb95','ub95'])
        dat['df_midsin_%s' %replicate].fillna(1, inplace=True)
    
    return dat


def get_data_BATV():
    
    fn = 'BATV_data.xlsx'
    
    df_cell_no = pandas.read_excel(fn,sheet_name='cell number',header=0)
    
    df_decayR1 = pandas.read_excel(fn,sheet_name='decay R1',header=0)
    df_decayR2 = pandas.read_excel(fn,sheet_name='decay R2',header=0)
    df_decayR3 = pandas.read_excel(fn,sheet_name='decay R3',header=0)
    
    dat = dict(
            df_low_R1 = pandas.read_excel(fn,sheet_name='low MOI R1',header=0),
            df_low_R2 = pandas.read_excel(fn,sheet_name='low MOI R2',header=0),
            df_low_R3 = pandas.read_excel(fn,sheet_name='low MOI R3',header=0),
            df_low_R4 = pandas.read_excel(fn,sheet_name='low MOI R4',header=0),
            df_low_R5 = pandas.read_excel(fn,sheet_name='low MOI R5',header=0),

            df_high_R1 = pandas.read_excel(fn,sheet_name='high MOI R1',header=0),
            df_high_R2 = pandas.read_excel(fn,sheet_name='high MOI R2',header=0),
            df_high_R3 = pandas.read_excel(fn,sheet_name='high MOI R3',header=0),
            df_high_R4 = pandas.read_excel(fn,sheet_name='high MOI R4',header=0),
            df_high_R5 = pandas.read_excel(fn,sheet_name='high MOI R5',header=0),
            
    		#all times and units in hours, counting zero hours as the start of the incubation period
            times_decay = df_decayR1['t'].to_numpy(dtype=float),
            times_all = df_cell_no['t'].to_numpy(dtype=float)
            )
    
    replicates = ['low_R1', 'high_R1', 'low_R2', 'high_R2', 
                  'low_R3', 'high_R3', 'low_R4', 'high_R4',
                  'low_R5', 'high_R5']
    
    for replicate in replicates:
        # keep only rows where the ED assay outcome is not null
        dat['df_%s' %replicate] = dat['df_%s' %replicate][dat['df_%s' %replicate][1].notna()]
        # get the set of times for this replicate
        dat['times_%s' %replicate] = dat['df_%s' %replicate]['t'].to_numpy(dtype=float)
    
    # Dilution factors for ED assay
    # dil is 1e[0,-8] for the infection experiments by steps of 1
    Dils_inf = np.array([1.e+00, 1.e-01, 1.e-02, 1.e-03, 1.e-04, 1.e-05, 1.e-06, 1.e-07, 1.e-08])
    # dil is 1e[-1,-6] for the decay experiments by steps of 1
    Dils_decay = np.array([1.e-01, 1.e-02, 1.e-03, 1.e-04, 1.e-05, 1.e-06])
    
    # Number of positive (infected) wells in the ED assay (for all time points and dilutions, in an array)
    for replicate in replicates:
        dat['ki_%s' %replicate] = dat['df_%s' %replicate][Dils_inf].to_numpy()

    dat['ki_decay_R1'] = df_decayR1[Dils_decay].to_numpy()
    dat['ki_decay_R2'] = df_decayR2[Dils_decay].to_numpy()
    dat['ki_decay_R3'] = df_decayR3[Dils_decay].to_numpy()
    
    dat['cells_low_R1'] = df_cell_no['low MOI R1'].to_numpy()
    dat['cells_low_R2'] = df_cell_no['low MOI R2'].to_numpy()
    dat['cells_low_R3'] = df_cell_no['low MOI R3'].to_numpy()
    dat['cells_low_R4'] = df_cell_no['low MOI R4'].to_numpy()
    dat['cells_low_R5'] = df_cell_no['low MOI R5'].to_numpy()
    
    dat['cells_high_R1'] = df_cell_no['high MOI R1'].to_numpy()
    dat['cells_high_R2'] = df_cell_no['high MOI R2'].to_numpy()
    dat['cells_high_R3'] = df_cell_no['high MOI R3'].to_numpy()
    dat['cells_high_R4'] = df_cell_no['high MOI R4'].to_numpy()
    dat['cells_high_R5'] = df_cell_no['high MOI R5'].to_numpy()
    
    
    # getting predictions of SIN using midSIN
    
    # Inoculum volume (in mL) for each well in ED assay
    Vinoc_inf = 0.05
    Vinoc_decay = 0.1
    
    # number of wells for each dilution
    n = 6
    
    ntotdecay = n*np.ones_like(dat['ki_decay_R1'][0,:])
    
    #volume of superatant over time in the decay assay
    sR1 = df_decayR1['volume (ml)'].to_numpy(dtype=float)
    sR2 = df_decayR2['volume (ml)'].to_numpy(dtype=float)
    sR3 = df_decayR3['volume (ml)'].to_numpy(dtype=float)
    
    Vinocs = {'low_R1': Vinoc_inf, 'high_R1': Vinoc_inf, 'low_R2': Vinoc_inf, 'high_R2': Vinoc_inf,
              'low_R3': Vinoc_inf, 'high_R3': Vinoc_inf, 'low_R4': Vinoc_inf, 'high_R4': Vinoc_inf,
              'low_R5': Vinoc_inf, 'high_R5': Vinoc_inf,
              'decay_R1': Vinoc_decay, 'decay_R2': Vinoc_decay, 'decay_R3': Vinoc_decay}
    
    Dils = {'low_R1': Dils_inf, 'high_R1': Dils_inf, 'low_R2': Dils_inf, 'high_R2': Dils_inf,
              'low_R3': Dils_inf, 'high_R3': Dils_inf, 'low_R4': Dils_inf, 'high_R4': Dils_inf,
              'low_R5': Dils_inf, 'high_R5': Dils_inf,
              'decay_R1': Dils_decay, 'decay_R2': Dils_decay, 'decay_R3': Dils_decay}
    
    ns = {'decay_R1': ntotdecay, 'decay_R2': ntotdecay, 'decay_R3': ntotdecay}
    
    for replicate in replicates:
        ns[replicate] = n*np.ones_like(dat['ki_%s' %replicate][0,:])
    
    sup_vols = {'low_R1': 2, 'high_R1': 2, 'low_R2': 2, 'high_R2': 2,
          'low_R3': 2, 'high_R3': 2, 'low_R4': 2, 'high_R4': 2,
          'low_R5': 2, 'high_R5': 2,
          'decay_R1': sR1, 'decay_R2': sR2, 'decay_R3': sR3}
    
    
    replicates = ['low_R1', 'high_R1', 'low_R2', 'high_R2', 
                  'low_R3', 'high_R3', 'low_R4', 'high_R4',
                  'low_R5', 'high_R5',
                  'decay_R1', 'decay_R2', 'decay_R3']
    
    for replicate in replicates:
        sinass = [] # To hold the full midSIN assay output
        tmp = [] # To temporarily hold midSIN mode, mean, bounds
    
        for row in dat['ki_%s' %replicate]:
            #Keep the midSIN assay outcome for each row
            sinass.append(midsin.Assay(Vinocs[replicate], Dils[replicate].max(), 0.1, row, ns[replicate]))
            #Keep the key values for each midSIN assay
            tmp.append( sup_vols[replicate]*10**np.array([sinass[-1].pack['mode'], sinass[-1].pack['mean']]+sinass[-1].pack['bounds']) )
    
        dat['df_midsin_%s' %replicate] = pandas.DataFrame(data=np.vstack(tmp), columns=['mode','mean','lb68','ub68','lb95','ub95'])
        dat['df_midsin_%s' %replicate].fillna(1, inplace=True)
    
    return dat