'''
    This code is used to replot Figure 4 of the main text.
    As this figure relies
    '''
# to make things py2 and py3 compatible
from __future__ import print_function, division

# major packages needed
import numpy as np
from scipy.stats import pearsonr, spearmanr

# import plotting-related packages
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
rc('text', usetex=True)

data_cellnames = np.loadtxt("/Users/rcubero/Dropbox/Peyrache_Data_NEVR3004/Results/data_cellnames",dtype=bytes,delimiter='\n').astype(str)
data_names = np.loadtxt("/Users/rcubero/Dropbox/Peyrache_Data_NEVR3004/MouseFilenames",dtype=bytes,delimiter='\n').astype(str)
data_color = np.loadtxt("/Users/rcubero/Dropbox/Peyrache_Data_NEVR3004/Results/data_color")
data_marker = np.loadtxt("/Users/rcubero/Dropbox/Peyrache_Data_NEVR3004/Results/data_marker",dtype=bytes,delimiter='\n').astype(str)

relevance = [];
information = [];
randomized_information = [];
mean_firing_rate = [];
mean_vector_length = [];
selectivity = [];
for data_name in data_names:
    relevance_directory = "/Users/rcubero/Dropbox/Peyrache_Data_NEVR3004/Results/%s/%s-HD"%(data_name,data_name)
    relevance += list(np.loadtxt('%s-relevance.d'%(relevance_directory)))
    
    calculation_directory = "/Users/rcubero/Dropbox/Peyrache_Data_NEVR3004/Temporal_Relevance/Results/%s-HD"%(data_name)
    information += list(np.loadtxt('%s-HD_information.d'%(calculation_directory)))
    randomized_information += list(np.loadtxt('%s-randomized_mean-HD_information.d'%(calculation_directory)))
    mean_vector_length += list(np.loadtxt('%s-mean_vector_length.d'%(calculation_directory)))
    mean_firing_rate += list(np.loadtxt('%s-mean_firing_rate.d'%(calculation_directory)))
    selectivity += list(np.loadtxt('%s-HD_selectivity.d'%(calculation_directory)))

relevance = np.array(relevance)
information = np.array(information)
randomized_information = np.array(randomized_information)
mean_vector_length = np.array(mean_vector_length)
mean_firing_rate = np.array(mean_firing_rate)
selectivity = np.array(selectivity)



fig = plt.figure(dpi=300)
fig.set_size_inches(35,20)

gs0 = gridspec.GridSpec(1, 2, wspace=0.15)

gs = gridspec.GridSpecFromSubplotSpec(3, 2, subplot_spec=gs0[0], wspace=0.25, hspace=0.2)

# Mouse12
Mouse_ToPlot = np.zeros(len(data_cellnames)).astype("bool")
Mouse_ToConsider = "Mouse12"
for i in np.arange(len(data_cellnames)):
    mouse_index, terminal_index = data_cellnames[i].split()
    mouse_index, experiment_index = mouse_index.split("-")
    terminal_index = terminal_index[terminal_index.find("T")+1 : terminal_index.find("C")]
    if (mouse_index == Mouse_ToConsider):
        if (int(terminal_index) in np.arange(1,9,1)): Mouse_ToPlot[i] = True;

non_nan = np.where(~np.isnan(relevance[Mouse_ToPlot]))[0]
x_data = relevance[Mouse_ToPlot][non_nan]
information_data = (information[Mouse_ToPlot][non_nan]-randomized_information[Mouse_ToPlot][non_nan])/(mean_firing_rate[Mouse_ToPlot][non_nan]*0.001)
selectivity_data = selectivity[Mouse_ToPlot][non_nan]
s_data = mean_vector_length[Mouse_ToPlot][non_nan]
c_data = np.array(data_color)[Mouse_ToPlot][non_nan]
m_data = np.array(data_marker)[Mouse_ToPlot][non_nan]

axScatterPlot = plt.subplot(gs[0,0])
axScatterPlot.scatter(x_data, information_data, marker="o", s=300*s_data, edgecolors="red", facecolors="None", linewidth='3', label="ADn neurons, "+str(Mouse_ToConsider))
axScatterPlot.set_xlabel(r'multiscale relevance, $\mathcal{R}_t$ (Mats$^2$)', fontsize=16)
axScatterPlot.set_ylabel(r'head directional information, $I(s, \theta)$ (bits per spike)', fontsize=16)
axScatterPlot.legend(loc="upper left", scatterpoints=1, fontsize=16)
axScatterPlot.set_xlim(right=0.305)
axScatterPlot.patch.set_facecolor("white")
axScatterPlot.tick_params(labelsize=14)
r_pearson, p_pearson = pearsonr(x_data, information_data)
r_spearman, p_spearman = spearmanr(x_data, information_data)
axScatterPlot.annotate('$\\rho_p=$ %.3f, $P=$ %.2e \n $\\rho_s=$ %.3f, $P=$ %.2e'%(r_pearson, p_pearson, r_spearman, p_spearman),(0.24,1.40), fontsize=16)
axScatterPlot.text(-0.2, 1.05, "A", transform=axScatterPlot.transAxes, fontsize=20, fontweight='bold', va='top')

axScatterPlot = plt.subplot(gs[0,1])
axScatterPlot.scatter(x_data, selectivity_data, marker="o", s=300*s_data, edgecolors="red", facecolors="None", linewidth='3', label="ADn neurons, "+str(Mouse_ToConsider))
axScatterPlot.set_xlabel(r'multiscale relevance, $\mathcal{R}_t$ (Mats$^2$)', fontsize=16)
axScatterPlot.set_ylabel(r'head directional sparsity, $sp_{\theta}$', fontsize=16)
axScatterPlot.legend(loc="upper left", scatterpoints=1, fontsize=18)
axScatterPlot.set_xlim(right=0.305)
axScatterPlot.patch.set_facecolor("white")
axScatterPlot.tick_params(labelsize=14)
r_pearson, p_pearson = pearsonr(x_data, selectivity_data)
r_spearman, p_spearman = spearmanr(x_data, selectivity_data)
axScatterPlot.annotate('$\\rho_p=$ %.3f, $P=$ %.2e \n $\\rho_s=$ %.3f, $P=$ %.2e'%(r_pearson, p_pearson, r_spearman, p_spearman),(0.24,0.70), fontsize=16)


# Mouse17
Mouse_ToPlot = np.zeros(len(data_cellnames)).astype("bool")
Mouse_ToConsider = "Mouse17"
for i in np.arange(len(data_cellnames)):
    mouse_index, terminal_index = data_cellnames[i].split()
    mouse_index, experiment_index = mouse_index.split("-")
    terminal_index = terminal_index[terminal_index.find("T")+1 : terminal_index.find("C")]
    if (mouse_index == Mouse_ToConsider):
        if (int(terminal_index) in np.arange(1,9,1)): Mouse_ToPlot[i] = True;

non_nan = np.where(~np.isnan(relevance[Mouse_ToPlot]))[0]
x_data = relevance[Mouse_ToPlot][non_nan]
information_data = (information[Mouse_ToPlot][non_nan]-randomized_information[Mouse_ToPlot][non_nan])/(mean_firing_rate[Mouse_ToPlot][non_nan]*0.001)
selectivity_data = selectivity[Mouse_ToPlot][non_nan]
s_data = mean_vector_length[Mouse_ToPlot][non_nan]
c_data = np.array(data_color)[Mouse_ToPlot][non_nan]
m_data = np.array(data_marker)[Mouse_ToPlot][non_nan]

axScatterPlot = plt.subplot(gs[1,0])
axScatterPlot.scatter(x_data, information_data, marker="o", s=300*s_data, edgecolors="red", facecolors="None", linewidth='3', label="ADn neurons, "+str(Mouse_ToConsider))
axScatterPlot.set_xlabel(r'multiscale relevance, $\mathcal{R}_t$ (Mats$^2$)', fontsize=16)
axScatterPlot.set_ylabel(r'head directional information, $I(s, \theta)$ (bits per spike)', fontsize=16)
axScatterPlot.legend(loc="upper left", scatterpoints=1, fontsize=16)
axScatterPlot.set_xlim(right=0.305)
axScatterPlot.patch.set_facecolor("white")
axScatterPlot.tick_params(labelsize=14)
r_pearson, p_pearson = pearsonr(x_data, information_data)
r_spearman, p_spearman = spearmanr(x_data, information_data)
axScatterPlot.annotate('$\\rho_p=$ %.3f, $P=$ %.2e \n $\\rho_s=$ %.3f, $P=$ %.2e'%(r_pearson, p_pearson, r_spearman, p_spearman),(0.239,1.10), fontsize=16)
axScatterPlot.text(-0.2, 1.05, "B", transform=axScatterPlot.transAxes, fontsize=20, fontweight='bold', va='top')

axScatterPlot = plt.subplot(gs[1,1])
axScatterPlot.scatter(x_data, selectivity_data, marker="o", s=300*s_data, edgecolors="red", facecolors="None", linewidth='3', label="ADn neurons, "+str(Mouse_ToConsider))
axScatterPlot.set_xlabel(r'multiscale relevance, $\mathcal{R}_t$ (Mats$^2$)', fontsize=16)
axScatterPlot.set_ylabel(r'head directional sparsity, $sp_{\theta}$', fontsize=16)
axScatterPlot.legend(loc="upper left", scatterpoints=1, fontsize=18)
axScatterPlot.set_xlim(right=0.305)
axScatterPlot.patch.set_facecolor("white")
axScatterPlot.tick_params(labelsize=14)
r_pearson, p_pearson = pearsonr(x_data, selectivity_data)
r_spearman, p_spearman = spearmanr(x_data, selectivity_data)
axScatterPlot.annotate('$\\rho_p=$ %.3f, $P=$ %.2e \n $\\rho_s=$ %.3f, $P=$ %.2e'%(r_pearson, p_pearson, r_spearman, p_spearman),(0.239,0.66), fontsize=16)


# Mouse20
Mouse_ToPlot = np.zeros(len(data_cellnames)).astype("bool")
Mouse_ToConsider = "Mouse20"
for i in np.arange(len(data_cellnames)):
    mouse_index, terminal_index = data_cellnames[i].split()
    mouse_index, experiment_index = mouse_index.split("-")
    terminal_index = terminal_index[terminal_index.find("T")+1 : terminal_index.find("C")]
    if (mouse_index == Mouse_ToConsider):
        if (int(terminal_index) in np.arange(1,9,1)): Mouse_ToPlot[i] = True;

non_nan = np.where(~np.isnan(relevance[Mouse_ToPlot]))[0]
x_data = relevance[Mouse_ToPlot][non_nan]
information_data = (information[Mouse_ToPlot][non_nan]-randomized_information[Mouse_ToPlot][non_nan])/(mean_firing_rate[Mouse_ToPlot][non_nan]*0.001)
selectivity_data = selectivity[Mouse_ToPlot][non_nan]
s_data = mean_vector_length[Mouse_ToPlot][non_nan]
c_data = np.array(data_color)[Mouse_ToPlot][non_nan]
m_data = np.array(data_marker)[Mouse_ToPlot][non_nan]

axScatterPlot = plt.subplot(gs[2,0])
axScatterPlot.scatter(x_data, information_data, marker="o", s=300*s_data, edgecolors="red", facecolors="None", linewidth='3', label="ADn neurons, "+str(Mouse_ToConsider))
axScatterPlot.set_xlabel(r'multiscale relevance, $\mathcal{R}_t$ (Mats$^2$)', fontsize=16)
axScatterPlot.set_ylabel(r'head directional information, $I(s, \theta)$ (bits per spike)', fontsize=16)
axScatterPlot.legend(loc="upper left", scatterpoints=1, fontsize=16)
axScatterPlot.set_xlim(right=0.305)
axScatterPlot.patch.set_facecolor("white")
axScatterPlot.tick_params(labelsize=14)
r_pearson, p_pearson = pearsonr(x_data, information_data)
r_spearman, p_spearman = spearmanr(x_data, information_data)
axScatterPlot.annotate('$\\rho_p=$ %.3f, $P=$ %.2e \n $\\rho_s=$ %.3f, $P=$ %.2e'%(r_pearson, p_pearson, r_spearman, p_spearman),(0.243,1.4), fontsize=16)
axScatterPlot.text(-0.2, 1.05, "C", transform=axScatterPlot.transAxes, fontsize=20, fontweight='bold', va='top')

axScatterPlot = plt.subplot(gs[2,1])
axScatterPlot.scatter(x_data, selectivity_data, marker="o", s=300*s_data, edgecolors="red", facecolors="None", linewidth='3', label="ADn neurons, "+str(Mouse_ToConsider))
axScatterPlot.set_xlabel(r'multiscale relevance, $\mathcal{R}_t$ (Mats$^2$)', fontsize=16)
axScatterPlot.set_ylabel(r'head directional sparsity, $sp_{\theta}$', fontsize=16)
axScatterPlot.legend(loc="upper left", scatterpoints=1, fontsize=18)
axScatterPlot.set_xlim(right=0.305)
axScatterPlot.patch.set_facecolor("white")
axScatterPlot.tick_params(labelsize=14)
r_pearson, p_pearson = pearsonr(x_data, selectivity_data)
r_spearman, p_spearman = spearmanr(x_data, selectivity_data)
axScatterPlot.annotate('$\\rho_p=$ %.3f, $P=$ %.2e \n $\\rho_s=$ %.3f, $P=$ %.2e'%(r_pearson, p_pearson, r_spearman, p_spearman),(0.243,0.68), fontsize=16)


gs = gridspec.GridSpecFromSubplotSpec(3, 2, subplot_spec=gs0[1], wspace=0.25, hspace=0.2)

# Mouse24
Mouse_ToPlot_ADn = np.zeros(len(data_cellnames)).astype("bool")
Mouse_ToPlot_PoS = np.zeros(len(data_cellnames)).astype("bool")
Mouse_ToConsider = "Mouse24"
for i in np.arange(len(data_cellnames)):
    mouse_index, terminal_index = data_cellnames[i].split()
    mouse_index, experiment_index = mouse_index.split("-")
    terminal_index = terminal_index[terminal_index.find("T")+1 : terminal_index.find("C")]
    if (mouse_index == Mouse_ToConsider):
        if (int(experiment_index) in [131213]):
            if (int(terminal_index) in np.arange(1,5,1)): Mouse_ToPlot_PoS[i] = True;
            elif (int(terminal_index) in np.arange(5,9,1)): Mouse_ToPlot_ADn[i] = True;
        elif (int(experiment_index) in [131216,131217,131218]):
            if (int(terminal_index) in np.arange(3,5,1)): Mouse_ToPlot_PoS[i] = True;
            elif (int(terminal_index) in np.arange(5,9,1)): Mouse_ToPlot_ADn[i] = True;

non_nan = np.where(~np.isnan(relevance[Mouse_ToPlot_ADn]))[0]
x_data_ADn = relevance[Mouse_ToPlot_ADn][non_nan]
information_data_ADn = (information[Mouse_ToPlot_ADn][non_nan]-randomized_information[Mouse_ToPlot_ADn][non_nan])/(mean_firing_rate[Mouse_ToPlot_ADn][non_nan]*0.001)
selectivity_data_ADn = selectivity[Mouse_ToPlot_ADn][non_nan]
s_data_ADn = mean_vector_length[Mouse_ToPlot_ADn][non_nan]

non_nan = np.where(~np.isnan(relevance[Mouse_ToPlot_PoS]))[0]
x_data_PoS = relevance[Mouse_ToPlot_PoS][non_nan]
information_data_PoS = (information[Mouse_ToPlot_PoS][non_nan]-randomized_information[Mouse_ToPlot_PoS][non_nan])/(mean_firing_rate[Mouse_ToPlot_PoS][non_nan]*0.001)
selectivity_data_PoS = selectivity[Mouse_ToPlot_PoS][non_nan]
s_data_PoS = mean_vector_length[Mouse_ToPlot_PoS][non_nan]

x_data = np.append(x_data_ADn, x_data_PoS)
information_data = np.append(information_data_ADn, information_data_PoS)
selectivity_data = np.append(selectivity_data_ADn, selectivity_data_PoS)
s_data = np.append(s_data_ADn, s_data_PoS)

axScatterPlot = plt.subplot(gs[0,0])
axScatterPlot.scatter(x_data_ADn, information_data_ADn, marker="o", s=300*s_data_ADn, edgecolors="red", facecolors="None", linewidth='3', label="ADn neurons, "+str(Mouse_ToConsider))
axScatterPlot.scatter(x_data_PoS, information_data_PoS, marker="o", s=300*s_data_PoS, edgecolors="blue", facecolors="None", linewidth='3', label="PoS neurons, "+str(Mouse_ToConsider))
axScatterPlot.set_xlabel(r'multiscale relevance, $\mathcal{R}_t$ (Mats$^2$)', fontsize=16)
axScatterPlot.set_ylabel(r'head directional information, $I(s, \theta)$ (bits per spike)', fontsize=16)
axScatterPlot.legend(loc="upper left", scatterpoints=1, fontsize=18)
axScatterPlot.set_xlim(right=0.305)
axScatterPlot.patch.set_facecolor("white")
axScatterPlot.tick_params(labelsize=14)
r_pearson, p_pearson = pearsonr(x_data, information_data)
r_spearman, p_spearman = spearmanr(x_data, information_data)
axScatterPlot.annotate('$\\rho_p=$ %.3f, $P=$ %.2e \n $\\rho_s=$ %.3f, $P=$ %.2e'%(r_pearson, p_pearson, r_spearman, p_spearman),(0.242,1.22), fontsize=16)
axScatterPlot.text(-0.2, 1.05, "D", transform=axScatterPlot.transAxes, fontsize=20, fontweight='bold', va='top')

axScatterPlot = plt.subplot(gs[0,1])
axScatterPlot.scatter(x_data_ADn, selectivity_data_ADn, marker="o", s=300*s_data_ADn, edgecolors="red", facecolors="None", linewidth='3', label="ADn neurons, "+str(Mouse_ToConsider))
axScatterPlot.scatter(x_data_PoS, selectivity_data_PoS, marker="o", s=300*s_data_PoS, edgecolors="blue", facecolors="None", linewidth='3', label="PoS neurons, "+str(Mouse_ToConsider))
axScatterPlot.set_xlabel(r'multiscale relevance, $\mathcal{R}_t$ (Mats$^2$)', fontsize=16)
axScatterPlot.set_ylabel(r'head directional sparsity, $sp_{\theta}$', fontsize=16)
axScatterPlot.legend(loc="upper left", scatterpoints=1, fontsize=18)
axScatterPlot.set_xlim(right=0.305)
axScatterPlot.patch.set_facecolor("white")
axScatterPlot.tick_params(labelsize=14)
r_pearson, p_pearson = pearsonr(x_data, selectivity_data)
r_spearman, p_spearman = spearmanr(x_data, selectivity_data)
axScatterPlot.annotate('$\\rho_p=$ %.3f, $P=$ %.2e \n $\\rho_s=$ %.3f, $P=$ %.2e'%(r_pearson, p_pearson, r_spearman, p_spearman),(0.242,0.6), fontsize=16)


# Mouse25
Mouse_ToPlot_ADn = np.zeros(len(data_cellnames)).astype("bool")
Mouse_ToPlot_PoS = np.zeros(len(data_cellnames)).astype("bool")
Mouse_ToConsider = "Mouse25"
for i in np.arange(len(data_cellnames)):
    mouse_index, terminal_index = data_cellnames[i].split()
    mouse_index, experiment_index = mouse_index.split("-")
    terminal_index = terminal_index[terminal_index.find("T")+1 : terminal_index.find("C")]
    if (mouse_index == Mouse_ToConsider):
        if (int(experiment_index) in [140123,140124,140128,140129,140130,140131,140203]):
            if (int(terminal_index) in np.arange(1,5,1)): Mouse_ToPlot_PoS[i] = True;
            elif (int(terminal_index) in np.arange(5,9,1)): Mouse_ToPlot_ADn[i] = True;
        elif (int(experiment_index) in [140204,140205,140206]):
            if (int(terminal_index) in np.arange(3,5,1)): Mouse_ToPlot_PoS[i] = True;
            elif (int(terminal_index) in np.arange(5,9,1)): Mouse_ToPlot_ADn[i] = True;

non_nan = np.where(~np.isnan(relevance[Mouse_ToPlot_ADn]))[0]
x_data_ADn = relevance[Mouse_ToPlot_ADn][non_nan]
information_data_ADn = (information[Mouse_ToPlot_ADn][non_nan]-randomized_information[Mouse_ToPlot_ADn][non_nan])/(mean_firing_rate[Mouse_ToPlot_ADn][non_nan]*0.001)
selectivity_data_ADn = selectivity[Mouse_ToPlot_ADn][non_nan]
s_data_ADn = mean_vector_length[Mouse_ToPlot_ADn][non_nan]

non_nan = np.where(~np.isnan(relevance[Mouse_ToPlot_PoS]))[0]
x_data_PoS = relevance[Mouse_ToPlot_PoS][non_nan]
information_data_PoS = (information[Mouse_ToPlot_PoS][non_nan]-randomized_information[Mouse_ToPlot_PoS][non_nan])/(mean_firing_rate[Mouse_ToPlot_PoS][non_nan]*0.001)
selectivity_data_PoS = selectivity[Mouse_ToPlot_PoS][non_nan]
s_data_PoS = mean_vector_length[Mouse_ToPlot_PoS][non_nan]

x_data = np.append(x_data_ADn, x_data_PoS)
information_data = np.append(information_data_ADn, information_data_PoS)
selectivity_data = np.append(selectivity_data_ADn, selectivity_data_PoS)
s_data = np.append(s_data_ADn, s_data_PoS)

axScatterPlot = plt.subplot(gs[1,0])
axScatterPlot.scatter(x_data_ADn, information_data_ADn, marker="o", s=300*s_data_ADn, edgecolors="red", facecolors="None", linewidth='3', label="ADn neurons, "+str(Mouse_ToConsider))
axScatterPlot.scatter(x_data_PoS, information_data_PoS, marker="o", s=300*s_data_PoS, edgecolors="blue", facecolors="None", linewidth='3', label="PoS neurons, "+str(Mouse_ToConsider))
axScatterPlot.set_xlabel(r'multiscale relevance, $\mathcal{R}_t$ (Mats$^2$)', fontsize=16)
axScatterPlot.set_ylabel(r'head directional information, $I(s, \theta)$ (bits per spike)', fontsize=16)
axScatterPlot.legend(loc="upper left", scatterpoints=1, fontsize=18)
axScatterPlot.set_xlim(right=0.305)
axScatterPlot.patch.set_facecolor("white")
axScatterPlot.tick_params(labelsize=14)
r_pearson, p_pearson = pearsonr(x_data, information_data)
r_spearman, p_spearman = spearmanr(x_data, information_data)
axScatterPlot.annotate('$\\rho_p=$ %.3f, $P=$ %.2e \n $\\rho_s=$ %.3f, $P=$ %.2e'%(r_pearson, p_pearson, r_spearman, p_spearman),(0.256,1.12), fontsize=16)
axScatterPlot.text(-0.2, 1.05, "E", transform=axScatterPlot.transAxes, fontsize=20, fontweight='bold', va='top')

axScatterPlot = plt.subplot(gs[1,1])
axScatterPlot.scatter(x_data_ADn, selectivity_data_ADn, marker="o", s=300*s_data_ADn, edgecolors="red", facecolors="None", linewidth='3', label="ADn neurons, "+str(Mouse_ToConsider))
axScatterPlot.scatter(x_data_PoS, selectivity_data_PoS, marker="o", s=300*s_data_PoS, edgecolors="blue", facecolors="None", linewidth='3', label="PoS neurons, "+str(Mouse_ToConsider))
axScatterPlot.set_xlabel(r'multiscale relevance, $\mathcal{R}_t$ (Mats$^2$)', fontsize=16)
axScatterPlot.set_ylabel(r'head directional sparsity, $sp_{\theta}$', fontsize=16)
axScatterPlot.legend(loc="upper left", scatterpoints=1, fontsize=18)
axScatterPlot.set_xlim(right=0.305)
axScatterPlot.patch.set_facecolor("white")
axScatterPlot.tick_params(labelsize=14)
r_pearson, p_pearson = pearsonr(x_data, selectivity_data)
r_spearman, p_spearman = spearmanr(x_data, selectivity_data)
axScatterPlot.annotate('$\\rho_p=$ %.3f, $P=$ %.2e \n $\\rho_s=$ %.3f, $P=$ %.2e'%(r_pearson, p_pearson, r_spearman, p_spearman),(0.256,0.63), fontsize=16)


# Mouse28
Mouse_ToPlot_ADn = np.zeros(len(data_cellnames)).astype("bool")
Mouse_ToPlot_PoS = np.zeros(len(data_cellnames)).astype("bool")
Mouse_ToConsider = "Mouse28"
for i in np.arange(len(data_cellnames)):
    mouse_index, terminal_index = data_cellnames[i].split()
    mouse_index, experiment_index = mouse_index.split("-")
    terminal_index = terminal_index[terminal_index.find("T")+1 : terminal_index.find("C")]
    if (mouse_index == Mouse_ToConsider):
        if (int(experiment_index) in [140310,140311,140312,140314,140317,140318]):
            if (int(terminal_index) in np.arange(1,8,1)): Mouse_ToPlot_PoS[i] = True;
            elif (int(terminal_index) in np.arange(8,12,1)): Mouse_ToPlot_ADn[i] = True;

non_nan = np.where(~np.isnan(relevance[Mouse_ToPlot_ADn]))[0]
x_data_ADn = relevance[Mouse_ToPlot_ADn][non_nan]
information_data_ADn = (information[Mouse_ToPlot_ADn][non_nan]-randomized_information[Mouse_ToPlot_ADn][non_nan])/(mean_firing_rate[Mouse_ToPlot_ADn][non_nan]*0.001)
selectivity_data_ADn = selectivity[Mouse_ToPlot_ADn][non_nan]
s_data_ADn = mean_vector_length[Mouse_ToPlot_ADn][non_nan]

non_nan = np.where(~np.isnan(relevance[Mouse_ToPlot_PoS]))[0]
x_data_PoS = relevance[Mouse_ToPlot_PoS][non_nan]
information_data_PoS = (information[Mouse_ToPlot_PoS][non_nan]-randomized_information[Mouse_ToPlot_PoS][non_nan])/(mean_firing_rate[Mouse_ToPlot_PoS][non_nan]*0.001)
selectivity_data_PoS = selectivity[Mouse_ToPlot_PoS][non_nan]
s_data_PoS = mean_vector_length[Mouse_ToPlot_PoS][non_nan]

x_data = np.append(x_data_ADn, x_data_PoS)
information_data = np.append(information_data_ADn, information_data_PoS)
selectivity_data = np.append(selectivity_data_ADn, selectivity_data_PoS)
s_data = np.append(s_data_ADn, s_data_PoS)

axScatterPlot = plt.subplot(gs[2,0])
axScatterPlot.scatter(x_data_ADn, information_data_ADn, marker="o", s=300*s_data_ADn, edgecolors="red", facecolors="None", linewidth='3', label="ADn neurons, "+str(Mouse_ToConsider))
axScatterPlot.scatter(x_data_PoS, information_data_PoS, marker="o", s=300*s_data_PoS, edgecolors="blue", facecolors="None", linewidth='3', label="PoS neurons, "+str(Mouse_ToConsider))
axScatterPlot.set_xlabel(r'multiscale relevance, $\mathcal{R}_t$ (Mats$^2$)', fontsize=16)
axScatterPlot.set_ylabel(r'head directional information, $I(s, \theta)$ (bits per spike)', fontsize=16)
axScatterPlot.legend(loc="upper left", scatterpoints=1, fontsize=18)
axScatterPlot.set_xlim(right=0.305)
axScatterPlot.patch.set_facecolor("white")
axScatterPlot.tick_params(labelsize=14)
r_pearson, p_pearson = pearsonr(x_data, information_data)
r_spearman, p_spearman = spearmanr(x_data, information_data)
axScatterPlot.annotate('$\\rho_p=$ %.3f, $P=$ %.2e \n $\\rho_s=$ %.3f, $P=$ %.2e'%(r_pearson, p_pearson, r_spearman, p_spearman),(0.24,1.37), fontsize=16)
axScatterPlot.text(-0.2, 1.05, "G", transform=axScatterPlot.transAxes, fontsize=20, fontweight='bold', va='top')

axScatterPlot = plt.subplot(gs[2,1])
axScatterPlot.scatter(x_data_ADn, selectivity_data_ADn, marker="o", s=300*s_data_ADn, edgecolors="red", facecolors="None", linewidth='3', label="ADn neurons, "+str(Mouse_ToConsider))
axScatterPlot.scatter(x_data_PoS, selectivity_data_PoS, marker="o", s=300*s_data_PoS, edgecolors="blue", facecolors="None", linewidth='3', label="PoS neurons, "+str(Mouse_ToConsider))
axScatterPlot.set_xlabel(r'multiscale relevance, $\mathcal{R}_t$ (Mats$^2$)', fontsize=16)
axScatterPlot.set_ylabel(r'head directional sparsity, $sp_{\theta}$', fontsize=16)
axScatterPlot.legend(loc="upper left", scatterpoints=1, fontsize=18)
axScatterPlot.set_xlim(right=0.305)
axScatterPlot.patch.set_facecolor("white")
axScatterPlot.tick_params(labelsize=14)
r_pearson, p_pearson = pearsonr(x_data, selectivity_data)
r_spearman, p_spearman = spearmanr(x_data, selectivity_data)
axScatterPlot.annotate('$\\rho_p=$ %.3f, $P=$ %.2e \n $\\rho_s=$ %.3f, $P=$ %.2e'%(r_pearson, p_pearson, r_spearman, p_spearman),(0.24,0.64), fontsize=16)

plt.savefig("Figures/Figure4.pdf", bbox_inches="tight", dpi=600)
