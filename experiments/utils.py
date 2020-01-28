import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import celerite as celery
import exoplanet as xo
import pymc3 as pm
import theano.tensor as tt
from exoplanet.gp import terms, GP
from scipy.interpolate import interp1d


def simulate_exposure(ts, rvs, start_time, exp_time):
    pad = 100. # seconds - ARBITRARY
    smaller_inds = (ts > (start_time - pad)) & (ts < (start_time + exp_time + pad))    
    interp = interp1d(ts[smaller_inds], rvs[smaller_inds], kind='cubic')
    tiny = 0.1 # 100 ms
    fine_ts = np.arange(start_time, start_time+exp_time, tiny) # fine grid
    fine_rvs = interp(fine_ts)
    return np.sum(fine_rvs)/len(fine_rvs) # ASSUMES EVEN WEIGHTING - technically incorrect for last point

### FOR NOTEBOOK 02:
xlim_data = np.array([10.2, 10.8]) * 86400 # for selecting all data
xlim_plot = [890000, 895000] # for zoomed-in plots

def plot_validation_test(t, y, yerr, y_pred, t_all, y_all, yerr_all, y_pred_all, 
                         t_grid, mu, sd, xlim_plot=xlim_plot):
    fig, (ax1,ax2) = plt.subplots(2, 1, figsize=(14,6), sharex=True, 
                              gridspec_kw={'height_ratios':[3,1], 'hspace':0.1})
                              
    art = ax1.fill_between(t_grid, mu + sd, mu - sd, color="C1", alpha=0.3)
    art.set_edgecolor("none")
    ax1.plot(t_grid, mu, color="C1", label="prediction")

    ax1.errorbar(t_all, y_all, yerr=yerr_all, fmt=".k", capsize=0, label='validation data')
    ax1.errorbar(t, y, yerr=yerr, fmt=".r", capsize=0, label="training data")
    ax1.legend(fontsize=12)

    ax2.errorbar(t_all, y_all - y_pred_all, yerr=yerr_all, fmt=".k", capsize=0, label="resids", alpha=0.3)
    ax2.errorbar(t, y - y_pred, yerr=yerr, fmt=".r", capsize=0, label="resids", alpha=0.3)

    inds = (t_all > xlim_plot[0]) & (t_all < xlim_plot[1])
    chisq = np.sum(((y_all - y_pred_all)/yerr_all)[inds])**2
    ax2.text(0.02, 0.7, r'$\chi^2$ = {0:.2f}'.format(chisq), fontsize=12, 
             transform=ax2.transAxes, bbox=dict(facecolor='white', alpha=0.5))
    ax2.set_xlabel('Time (s)', fontsize=14)
    ax1.set_ylabel(r'RV (m s$^{-1}$)', fontsize=14)
    ax2.set_ylabel('Resids', fontsize=12)

    ax1.set_xlim(xlim_plot)
    
    return fig

def plot_validation_test_full(t, y, yerr, y_pred, t_all, y_all, yerr_all, y_pred_all, 
                         t_grid, mu, sd, xlim_plot=xlim_plot):
    fig, (ax1,ax2) = plt.subplots(2, 1, figsize=(14,6), sharex=True, 
                              gridspec_kw={'height_ratios':[3,1], 'hspace':0.1})
                              
    art = ax1.fill_between(t_grid, mu + sd, mu - sd, color="C1", alpha=0.3)
    art.set_edgecolor("none")
    ax1.plot(t_grid, mu, color="C1", label="prediction")

    ax1.errorbar(t_all, y_all, yerr=yerr_all, fmt=".k", capsize=0, label='validation data')
    ax1.errorbar(t, y, yerr=yerr, fmt=".r", capsize=0, label="training data")
    ax1.legend(fontsize=12)

    ax2.errorbar(t_all, y_all - y_pred_all, yerr=yerr_all, fmt=".k", capsize=0, label="resids", alpha=0.3)
    ax2.errorbar(t, y - y_pred, yerr=yerr, fmt=".r", capsize=0, label="resids", alpha=0.3)

    chisq = np.sum(((y_all - y_pred_all)/yerr_all))**2
    ax2.text(0.02, 0.7, r'$\chi^2$ = {0:.2f}'.format(chisq), fontsize=12, 
             transform=ax2.transAxes, bbox=dict(facecolor='white', alpha=0.5))
    ax2.set_xlabel('Time (s)', fontsize=14)
    ax1.set_ylabel(r'RV (m s$^{-1}$)', fontsize=14)
    ax2.set_ylabel('Resids', fontsize=12)
    
    return fig

### FOR NOTEBOOK 03:

def plot_nights(t, y, yerr, y_pred, start_ts, t_grid, mu, sd):
    fig, (ax1,ax2) = plt.subplots(2, 3, figsize=(20,6), sharey=True, 
                              gridspec_kw={'height_ratios':[3,1], 'hspace':0.05, 'wspace':0.1})

    for ax in ax1: # data + fit
        art = ax.fill_between(t_grid, mu + sd, mu - sd, color="C1", alpha=0.3)
        art.set_edgecolor("none")
        ax.plot(t_grid, mu, color="C1", label="prediction")

        ax.errorbar(t, y, yerr=yerr, fmt=".k", capsize=0, label="data")

    for ax in ax2: # residuals
        ax.axhline(0., color='C1', ls='--', alpha=0.5)
        ax.errorbar(t, y - y_pred, yerr=yerr, fmt=".k", capsize=0, label="resids", alpha=0.6)


    ax2[1].set_xlabel('Time (s)', fontsize=14)
    ax1[0].set_ylabel(r'RV (m s$^{-1}$)', fontsize=14)
    ax2[0].set_ylabel('Resids', fontsize=12)

    for i,ax in enumerate(ax1):
        ax.set_xlim([start_ts[i] - 60, start_ts[i] + 900])
        ax.text(0.05, 0.9, 't = {0:.1f} days'.format(start_ts[i]/24./3600.), 
                fontsize=12, transform=ax.transAxes)
        ax.xaxis.set_ticklabels([])
    for i,ax in enumerate(ax2):
        ax.set_xlim([start_ts[i] - 60, start_ts[i] + 900])
        
    return fig

def plot_year(t, y, yerr, y_pred, start_ts, t_grid, mu, sd):
    fig, (ax1,ax2) = plt.subplots(2, 1, figsize=(16,6), sharex=True, 
                              gridspec_kw={'height_ratios':[3,1], 'hspace':0.05})

    art = ax1.fill_between(t_grid, mu + sd, mu - sd, color="C1", alpha=0.3)
    art.set_edgecolor("none")
    ax1.plot(t_grid, mu, color="C1", label="prediction")

    ax1.errorbar(t, y, yerr=yerr, fmt=".k", capsize=0, label="data")

    ax2.axhline(0., color='C1', ls='--', alpha=0.5)
    ax2.errorbar(t, y - y_pred, yerr=yerr, fmt=".k", capsize=0, label="resids", alpha=0.3)
    
    ax1.text(0.05, 0.9, 'RMS = {0:.2f} m/s'.format(np.sqrt(np.sum((y - y_pred)**2/len(y)))), 
             fontsize=12, transform=ax1.transAxes, bbox=dict(facecolor='white', alpha=0.5))
    ax2.set_xlabel('Time (s)', fontsize=14)
    ax1.set_ylabel(r'RV (m s$^{-1}$)', fontsize=14)
    ax2.set_ylabel('Resids', fontsize=12)
        
    return fig