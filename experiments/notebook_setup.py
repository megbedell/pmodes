get_ipython().magic("matplotlib inline")
get_ipython().magic('config InlineBackend.figure_format = "retina"')

import numpy as np
import astropy.units as u
import matplotlib.pyplot as plt

plt.style.use("default")
plt.rcParams["savefig.dpi"] = 100
plt.rcParams["figure.dpi"] = 100
plt.rcParams["font.size"] = 16
plt.rcParams["font.family"] = "sans-serif"
#plt.rcParams["font.sans-serif"] = ["Liberation Sans"]
plt.rcParams["mathtext.fontset"] = "custom"
plt.rcParams['xtick.major.size'] = 8
plt.rcParams['xtick.major.width'] = 1
plt.rcParams['xtick.minor.size'] = 4
plt.rcParams['xtick.minor.width'] = 0.5
plt.rcParams['ytick.major.size'] = 8
plt.rcParams['ytick.major.width'] = 1

plot_dir = '../paper/figures/'
