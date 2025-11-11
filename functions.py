import numpy as np
import scipy.integrate as spi
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.integrate import solve_ivp
from math import sqrt, pi
import matplotlib as mpl
from ipywidgets import interact, FloatSlider
from ipywidgets import Layout

mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
mpl.rc('font',**{'family':'sans-serif','sans-serif':['Arial']})
mpl.rcParams['font.size'] = 16

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module='matplotlib.font_manager')


colours = {
    "green": "#00B828",
    "yellow": "#FFD900",
    "purple": "#800FF2",
    "blue": "#0073FF",
    "orange": "#FF5000",
    "grey": "#B3B3B3",
}
plt.rcParams.update({
    'xtick.major.width': 1.5,     # x-tick thickness
    'ytick.major.width': 1.5,     # y-tick thickness
    'xtick.major.size': 5,        # x-tick length
    'ytick.major.size': 5,        # y-tick length
    'axes.linewidth': 1,         # Thickness of axis border (applies to spines)
    'lines.linewidth': 2
})

def integrate_approx(y_full, x_full, n):
    """
    Integrate using n trapezoids by downsampling the data.
    
    Parameters:
    -----------
    x_full : array
        Full x data
    y_full : array
        Full y data
    n : int
        Number of trapezoids to use
    
    Returns:
    --------
    area : float
        Computed area under curve
    x_subset : array
        Downsampled x values (n+1 points)
    y_subset : array
        Downsampled y values (n+1 points)
    """
    # Create n+1 equally spaced indices (n trapezoids need n+1 points)
    indices = np.linspace(0, len(x_full)-1, n+1, dtype=int)
    x_subset = x_full[indices]
    y_subset = y_full[indices]
    
    area = spi.trapezoid(y_subset, x_subset)
    
    return area, x_subset, y_subset

def gauss(x, A, mu, sigma):
    return A * np.exp(-0.5 * ((x - mu) / sigma) ** 2)

def two_gauss(x, A1, mu1, s1, A2, mu2, s2):
    return gauss(x, A1, mu1, s1) + gauss(x, A2, mu2, s2)



def first_order(t, y, k):
    title_text = f"d$[\\mathrm{{R}}]$/d$t$ = $-k [\\mathrm{{R}}]$"
    return -k * y, title_text

def second_order(t, y, k):
    title_text = f"d$[\\mathrm{{R}}]$/d$t$ = $-2 k [\\mathrm{{R}}]^2$"
    return -2 * k * y**2, title_text

def third_order(t, y, k):
    title_text = f"d$[\\mathrm{{R}}]$/d$t$ = $-3 k [\\mathrm{{R}}]^3$"
    return -3 * k * y**3, title_text
