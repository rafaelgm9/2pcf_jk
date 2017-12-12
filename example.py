import numpy as np
import cross_correlation
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})


# Read data
d1 = np.loadtxt('halos')
d2 = np.loadtxt('particles')

# Set parameters
L = 1050.               # Size of simulation box
l = 262.5               # Size of jackknife region. In this example there is 4*4*4 jk regions.
minsep = 0.1            # Minimum separation of pairs
maxsep = 80.            # Maximum
nbins = 50              # Number of radial bins. The code will set up logaritmic bins
nthreads = 8            # Number of cpu threads

# Compute xi
# meanxi_i is the mean of xi_i
# cov is covariance matrix
# xi_i is the set of CFs measured in each JK sample. jk_estimates = True returns this array.
# xi is the true CF
meanxi_i, cov, xi_i, xi = cross_correlation.cross_tpcf_jk(d1 = d1, d2 = d2, boxsize = L, gridsize = l,
                                                      minsep = minsep, maxsep = maxsep, nbins = nbins,
                                                      nthreads = nthreads, jk_estimates = True)

# Let's plot!
r = 10 ** np.linspace(np.log10(minsep), np.log10(maxsep), nbins+1)
rm = (r[1:] + r[:-1]) / 2.
plt.rc("text", usetex=True)
plt.xlabel(r'$r\ [h^{-1} Mpc]$', fontsize=24)
plt.ylabel(r'$\xi_{hm}$', fontsize=24)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.xscale('log')
plt.yscale('log')
plt.errorbar(rm, meanxi_i, yerr=np.sqrt(cov.diagonal()), color='blue', linewidth=1, capsize=2, label='HM correlation', fmt='')
plt.show()


