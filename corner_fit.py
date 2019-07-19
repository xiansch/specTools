import glob
import numpy as np
from obspy.signal.filter import lowpass
from obspy import read, Stream, Trace, UTCDateTime, read_events
import matplotlib as mpl
mpl.use('Agg')
from matplotlib import pylab as plt
from functools import partial
plt.style.use("ggplot")
import sys, traceback, logging
import multiprocessing
import argparse as ap
import scipy.optimize as opt
#import spectrum
import copy
from obspy.geodetics.base import gps2dist_azimuth
#from mpl_toolkits.basemap import Basemap
#from spectrum import *
from scipy.fftpack import fft, rfft, fftfreq
from obspy.signal.util import smooth
from obspy.geodetics.base import gps2dist_azimuth
from spec_rat import *

	
def specrat_fit(freqs,domega,fcs,fcl,gamma=1,n=2):
	"""
	functional form of the spectral ratio
	"""
	return domega+np.multiply(1.0/gamma,np.log10(1+np.power(np.divide(freqs,fcs),n*gamma)))-np.multiply(1.0/gamma,np.log10(1+np.power(np.divide(freqs,fcl),n*gamma)))

def spec_global_errs(fc,freqsins,ratios,guess0s,gamma,n,loss='linear',debug=False,debugname=None,returnfit=False):
	globalerr=0
	avg_rsquared=0
	if debug:
		fig,ax=plt.subplots(nrows=1,ncols=1)
	if returnfit:
		fit_omega=[]
		fit_fcs=[]
	no_rats = len(ratios)
	for ir,ratio in enumerate(ratios):
		freqsin=freqsins[ir]
		try:
			res=opt.least_squares(lambda x: specrat_fit(freqsin,x[0],x[1],fc,gamma,n)-np.log10(ratio),[guess0s[ir],20], loss=loss, bounds=([1,1],[1e5,np.inf]), max_nfev=1000)
			#popt,pcov=opt.curve_fit(lambda freqs, domega, fcs: specrat_fit(freqs, domega, fcs, fc,gamma,n), freqsin,np.log10(ratio))
			popt=res['x']
	#		rnorm=res['cost']
			if debug:
				ax.semilogx(freqsin,np.log10(ratio),basex=10,color='c')
				bestfit=specrat_fit(freqsin,popt[0],popt[1],fc,gamma,n)
				ax.semilogx(freqsin,bestfit,basex=10,color='m')	
			if returnfit:
				fit_omega.append(popt[0])
				fit_fcs.append(popt[1])		
			guess=specrat_fit(freqsin,popt[0],popt[1], fc,gamma,n)
			rnorm=np.linalg.norm(np.log10(ratio)-guess)
			ss_res=rnorm**2
			ss_tot=np.sum((np.log10(ratio)-np.mean(np.log10(ratio)))**2)
			rsquare=1-(ss_res/ss_tot)
			avg_rsquared=avg_rsquared+rsquare
			globalerr=globalerr+rnorm/np.sqrt(len(ratio))
		except:
			no_rats=no_rats-1
			logging.exception('values at exception: ')
	#		rnorm=np.linalg.norm(np.log10(ratio)-specrat_fit(freqsin,2000,10,fc,gamma,n))
	#		globalerr=globalerr+rnorm/np.sqrt(len(ratio))
	if debug:
		plt.savefig(debugname)
	if no_rats > 0:
		globalerr=globalerr/no_rats
		avg_rsquared=avg_rsquared/no_rats
	if len(ratios)-no_rats > 15:
		print(len(ratios))
		print(no_rats)
		print(len(ratios)-no_rats)
		print('too many egfs removed')
#		globalerr=np.inf
	if returnfit:
		return globalerr, fit_omega, fit_fcs, avg_rsquared
	return globalerr/len(ratios)	

def corner(i,ratsin,freqsins,guess0s,gridsearch,gamma,n,loss,debug):
	misfits=[]
	irats=np.random.choice(len(ratsin), np.random.randint(1,len(ratsin)))
	randrats=[ratsin[ir] for ir in irats]
	freqsin=[freqsins[ir] for ir in irats]
	randguess=[guess0s[ir] for ir in irats]
	for fc_guess in gridsearch:
		try:
			res=spec_global_errs(fc_guess,freqsin,randrats,randguess,gamma=gamma,n=n,loss=loss,debug=debug)
		except:
			logging.exception('values at exception: ')
			res=np.inf
		misfits.append(res)
	sbest=gridsearch[np.argmin(misfits)]
	return sbest 


def spec_fit(freqs,moment,fcl,gamma=1,n=2):
	return np.log10(moment)-np.multiply(1.0/gamma,np.log10(1+np.power(np.divide(freqs,fcl),n*gamma)))

def spec_fit_mrate(freqs,moment,fcl,gamma=1,n=2):
	return (2.*np.pi*freqs*moment)/(((1+np.divide(freqs,fcl)**(n*gamma)))**(1./gamma))
#	return np.log10(moment)-np.multiply(1.0/gamma,np.log10(1+np.power(np.divide(freqs,fcl),n*gamma)))
