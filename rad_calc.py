#once the corner frequencies have been determined, we can determine stress drops 
#and radiated energy.
#there is a separate code to run regression.

#import stuff.
import glob
from scipy import odr
from scipy import stats
#import seaborn as sns
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
plt.style.use('ggplot')
from scipy.io import savemat
from collections import Counter
import functools
from multiprocessing import Pool
import random
from spec_rat import *
from corner_fit import *

class scaled_specrat(specrat):
	"""
	child of specrat 
	"""
	def __init__(self, corner=None, master_moment=None, egf_corner=None, egf_moment=None):
		self.corner=corner
		self.master_moment=master_moment
		self.egf_corner=egf_corner
		self.egf_moment=egf_moment
		super().__init__(self)
	#def integrate():

#function to correct for non-infinite integration limit as per Ide and Beroza (2001)
def corrfun(fm,fc,gamma):
	if gamma == 1:
		return (2./np.pi)*((-fm/fc)*(1/(1+(fm/fc))**2)+np.arctan(fm/fc))
	elif gamma == 2:
		return (1./(2.*np.pi))*(np.log((1-np.sqrt(2.)*(fm/fc)+(fm/fc)**2)/(1.+np.sqrt(2.)*(fm/fc)+(fm/fc)**2))+2.*(np.arctan(1.+np.sqrt(2)*(fm/fc))-np.arctan(1-np.sqrt(2)*(fm/fc))))

#function for the general shape of spectra ratio.
def specrat_fit(freqs,domega,fcs,fcl,gamma=1,n=2):
	return domega+np.multiply(1.0/gamma,np.log10(1+np.power(np.divide(freqs,fcs),n*gamma)))-np.multiply(1.0/gamma,np.log10(1+np.power(np.divide(freqs,fcl),n*gamma)))

#function to return weighted mean
def get_sdw(x,w):
    if len(x) == 0 or len(w) == 0:
	print('a zero file')
        return np.nan,np.nan
    cleaned_data=np.ma.masked_array(x, np.isnan(w))
    cleaned_weights=np.ma.masked_array(w, np.isnan(w))
    if sum(np.isnan(w))>0:
	print('we had NaN weights')
#    	print(cleaned_data)
#	print(cleaned_weights)
	mean=np.ma.average(cleaned_data, weights=cleaned_weights)
#	print('non nan mean is ' +str(mean))
    else:
    	mean=np.average(x,weights=w)
#	print('mean is ' + str(mean))
    N=float(len(w))-float(sum(np.isnan(w)))
    if N == 0:
	print('a zero file')
	return np.nan, np.nan
    if sum(np.isnan(w))>0:
	sdw=np.sqrt(np.divide(np.ma.sum(((cleaned_data-mean)**2)*w),np.divide((N-1.)*np.ma.sum(cleaned_weights),N)))
    else:
    	sdw=np.sqrt(np.divide(sum(np.multiply(w,(x-mean)**2)),np.divide((N-1.)*sum(w),N)))
    return mean, sdw

#could be cleaner.  idea is to scale spectral ratio by moment of smaller event
#to calculated radiated energy, also tried getting an "average attn fun"
def get_scaled_specrats(egfdict,mastname,mast_moment,corner,data,alldata,gamma=1,xcorr=0.4,snr=3.0,n=2,debug=False):
    egfnames=list(egfdict['egfs'])
    smoments=list(egfdict['small_moments'])
    scors=list(egfdict['corners'])
    print('mast_moment: ' + str(mast_moment))
    allstats=[]
    stats=[i for (i,j) in zip(data['stations'], data['egfnames']) if j in egfnames]
    swts=[i for (i,j) in zip(data['boot_stds'], data['egfnames']) if j in egfnames]
    egfs=list([egf for egf in data['egfnames'] if egf in egfnames])
    for stat in stats:
        for st in stat:
            allstats.append(st)
    allstats=list(set(allstats))
    scaled_specrats=[]
	#for saving
    unscaled_specrats_freqs=[]
    unscaled_specrats_specs=[]
    scalebys=[]
    for sr in alldata:
	if not sr.egf in egfnames:
		continue
	sr_scale=copy.deepcopy(sr)
	sr_scale.__class__=scaled_specrat
	if snr>3.0:
		sr_scale=sr_scale.holed(min_snr=snr)
		if sr_scale.datapercent <0.4:
			print('not enough data left after holing')
			continue
	smoment=10**smoments[egfnames.index(sr.egf)]
		#print(smoment)
	scor=scors[egfnames.index(sr.egf)]	
	idealspec_small=spec_fit(sr_scale.freqs,smoment,scor,gamma=gamma,n=n)
	scaled_spec=np.multiply(sr_scale.specratio, 10**idealspec_small)
	sr_scale.specratio = scaled_spec
	sr_scale.egf_corner=scor
	sr_scale.egf_moment=smoment
	sr_scale.master_moment=mast_moment
	sr_scale.corner=corner
	scaled_specrats.append(sr_scale)
	unscaled_specrats_freqs.append(sr.freqs)
	unscaled_specrats_specs.append(sr.specratio)
	scalebys.append(idealspec_small)
    mdict={'unscaled_freqs': unscaled_specrats_freqs, 'smallspecs': scalebys, 'unscaled_specs': unscaled_specrats_specs}
    savemat('/data/beroza/schu3/clusters/plots/unscaled_'+mastname+'_'+str(gamma)+'.mat', mdict)
    return scaled_specrats

def avg_of_specrats(specrats, weights=False, gamma=1, plot=True, plotname=None):
	freqlens=[len(spec.freqs) for spec in specrats]
	if len(freqlens) < 1:
		print('an empty set of spectral ratios')
		return
	maxfreqlen=max(freqlens)
	freqs=[spec.freqs for spec in specrats if len(spec.freqs) == maxfreqlen]
	freqs=freqs[0]
	avgarr=[]
	matfreqs=[]
	matspecs=[]
	for tfr in freqs:
		avgwt=[]
		avgspecval=[]
		for i, spec in enumerate(specrats):
			specrat=spec.specratio
			ifreq=spec.freqs
			if tfr in ifreq:
				avgspecval.append(specrat[list(ifreq).index(tfr)])	
				if weights:
					avgwt.append(weights[i])
		if weights:
			avgspecval=np.average(avgspecval, weights=avgwt)
		else:
			avgspecval=np.nanmean(avgspecval)	
		avgarr.append(avgspecval)
	if plot:
		fig,ax=plt.subplots(nrows=1,ncols=1)
		ax.set_title('scaled spectral ratio')
		for i, spec in enumerate(specrats):
			egfmag=float(spec.egf.split('_')[-1])
			if egfmag < 2.3:
				colour='cyan'
			elif egfmag < 2.6:
				colour='cyan'
			elif egfmag < 3.0:
				colour='cyan'
			else:
				colour='cyan'
			ax.loglog(spec.freqs, spec.specratio,color=colour,basex=10,basey=10)
			matfreqs.append(spec.freqs)
			matspecs.append(spec.specratio)
			idealarr=spec_fit(freqs,spec.master_moment, spec.corner,gamma=gamma)
		ax.loglog(freqs, 10**idealarr, 'r--', basex=10, basey=10)
		ax.loglog(freqs,avgarr,'m*')
		ax.set_xlim((10**0, 10**1))
		if plotname:
			plt.savefig(plotname)
			mdict={'freqs': matfreqs, 'specs': matspecs, 'ideal': 10**idealarr, 'avg': avgarr}
			savemat(plotname.split('.png')[0]+'.mat', mdict)
		else:
			plt.show()
	return freqs, avgarr

def get_breakpoints(freqs, arr, corner, window=3, threshold=2):
	from scipy.signal import savgol_filter
#obtain points where slope changes to cut off integration
	freqs=np.asarray(freqs)
	arr=np.asarray(arr)
	sdiff=savgol_filter(np.log10(arr), window_length=window, polyorder=2, deriv=2)
	maxdiff=np.max(np.abs(sdiff))
	large=np.where(np.abs(sdiff) > maxdiff/threshold)[0]	
	gaps=np.diff(large) > window
	begins=np.insert(large[1:][gaps],0,large[0])
	changes=(begins).astype(np.int)
	if len(changes) < 1:
		return []
	changes=[change for change in changes if freqs[change] > corner]
	if len(changes) < 1:
		return []
	changes=np.asarray(changes)
	#print(changes)
	breakpoints=freqs[changes]
	return breakpoints

def get_rad_energy(i, q, vs, rho, mu, specrats,gamma,plotname=None):
#wraps integration of the attnfuns for parallelizability
    radiated_energies=[]
    original_len=len(specrats)
    #selection of subset for bootstrapping
    if len(specrats) == 1:
	specrats=specrats
	itrain=[0]
    else:
	    itrain=np.random.choice(np.arange(len(specrats)), np.random.randint(1,len(specrats)))
	    specrats=[specrats[it] for it in itrain]
    if float(len(itrain))/float(original_len)>= 0.8:
	plot=True
	fig,ax=plt.subplots(nrows=1,ncols=1)
	for specrat in specrats:
		ax.loglog(specrat.freqs,specrat.specratio,basex=10, basey=10, color='c')
    else:
	plot=False
    avg_freqs,avg_spec=avg_of_specrats(specrats,gamma=gamma,plot=True, plotname='/data/beroza/schu3/clusters/plots/avgarray_'+specrats[0].master+'_'+str(gamma)+'.png')
    if plot == True:
	ax.loglog(avg_freqs,avg_spec, basex=10, basey=10, color='m')
    slopechanges=get_breakpoints(avg_freqs,avg_spec,specrats[0].corner)
 #   maxfreq=min(specrats[0].corner+3, specrats[0].egf_corner)
    if len(slopechanges) < 1:
	maxfreq=10.0
    else:
    	maxfreq=min(slopechanges[0], 10.0)
    print('we are integrating up to ' + str(maxfreq) + ' Hz')
    if plot == True:
	ax.axvline(x=maxfreq, color='r')
    avg_spec_cut=[j for (i,j) in zip (avg_freqs, avg_spec) if i < maxfreq]
    avg_freqs_cut=[i for i in avg_freqs if i < maxfreq]
    cut_datapercent=float(len(avg_freqs_cut))/float(len(avg_freqs))
    print('cut datapercent: ' + str(cut_datapercent))
#    if len(avg_spec) < 1:
    if cut_datapercent < 0.3:
	print('there is not enough original data left to integrate.')
	return np.nan 
    avg_freqs=avg_freqs_cut
    avg_spec=avg_spec_cut
    integrand=np.multiply(avg_spec, avg_freqs)
    integral=np.trapz(y=np.multiply(integrand,integrand), x=avg_freqs)
    R=corrfun(avg_freqs[-1], specrats[0].corner, gamma)
    factor=(1.+(1./q))*((8.*np.pi)/(10.*rho*(vs**5)))
    integral=factor*integral
    if np.isinf(integral) or np.isnan(integral):
	print('we had a nan value')
	if plot == True:
		ax.title('E=NaN')
		plt.savefig(plotname+specrats[0].master+'.png')
	return np.nan
    else:
	if plot == True:
		ax.set_title('E='+str(integral)+'J')
		plt.savefig(plotname+specrats[0].master+'.png')
	return integral 
