import glob
import numpy as np
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
from mpl_toolkits.basemap import Basemap
#from spectrum import *
from scipy.fftpack import fft, rfft, fftfreq
from obspy.signal.util import smooth
from obspy.geodetics.base import gps2dist_azimuth

class specrat(object):
	def __init__(self,freqs,specratio,snr,master,mla,mlo,egf,ela,elo,station,stla,stlo,channel):
		self.master=master
		self.egf=egf
		self.station=station
		self.stla=stla
		self.stlo=stlo
		self.mla=mla
		self.mlo=mlo
		self.ela=ela
		self.elo=elo
		self.channel=channel
		self.specratio=specratio
		self.freqs=freqs
		self.snr=snr

def logbin(f,a,nbins,flims):
	"""
	smooth 
	"""
	fout=[]
	aout=[]
	logbins=np.linspace(flims[0], flims[1], nbins+1)
	dlogbin=logbins[1]-logbins[0]
	logf=np.log10(f)
	for i in range(nbins):
		binamps=a[(logf>=logbins[i])*(logf<=logbins[i+1])]
		if len(binamps)>0:
			aout.append(np.mean(binamps))
			fout.append(10**(logbins[i]+0.5*dlogbin))
	return fout, aout

def spectrum_gen(timeseries, picksfile, station, args, mt_tb=4, debug=False):
	"""
	returns the spectrum of a timeseries in msfile format.
	"""
	from mtspec import mtspec
	from scipy.signal import welch
#	print('starting the call to spectrum')
	prefilter=args.F
	fftype=args.ft
	winlength=args.W
	shift=args.S
	stype=args.P
	nwins=args.N
	nlogbins=args.lb
	timeseries=timeseries.detrend()
	timeseries=timeseries.detrend('demean')
	cat = read_events(picksfile)
	picks=cat[0].picks
	flims=[-0.5, 1.2]
	if prefilter:
		flims=np.log10(prefilter)
	if stype == 'S' or stype == 'coda' or stype == 'lateS':
		pick = [ipick for ipick in picks if ipick.phase_hint == 'S' and ipick.waveform_id.station_code[-4:]==station[-4:]]
		if len(pick) < 1:
			print('no pick found in the pickfile for station ' + station + ' in ' + picksfile)
			return np.asarray([]), np.asarray([])
		pick=pick[0].time
		if stype=='coda':
			ttime = pick-timeseries.stats.starttime
			pick = timeseries.stats.starttime+1.5*ttime
		if stype=='lateS':
			pick=pick+2
	elif stype == 'P' or stype == 'noise':
		pick = [ipick for ipick in picks if ipick.phase_hint == 'P' and ipick.waveform_id.station_code[-4:]==station[-4:]]
		if len(pick) < 1:
			print('no pick found in the pickfile for station ' + station)
			#return None, None
			return np.asarray([]), np.asarray([])
		pick=pick[0].time
		if stype == 'noise':
			pick=pick-5
			if pick-timeseries.stats.starttime < 0:
				pick=timeseries.stats.starttime
	else:
		print('please input valid wave type')
		return np.asarray([]), np.asarray([])
	if fftype == 'welch':
		smoothfactor=5
		welch_winlength=nwins*(winlength-shift)
		clipwin=timeseries.copy()
		clipstart=pick
		clipend=clipstart+welch_winlength
		clipwin.trim(clipstart,clipend)
		clip2=clipwin.copy()
		clip2.trim(clipstart,clipstart+winlength)
		lenwin=len(clip2.data)
		if lenwin<=256:
			nopts=256
		elif lenwin<=512:
			nopts=512
		elif lenwin<1024:
			nopts=1024
		else:
			nopts=2048
		ffreq,specs=welch(clipwin.data,clipwin.stats.sampling_rate,nperseg=int(clipwin.stats.sampling_rate*winlength),noverlap=int(clipwin.stats.sampling_rate*(winlength-shift)),nfft=nopts)
		specs=np.sqrt(specs)
	else:
		specs=[]
		ffreq=[]
		lenwin=0
		for i in range(1,nwins+1):
			clipwin=timeseries.copy()
			clipstart=pick+(i-1)*shift
			clipend=clipstart+winlength
			if clipwin.stats.endtime-clipend < 0:
				print('at the end of the data')
				continue
			clipwin.trim(clipstart,clipend)
			if fftype=='simple':
				spec=rfft(clipwin.data)
				ffreq=fftfreq(n=len(clipwin.data), d=clipwin.stats.delta)
				specs.append(spec)
			elif fftype=='multitaper2.5' or fftype=='multitaper4' or fftype=='multitaper':
				if fftype=='multitaper2.5':
					mt_tb=2.5
				smoothfactor=3
				lenwin=len(clipwin.data)
				if lenwin<=256:
					nopts=256
				elif lenwin<=512:
					nopts=512
				elif lenwin<1024:
					nopts=1024
				else:
					nopts=2048
				spec,freq,jackknife,_,_=mtspec(data=clipwin.data, delta=clipwin.stats.delta, time_bandwidth=mt_tb,nfft=nopts,statistics=True)
				specs.append(np.sqrt(spec))
				ffreq=freq
		print('len specs is ' + str(len(specs)))
		specs=np.average(specs,axis=0)
	print('len ffreq is ' + str(len(ffreq)))
	#smooth the individual spectra before taking spectral ratio.
	if len(ffreq)==0:
		return np.asarray([]), np.asarray([])
	ffreq,specs=logbin(ffreq,specs,nbins=nlogbins,flims=flims)
	if args.sm:
		specs=smooth(specs,smoothfactor)	
	if debug:
		fig,ax=plt.subplots(nrows=2,ncols=1)
		ax[0].plot(timeseries.data,color='k')
		ax[0].axvline((pick-timeseries.stats.starttime)*timeseries.stats.sampling_rate,color='r')
		ax[1].loglog(ffreq,specs,basex=10,basey=10)
		plt.savefig('qc_plot_'+timeseries.stats.starttime.strftime('%Y%m%dT%H%M%S')+'_'+stype+'_'+station+'.png')
	return ffreq, specs	

def specrat_fit(freqs,domega,fcs,fcl,gamma=1,n=2):
#	print('len frewqs: ' + str(len(freqs)))
#	print('domega:')
#	print(domega)
#	print('fcs:')
#	print(fcs)
#	print('fcl:')
#	print(fcl)
#	print('gamma:')
#	print(gamma)
#	print('n:')
#	print(n)
	return domega+np.multiply(1.0/gamma,np.log10(1+np.power(np.divide(freqs,fcs),n*gamma)))-np.multiply(1.0/gamma,np.log10(1+np.power(np.divide(freqs,fcl),n*gamma)))

def spec_global_errs(fc,freqsins,ratios,guess0s,gamma,n,loss='linear',debug=False,debugname=None,returnfit=False):
	globalerr=0
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
			rnorm=res['cost']
			if debug:
				ax.semilogx(freqsin,np.log10(ratio),basex=10,color='c')
				bestfit=specrat_fit(freqsin,popt[0],popt[1],fc,gamma,n)
				ax.semilogx(freqsin,bestfit,basex=10,color='m')	
			if returnfit:
				fit_omega.append(popt[0])
				fit_fcs.append(popt[1])		
	#		guess=specrat_fit(freqsin,popt[0],popt[1], fc,gamma,n)
	#		rnorm=np.linalg.norm(np.log10(ratio)-guess)
			globalerr=globalerr+rnorm
		except:
			logging.exception('values at exception: ')
			rnorm=np.linalg.norm(np.log10(ratio)-specrat_fit(freqsin,2000,10,fc,gamma,n))
			globalerr=globalerr+rnorm/np.sqrt(len(ratio))
	if debug:
		plt.savefig(debugname)
	if returnfit:
		return globalerr/len(ratios), fit_omega, fit_fcs
	return globalerr/len(ratios)	

def corner(i,ratsin,freqsin,gridsearch):
	misfits=[]
	irats=np.random.choice(len(ratsin), np.random.randint(1,len(ratsin)))
	randrats=[ratsin[ir] for ir in irats]
	for fc_guess in gridsearch:
		try:
			res=brune_global_errs(fc_guess,freqsin,randrats)
		except:
			logging.exception('values at exception: ')
			res=np.inf
		misfits.append(res)
	sbest=gridsearch[np.argmin(misfits)]
	return sbest 

def get_spec_ratio(data_small, ms_large, pick_large, station, args, debug=False):
	"""
	generate spectral ratios
	"""
	ms_small=data_small[0]
	ela=float(ms_small.split('_evla')[-1].split('_')[0])
	elo=float(ms_small.split('_lo')[1].split('_')[0])
	pick_small=data_small[1]
	stla=float(ms_small.split('stla')[-1].split('_')[0])
	stlo=float(ms_small.split('_lo')[-1].split('.ms')[0])
	egf_name=ms_small.split('evt_')[-1].split('_evla')[0]
	master_name=ms_large.split('evt_')[-1].split('_evla')[0]
	mla=float(ms_large.split('_evla')[-1].split('_')[0])
	mlo=float(ms_large.split('_lo')[1].split('_')[0])
	edata=read(ms_small)
	mdata=read(ms_large)
	echannels=[tr.stats.channel for tr in edata]
	mchannels=[tr.stats.channel for tr in mdata]
	if not args.C in echannels and len(list(set(echannels)&set(args.altC))) == 0:
		empty_specrat=specrat(np.asarray([]),np.asarray([]), np.asarray([]),master_name,mla,mlo,egf_name,ela,elo,station,stla,stlo,args.C)
		print('we didnt find this channel')
		return empty_specrat
	if not args.C in mchannels and len(list(set(mchannels)&set(args.altC))) == 0:
		empty_specrat=specrat(np.asarray([]),np.asarray([]), np.asarray([]),master_name,mla,mlo,egf_name,ela,elo,station,stla,stlo,args.C)
		print('we didnt find this channel')
		return empty_specrat
	allchans=args.altC
	allchans.append(args.C)
	echannel=list(set(echannels)&set(allchans))[0]
	mchannel=list(set(mchannels)&set(allchans))[0]
	edata=edata.select(channel=echannel)[0]
	mdata=mdata.select(channel=mchannel)[0]
	freq1,spec1=spectrum_gen(mdata,pick_large,station,args,debug=debug)	
	freq2,spec2=spectrum_gen(edata,pick_small,station,args,debug=debug) 
	argsn=copy.deepcopy(args)
	argsn.P='noise'
	freqn,specn=spectrum_gen(edata,pick_small,station,argsn,debug=debug)
#	if debug:
#		fig,ax=plt.subplots(nrows=3,ncols=2)
#		ax[0].plot(mplottr.data)
#		ax[1].plot(eplottr.data)
	if len(freq1) < 1 or len(spec1)<1 or len(freq2)<1 or len(spec2)<1:
		empty_specrat=specrat(np.asarray([]),np.asarray([]), np.asarray([]),master_name,mla,mlo,egf_name,ela,elo,station,stla,stlo,args.C)
		print('we were not able to generate spectrum')
		return empty_specrat
	if not len(spec1)==len(spec2) or not len(spec2)==len(specn):
		empty_specrat=specrat(np.asarray([]),np.asarray([]), np.asarray([]),master_name,mla,mlo,egf_name,ela,elo,station,stla,stlo,args.C)
		print('length of spectra are not equal! check the picks')
		return empty_specrat
	try:
		snr=np.divide(spec2,specn)
		specratio=np.divide(spec1,spec2)
		print('we divided the spectral ratio')
	except:
		print('freq1:')
		print(freq1)
		print('freq2:')
		print(freq2)
		print('freqn')
		print(freqn)
		empty_specrat=specrat(np.asarray([]),np.asarray([]), np.asarray([]),master_name,mla,mlo,egf_name,ela,elo,station,stla,stlo,args.C)
		return empty_specrat
	specratio=specrat(freq1,specratio,snr,master_name,mla,mlo,egf_name,ela,elo,station,stla,stlo,args.C)
	return specratio

#deprecated function
def check_for_picks(egfdata, args):
	egf_file=egfdata[0]
	egf_qc=read(egf_file)
	egf_chans=[tr.stats.channel for tr in egf_qc]
	if not args.C in egf_chans and len(list(set(egf_chans)&set(args.altC))) == 0:
		egf_files=None
		egf_picks=None
		return egf_files, egf_picks
	station=egf_file.split('_stla')[0].split('_')[-1]
	egf_pick=egfdata[1]
	catalog=read_events(egf_pick)[0].picks
	stations = [pick.waveform_id.station_code for pick in catalog]
	picks_at_stat = [pick for pick in catalog if station in pick.waveform_id.station_code]
	picks_at_stat=[pick for pick in picks_at_stat if pick.waveform_id.channel_code==args.C or pick.waveform_id.channel_code in args.altC]
#	print('picks:')
#	print(picks_at_stat)
	if len(picks_at_stat)>0:
		egf_files=egf_file
		egf_picks=egf_pick	
	#function to check that each ms_file trace has a corresponding pick
	else:
		egf_files=None
		egf_picks=None
	return egf_files, egf_picks

