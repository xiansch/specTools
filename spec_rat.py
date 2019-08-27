import glob
import sys, traceback, logging
import numpy as np
from obspy.signal.filter import lowpass
from obspy import read, Stream, Trace, UTCDateTime, read_events
import matplotlib as mpl
mpl.use('Agg')
from matplotlib import pylab as plt
from functools import partial
#plt.style.use("ggplot")
import sys, traceback, logging
import multiprocessing
import argparse as ap
import scipy.optimize as opt
#import spectrum
import copy
from obspy.geodetics.base import gps2dist_azimuth
#from mpl_toolkits.basemap import Basemap
#from spectrum import *
from numpy import max, pi
from obspy.signal.util import next_pow_2
from scipy.signal import correlate
from scipy.fftpack import fft, ifft, rfft, fftfreq
from obspy.signal.util import smooth
from obspy.geodetics.base import gps2dist_azimuth
from obspy.signal.cross_correlation import xcorr_pick_correction

class specrat(object):
	"""
	spectral ratio object
	type master, egf: string
	param master, egf: name of the master event and eGf event in datetime
	type station, channel: string
	param station, channel: name of station and channel
	type stla, stlo, mla, mlo, ela, elo: float
	param stla, stlo, mla, mlo, ela, elo: lat-lon of station, master event, eGf event
	type specratio, freqs, snr: ndarray
	param specratio, freqs, snr: spectral ratio amplitude, frequency, signal to noise ratio
	"""
	def __init__(self,freqs,specratio,snr,master,mla,mlo,egf,ela,elo,station,stla,stlo,channel,mdp=None,edp=None,stf=None,mastertr=None,egftr=None,mspec=None,espec=None,datapercent=100.0,snrtype='pre-P',xcorr=None,focal_mech=None,snrPcoda=None, msnr=None, esnr=None, msnrPcoda=None, esnrPcoda=None):
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
		self.snrPcoda=snrPcoda
		self.msnr=msnr
		self.esnr=esnr
		self.msnrPcoda=msnrPcoda
		self.esnrPcoda=esnrPcoda
		self.snrtype=snrtype
		self.focal_mech=focal_mech
		self.datapercent=datapercent
		self.xcorr=xcorr
		self.mdp=mdp
		self.edp=edp
		self.mastertr=mastertr
		self.egftr=egftr
		self.mspec=mspec
		self.espec=espec
		self.stf=stf
	def __repr__(self):
		"""
		print spectral ratio with attributes.
		"""
		print_str=('Spectral ratio %s and %s;'
		'\n\t station %s channel %s;'
		%(self.master, self.egf, self.station, self.channel))
		return print_str
	def holed(self, min_snr=float(3), holeby='preP'):
		"""
		holes the function by min_snr, selecting the noise window
		by 'preP' - the window before the P arrival - use for P waves
		by 'Pcoda' - the window before the S arrival - use for S waves.
		"""
		freq1=[float(fr) for fr in self.freqs]
		srat1=[float(sr) for sr in self.specratio]
		spec1=self.mspec
		spec2=self.espec
		snr1=self.snr
		snr2=list(self.snrPcoda)
		min_snr=float(min_snr)
		snr1a=self.msnr
		snr1b=self.esnr
		snr2a=self.msnrPcoda
		snr2b=self.esnrPcoda
		print(self.snr)
		if holeby=='Pcoda':
			self.freqs=np.asarray([i for i,j in zip(freq1, snr2) if j > min_snr])
			if len(freq1) <1:
				self.datapercent = 0
			else:
				self.datapercent=float(len(self.freqs))/float(len(freq1))
			self.specratio=filterby(srat1,snr2,min_snr)
			self.mspec=filterby(spec1,snr2,min_snr)
			self.espec=filterby(spec2,snr2,min_snr)
			self.snr=filterby(snr1,snr2,min_snr)
			self.snrPcoda=np.asarray([i for i in snr2 if i >=min_snr])
			if not snr1a is None and not snr1b is None and not snr2a is None and not snr2b is None:
				self.msnr=filterby(snr1a,snr2, min_snr)
				self.esnr=filterby(snr1b,snr2,min_snr)
				self.msnrPcoda=filterby(snr2a,snr2,min_snr)
				self.esnrPcoda=filterby(snr2b, snr2,min_snr)
		else:
			print(holeby)
			self.freqs=[i for i,j in zip(freq1, self.snr) if j >= min_snr]
			if len(freq1) <1:
				self.datapercent = 0
			else:
				self.datapercent=float(len(self.freqs))/float(len(freq1))
			self.specratio=[i for i,j in zip(srat1, self.snr) if j >=min_snr]
			self.mspec=np.asarray([i for i,j in zip(spec1, self.snr) if j >=min_snr])
			self.espec=np.asarray([i for i,j in zip(spec2, self.snr) if j >=min_snr])
			self.freqs=np.asarray(self.freqs)
			self.snrPcoda=filterby(snr2,snr1,min_snr)
			self.specratio=np.asarray(self.specratio)
			self.snr=[i for i in snr1 if i >=min_snr]
			self.snr=np.asarray(self.snr) 
			if not snr1a is None and not snr1b is None and not snr2a is None and not snr2b is None:
				self.msnr=filterby(snr1a,snr1, min_snr)
				self.esnr=filterby(snr1b,snr1,min_snr)
				self.msnrPcoda=filterby(snr2a,snr1,min_snr)
				self.esnrPcoda=filterby(snr2b, snr1,min_snr)
		return self

def filterby(lista, listb, cutoff):
	listc=[i for i,j in zip(lista, listb) if j >= cutoff]
	return np.asarray(listc)
def deconvolve(specratio,method='multitaper',winlength=5.0,freqmin=0.5, freqmax=5.0,trim=None):
	#trim down the traces
	mdata=specratio.mastertr
	edata=specratio.egftr
	pick_large=mdata.stats.starttime+2.
	pick_small=edata.stats.starttime+2.
	if trim:
		mdata.trim(starttime=pick_large+trim[0], endtime=pick_large+trim[1])
		edata.trim(starttime=pick_small+trim[0], endtime=pick_small+trim[1])
	dt, coeff=xcorr_pick_correction(pick_large, mdata, pick_small, edata, t_before=0.25, t_after=1.0, cc_maxlag=1.5, filter="bandpass", filter_options={'freqmin': freqmin, 'freqmax': freqmax})
	pick_small=pick_small+dt #realign the traces by cross-correlation
	ts1=mdata.copy()
	ts1.trim(pick_large, pick_large+winlength)
	N = len(ts1)
	nfft=next_pow_2(N)
	ts2=edata.copy()
	ts2.trim(pick_small, pick_small+winlength)
	if method == 'multitaper':
		freqs,specs,mspecs,especs,deconvolved=specrat_gen(ts1,ts2,nfft,4)
	elif method == 'traditional':
		deconvolved=deconvf(ts2,ts1, ts1.stats.sampling_rate)
	M=np.arange(0,len(deconvolved))
	N=len(M)
	SeD=np.where(np.logical_and(M>=0, M<N/2))
	d1=deconvolved[SeD]
	SeD2=np.where(np.logical_and(M>N/2, M<=N+1))
	d2=deconvolved[SeD2]
	stf=np.concatenate((d2,d1))
	stf/=stf.max()
	return stf, coeff

def convolve(specratio, tr, stf='triangular',filterlim=None,trim=None):
	from scipy.signal import triang, convolve
	edata=specratio.egftr
	pick_small=edata.stats.starttime+2.
	ts1=edata.copy()
	if trim:
		ts1.trim(starttime=pick_small+trim[0], endtime=pick_small+trim[1])
	if filterlim:
		ts1.filter(type='bandpass', freqmin=filterlim[0], freqmax=filterlim[1])
	if stf == 'triangular': 
		n=tr*edata.stats.sampling_rate
		tstf=triang(round(n))
	return convolve(ts1,tstf)

def plot(specratios,min_snr=0.0, min_xcorr=0.0, snrtype='Pcoda',freqmin=0.5, freqmax=5.0,savefile=None):
	"""
	plots the spectral ratios by station.
	data can be selected by holign the spectra by min_snr
	or by throwing out egf events that do not exceed min_xcorr.
	to calculate xcorr and for plotting purposes the user
	can change the bandwidth by modifying freqmin and freqmax.
	the default is between 0.5 and 5 Hz which is the range
	of corner frequencies for M3-5 earthquakes at about 100 km depth.
	"""
	from mpl_toolkits.basemap import Basemap
	fig=plt.figure(figsize=(10,7))
	ax1=plt.subplot2grid((7,3),(0,0), rowspan=7)
	ax2=plt.subplot2grid((7,3),(0,1)) #master event trace
	ax3=plt.subplot2grid((7,3),(1,1), rowspan=6) #egf traces
#	ax7=plt.subplot2grid((7,3),(5,1), rowspan=2) #stfs.
	ax4=plt.subplot2grid((7,3),(0,2)) #master spectrum
	ax5=plt.subplot2grid((7,3),(1,2), rowspan=3) #egf spectrum
	ax6=plt.subplot2grid((7,3),(4,2), rowspan=3) #spetral ratios
	print('len specrats was ' + str(len(specratios)))	
	specrats=[specrat for specrat in specratios if specrat.datapercent > 0.4]
	print('len specrats is '+str(len(specrats))) #spectral ratios
	if len(specrats) < 1:
		return
	xcorrs=np.asarray([np.max(specrat.xcorr) for specrat in specrats])
	inds=np.argsort(xcorrs)
	new_specrats=[]
	for ind in inds:
		new_specrats.append(specrats[ind])
	specrats=new_specrats
	print('len specrats is ' + str(len(specrats)))
	stla=specrats[0].stla
	stlo=specrats[0].stlo
	mla=specrats[0].mla
	mlo=specrats[0].mlo
	mastertr=specrats[0].mastertr
	mastertr.detrend()
	mastertr.detrend('demean')
	mastertr.filter('bandpass', freqmin=freqmin, freqmax=freqmax)
	mspec=specrats[0].mspec
	tarray=np.arange(len(mastertr))*mastertr.stats.delta #time array
	if len(mspec) < 1:
		print('theres no spectrum?')
		return
	print('loading maps')
	map1=Basemap(projection='merc', llcrnrlat=mla-1.25, llcrnrlon=mlo-1.25,urcrnrlat=mla+1.25,urcrnrlon=mlo+1.25, resolution='f', ax=ax1)
	map1.drawmapboundary()
	map1.drawcoastlines()
	x,y=map1(stlo,stla)
	map1.scatter(x,y,marker='v', color='k')
	x1,y1=map1(mlo,mla)
	map1.scatter(x1,y1,marker='*', color='r')
	ax2.plot(tarray,mastertr.data, color='r')
	ax2.set_xlim((tarray[0], tarray[-1]))
	ax4.loglog(specrats[0].freqs, mspec, basex=10,basey=10, color='r')
	ax4.set_xticklabels([])
	ax2.set_xticklabels([])
	ax2.set_yticklabels([])
	ax2.set_title('traces')
	ax4.set_title('spectra')
	ax6.set_xlabel('spec ratio vs. log frequency')
#	ax6.set_ylabel('spectral ratio')
#	ax3.set_xticklabels([])
	ax3.set_xlabel('realigned time (s)')
	ax3.set_yticklabels([])
	plt.suptitle(specrats[0].master+' at ' + specrats[0].station)	
	print('still loading maps')
	print('plotting spectral ratios')
	for ie, specrat in enumerate(specrats):
		if min_snr > 0.0:
			print('holing to min snr ' + str(min_snr))
			specrat.holed(min_snr=min_snr, holeby=snrtype)
		if min_xcorr > 0.0 and specrat.xcorr < min_xcorr:
			print('removing the spectral ratios with xcorr below ' + str(min_xcorr))
			continue
		egftr=specrat.egftr
		egftr.detrend()
		egftr.detrend('demean')
		stf=specrat.stf
		egftr.filter('bandpass', freqmin=freqmin, freqmax=freqmax)
		dt, coeff=xcorr_pick_correction(mastertr.stats.starttime+2, mastertr, egftr.stats.starttime+2, egftr, t_before=0.25, t_after=1.0, cc_maxlag=1.5)
		ela=specrat.ela
		elo=specrat.elo
		xcorr=np.max(specrat.xcorr)
#		stf_simple = deconvf(egftr, mastertr, mastertr.stats.sampling_rate) 
		print('len stf is '  +str(len(stf)))
		if xcorr > 0.5: 
			colour='g'
		elif xcorr > 0.3:
			colour='b'
		elif xcorr > 0.2:
			colour='c'
		else:
			colour='m'
		ax5.loglog(specrat.freqs,specrat.espec,basex=10,basey=10,color=colour)
		ax6.loglog(specrat.freqs,specrat.specratio,basex=10,basey=10,color=colour)
		plotdata=egftr.data/max(abs(egftr.data))
		datalen=min(len(plotdata), len(tarray))
		ax3.plot(tarray[0:datalen]+dt,plotdata[0:datalen]+ie, linewidth=0.6, color=colour)
		ax3.text(0,ie,'{:0.2f}'.format(coeff))
		x2,y2=map1(elo,ela)
		map1.scatter(x2,y2,marker='+',color=colour)
	#	try:
	#		if xcorr >0.2:
	#			ax7.plot(stf, color=colour)
#	#			ax7.plot(stf_simple, color='m')
	#	except:
	#		logging.exception('values at exception ')
	ax3.set_xlim((tarray[0], tarray[-1]))
	ax6.set_xlim((10**-1.2, 10**1.2))
	ax4.set_xlim((10**-1.2, 10**1.2))
	ax5.set_xlim((10**-1.2, 10**1.2))
	ax5.set_xticklabels([])
	ax3.set_ylim((ie-12, ie+1))
	if savefile:
		plt.savefig(savefile)
	else:
		plt.show()
	return 

def logbin(f,a,nbins,flims):
	"""
	smooth spectrum in log space
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

def specrat_gen(clipwin, clipwin2,nopts, mt_tb):
	from mtspec import mtspec, mt_deconvolve
	if not len(clipwin.data) == len(clipwin2.data):
		minlength=min(len(clipwin.data), len(clipwin2.data))
		clipwin.data=clipwin.data[0:minlength-1]
		clipwin2.data=clipwin2.data[0:minlength-1]
	r=mt_deconvolve(clipwin.data, clipwin2.data, clipwin.stats.delta, nfft=nopts, time_bandwidth=mt_tb, number_of_tapers=7, weights='constant', demean=True)
        mspecs=r["spectrum_a"]
        especs=r["spectrum_b"]
        specs=r["spectral_ratio"]
        ffreq=r["frequencies"]
        decons=r["deconvolved"]
	return ffreq, specs, mspecs, especs, decons

#def spectrum_gen(ts_in, picksfile, station, args, mt_tb=4, debug=0, return_timeseries=False):
def spectrum_gen(ts_in, pick, args, mt_tb=4, debug=0, return_timeseries=False, usedecon=False, ts_in2=None, pick2=None, npick=None, npick2=None, npickp=None, npickp2=None):
	"""
	returns the spectrum of a timeseries
	INPUT
	ts_in timeseries as an obspy trace
	pick pick as time
	station string
	args see 
	"""
	from mtspec import mtspec, mt_deconvolve
	timeseries=ts_in.copy()
	if usedecon:
		timeseries2=ts_in2.copy()
	#arguments
	#fftype=args.ft
	winlength=args.W
	shift=args.S
	stype=args.P
	nwins=args.N
	snrtype=args.snrtype
	nlogbins=args.lb
	smoothfactor=3	
	flims=[-2, 1.2]
	specs=[]
	ffreq=[]
	if usedecon:
		msnrs=[]
		esnrs=[]
		mspecs=[]
		especs=[]
		decons=[]
		esnrps=[]
		msnrps=[]
		noisewin=timeseries.copy()
		noisewin.trim(npick,npick+winlength)
		noisewin2=timeseries2.copy()
		noisewin2.trim(npick2, npick2+winlength)
		noisewinpcoda=timeseries.copy()
		noisewinpcoda.trim(npickp, npickp+winlength)
		noisewinpcoda2=timeseries2.copy()
		noisewinpcoda2.trim(npickp2,npickp2+winlength)
	lenwin=0
	for i in range(1,nwins+1):
		clipwin=timeseries.copy()
		clipstart=pick+(i-1)*shift
		clipend=clipstart+winlength
		if clipwin.stats.endtime-clipend < 0:
			if debug > 1:
				print('at the end of the data')
			continue
		if clipstart-clipwin.stats.starttime < 0:
			if debug > 1:
				print('starting before the data')
			continue
		clipwin.trim(clipstart,clipend)
		lenwin=len(clipwin.data)
		if lenwin<=256:
			nopts=256
		elif lenwin<=512:
			nopts=512
		elif lenwin<1024:
			nopts=1024
		else:	
			nopts=2048

		if not usedecon:
			spec,freq,jackknife,_,_=mtspec(data=clipwin.data, delta=clipwin.stats.delta, time_bandwidth=mt_tb,nfft=nopts,statistics=True)
			specs.append(np.sqrt(spec))
			ffreq=freq
		else:
			clipwin2=timeseries2.copy()
			clipstart2=pick2+(i-1)*shift
			clipend2=clipstart2+winlength
			if clipwin2.stats.endtime-clipend2 < 0:
				if debug > 1:
					print('at the end of the data')
				continue
			if clipstart2-clipwin2.stats.starttime < 0:
				if debug > 1:
					print('starting before the data')
				continue
			clipwin2.trim(clipstart2,clipend2)
			freq, spec, spec1, spec2, decon=specrat_gen(clipwin,clipwin2,nopts, mt_tb)
                	freqnL,msnr,_,_,denoiseL=specrat_gen(clipwin,noisewin, nopts,mt_tb)
                	freqnS,esnr,_,_,denoiseS=specrat_gen(clipwin2, noisewin2, nopts, mt_tb)
                	freqnpL,msnrp,_,_,denoisepL=specrat_gen(clipwin,noisewinpcoda, nopts,mt_tb)
                	freqnpS,esnrp,_,_,denoisepS=specrat_gen(clipwin2, noisewinpcoda2, nopts, mt_tb)
#			specs.append([spec,spec1,spec2,msnr,esnr,msnrp,esnrp])
			msnrs.append(msnr)
			esnrs.append(esnr)
			specs.append(spec)
			mspecs.append(spec1)
			especs.append(spec2)
			decons.append(decon)
			msnrps.append(msnrp)
			esnrps.append(esnrp)
			ffreq=freq
	specs=np.average(specs,axis=0)
	if usedecon:
		msnrs=np.average(msnrs,axis=0)
		esnrs=np.average(esnrs,axis=0)
		mspecs=np.average(mspecs,axis=0)
		especs=np.average(especs,axis=0)
		decons=np.average(decons,axis=0)
		msnrps=np.average(msnrps,axis=0)
		esnrps=np.average(esnrps,axis=0)
	#smooth the individual spectra before taking spectral ratio.
	if len(ffreq)==0:
		print('the spectrum has length 0')
		return
	ffreq1,specs=logbin(ffreq,specs,nbins=nlogbins,flims=flims)
	if usedecon:
		ffreq1,msnrs=logbin(ffreq,msnrs,nbins=nlogbins,flims=flims)
		ffreq1,esnrs=logbin(ffreq,esnrs,nbins=nlogbins,flims=flims)
		ffreq1,mspecs=logbin(ffreq,mspecs,nbins=nlogbins, flims=flims)
		ffreq1,especs=logbin(ffreq,especs,nbins=nlogbins,flims=flims)
		ffreq1,msnrps=logbin(ffreq,msnrps,nbins=nlogbins,flims=flims)
		ffreq1,esnrps=logbin(ffreq,esnrps,nbins=nlogbins,flims=flims)
	if args.sm:
		specs=smooth(specs,smoothfactor)
	if usedecon:
		msnrs=smooth(msnrs,smoothfactor)
		esnrs=smooth(esnrs,smoothfactor)
		mspecs=smooth(mspecs, smoothfactor)
		especs=smooth(especs, smoothfactor)
		msnrps=smooth(msnrps, smoothfactor)
		esnrps=smooth(esnrps, smoothfactor)	
		return ffreq1, specs, mspecs, especs, decons, msnrs, esnrs, msnrps, esnrps
	else:
		return ffreq1,specs
 
def xcorr(master, egf, prefilter=6.0):
	from obspy.signal.cross_correlation import correlate
	"""
	function to take cross correlation of master and egf
	prefilter = low cut to take before cross correlation
	"""
	m1=master.copy()
#	print(m1)
	m1.data=np.divide(m1.data, max(m1.data))
#	m1=m1.filter('lowpass',freq=prefilter)
	e1=egf.copy()
	e1.taper(max_percentage=0.2, type='cosine')
	e1.data=np.divide(e1.data, max(e1.data))
	e1.filter('lowpass',freq=prefilter)
	correlation=correlate(m1,e1,m1.stats.npts/2)		
	return correlation, m1, e1	
	
def get_spec_ratio(data_small, mdata, data_large, station, args, debug=0):
	"""
	generate spectral ratios
	preprocessing of Kiban network metadata.
	DO NOT USE
	function only works if you named your files exactly like I did -
	TODO
	update to take in all file types regardless of name type
	"""
	flims=[-1, np.log10(50)] #the limits of the log binning
	#unpack arguments
	prefilter=args.F
	stype=args.P #phase to generate the spectral ratio.
	winlength=args.W
	shift=args.S
	nwins=args.N
	nlogbins=args.lb
	fftype=args.ft
	edata=data_small[0]
	#load data from small and large events
#	ms_small=data_small[0]
#	ela=float(ms_small.split('_evla')[-1].split('_')[0])
#	elo=float(ms_small.split('_lo')[1].split('_')[0])
#	edp=float(ms_small.split('_dp')[1].split('_')[0])
#	pick_small=data_small[1]
#	stla=float(ms_small.split('stla')[-1].split('_')[0])
#	stlo=float(ms_small.split('_lo')[-1].split('.ms')[0])
#	egf_name=ms_small.split('evt_')[-1].split('_evla')[0]
#	master_name=ms_large.split('evt_')[-1].split('_evla')[0]
#	mla=float(ms_large.split('_evla')[-1].split('_')[0])
#	mlo=float(ms_large.split('_lo')[1].split('_')[0])
#	mdp=float(ms_large.split('_dp')[1].split('_')[0])
#	edata=read(ms_small)
#	mdata=read(ms_large)
	ela=data_small[1].la
	elo=data_small[1].lo
	edp=data_small[1].dp
	pick_small=data_small[1].pick
	stla=data_small[1].stla
	stlo=data_small[1].stlo
	egf_name=data_small[1].name
	master_name=data_large.name
	mla=data_large.la
	mlo=data_large.lo
	mdp=data_large.dp
	#select the channel for the data
	#use alternate names for channels from different sources
	echannels=[tr.stats.channel for tr in edata]
	mchannels=[tr.stats.channel for tr in mdata]
	empty_specrat=specrat(np.asarray([]),np.asarray([]), np.asarray([]),master_name,mla,mlo,egf_name,ela,elo,station,stla,stlo,args.C)
	if not args.C in echannels and len(list(set(echannels)&set(args.altC))) == 0:
		if debug > 0:
			print('we didnt find this channel in egf at station '+station)
		return empty_specrat
	if not args.C in mchannels and len(list(set(mchannels)&set(args.altC))) == 0:
		if debug > 0:
			print('we didnt find this channel in master at station '+station)
		return empty_specrat
	allchans=args.altC
	allchans.append(args.C)
	echannel=list(set(echannels)&set(allchans))[0]
	mchannel=list(set(mchannels)&set(allchans))[0]
	edata=edata.select(channel=echannel)[0]
	mdata=mdata.select(channel=mchannel)[0]
	mdata.detrend()
	mdata.detrend('demean')
	edata.detrend()
	edata.detrend('demean')
	if prefilter:
#		flims=np.log10(prefilter)
		edata.filter(type='bandpass', freqmin=prefilter[0], freqmax=prefilter[1])
		mdata.filter(type='bandpass', freqmin=prefilter[0], freqmax=prefilter[1])
	#find the correct picks based on phase desired
#	pick_large=str(pick_large)
	pick_large=data_large.pick
#	pick_small=read_events(pick_small)
#	pick_small=pick_small[0].picks
#	pick_large = read_events(pick_large)
#	pick_large = pick_large[0].picks
#	print(station)
	spick_large = [ipick for ipick in pick_large if ipick.phase_hint == 'S' and ipick.waveform_id.station_code[-4:]==station[-4:]]
	spick_small = [ipick for ipick in pick_small if ipick.phase_hint == 'S' and ipick.waveform_id.station_code[-4:]==station[-4:]]
	ppick_large = [ipick for ipick in pick_large if ipick.phase_hint == 'P' and ipick.waveform_id.station_code[-4:]==station[-4:]]
	ppick_small = [ipick for ipick in pick_small if ipick.phase_hint == 'P' and ipick.waveform_id.station_code[-4:]==station[-4:]]
	if stype == 'S' or stype == 'coda' or stype == 'lateS':
		pick_large=spick_large
		pick_small=spick_small
	if stype == 'P':
		pick_large=ppick_large
		pick_small=ppick_small
	if len(pick_large) < 1 or len(pick_small) < 1:
		if debug > 0:
			print('no pick found in the pickfile for station ' + station)
		return empty_specrat
	print(pick_large[0].waveform_id.station_code)
	print(pick_small[0].waveform_id.station_code)
	pick_large=pick_large[0].time
	pick_small=pick_small[0].time
#	print('pick large time: ')
#	print(pick_large)
#	print(mdata.stats.starttime)
#	print(mdata.stats.endtime)
#	print('pick smal time')
#	print(pick_small)
#	print(edata.stats.starttime)
#	print(edata.stats.endtime)
	try:
		if stype == 'coda' or stype == 'S' or stype == 'lateS':
			dt, coeff=xcorr_pick_correction(pick_large, mdata, pick_small, edata, t_before=0.25, t_after=1.0, cc_maxlag=1.5, filter="bandpass", filter_options={'freqmin': 0.5, 'freqmax': 5.0})
		else:
			dt, coeff=xcorr_pick_correction(pick_large, mdata, pick_small, edata, t_before=0.25, t_after=1.0, cc_maxlag=1.5, filter="bandpass", filter_options={'freqmin': 0.5, 'freqmax': 10.0})
	except:
		logging.exception('values at exception:')
		return empty_specrat
	pick_small=pick_small+dt
	if stype=='coda':
		ttime_l = pick_large-mdata.stats.starttime
		ttime_s = pick_small-edata.stats.starttime
		pick_large = mdata.stats.starttime+1.5*ttime_l
		pick_small = edata.stats.starttime+1.5*ttime_s
	if stype=='lateS':
		pick_large=pick_large+2
		pick_small=pick_small+2
	#generate p coda noise windo
#	if stype == 'coda' or stype == 'lateS' or stype == 'S':
	picknp_large=spick_large[0].time-winlength-1
	picknp_small=spick_small[0].time-winlength-1
	#generate pre p noise window
	if len(ppick_large) < 1:
		pickn_large=mdata.stats.starttime
	else:
		pickn_large=ppick_large[0].time-winlength-1
	if len(ppick_small) < 1:
		pickn_small=edata.stats.starttime
	else:
		pickn_small=ppick_small[0].time-winlength-1
		if pickn_large-mdata.stats.starttime < 0:
			pickn_large=mdata.stats.starttime
		if pickn_small-edata.stats.starttime < 0:
			pickn_small=edata.stats.starttime
	ts1=mdata.copy()
	ts1.trim(pick_large-2, pick_large+8)
	ts2=edata.copy()
	ts2.trim(pick_small-2, pick_small+8)
	if fftype=='multitaper':
		try:
			freq1,spec1=spectrum_gen(mdata,pick_large,args,debug=debug)	
			freq2,spec2=spectrum_gen(edata,pick_small,args,debug=debug) 
			argsn=copy.deepcopy(args)
			#generate snr
			argsn.N=1
			freqn0,specn0=spectrum_gen(mdata,pickn_large,argsn,debug=debug)
			freqn,specn=spectrum_gen(edata,pickn_small,argsn,debug=debug)
			freqn0p,specn0p=spectrum_gen(mdata,picknp_large,argsn,debug=debug)
			freqnp,specnp=spectrum_gen(edata,picknp_small,argsn,debug=debug)
			freq=freq1
		except:
			logging.exception('values at exception:')
			return empty_specrat
		if len(freq) < 1 or len(spec1)<1 or len(spec2)<1:
			if debug > 0:
				print('we were not able to generate spectrum')
			return empty_specrat
		if not len(spec1)==len(spec2) or not len(spec2)==len(specn):
			if debug > 0:
				print('length of spectra are not equal! check the picks')
			return empty_specrat
		try:
		#fill very high SNR with some finite value
			esnr=np.divide(spec2,specn)
			msnr=np.divide(spec1,specn0)
			esnr[np.isnan(esnr)]=100.0
			msnr[np.isnan(msnr)]=100.0
			snr=np.minimum(esnr,msnr)
			specratio=np.divide(spec1,spec2)
			snrpcoda1=np.divide(spec1,specn0p)
			snrpcoda2=np.divide(spec2,specnp)
			snrpcoda1[np.isnan(snrpcoda1)]=100.0
			snrpcoda2[np.isnan(snrpcoda2)]=100.0
			snrpcoda=np.minimum(snrpcoda1, snrpcoda2)
		except:
			logging.exception('values at exception:')
			return empty_specrat
	elif fftype == 'decon':
		try:
			freq,specratio,spec1, spec2, decon,msnr,esnr,snrpcoda1,snrpcoda2=spectrum_gen(mdata,pick_large,args, usedecon=True, ts_in2=edata,pick2=pick_small,npick=pickn_large, npick2=pickn_small, npickp=picknp_large, npickp2=picknp_small,debug=debug)
		except:
			logging.exception('values at exception')
			return empty_specrat
		M=np.arange(0,len(decon))
		N=len(M)
		SeD=np.where(np.logical_and(M>=0, M<N/2))
		d1=decon[SeD]
		SeD2=np.where(np.logical_and(M>N/2, M<=N+1))
		d2=decon[SeD2]
		stf=np.concatenate((d2,d1))
		stf/=stf.max()
		snr=np.minimum(esnr, msnr)
		snrpcoda=np.minimum(snrpcoda1,snrpcoda2)
	specratio=specrat(freq,specratio,snr,master_name,mla,mlo,egf_name,ela,elo,station,stla,stlo,args.C,mdp=mdp, edp=edp, mastertr=ts1,egftr=ts2,mspec=spec1,espec=spec2,xcorr=coeff, snrPcoda=snrpcoda, msnr=msnr, esnr=esnr, msnrPcoda=snrpcoda1, esnrPcoda=snrpcoda2)
	if fftype == 'decon':
		specratio.stf=stf
	return specratio

