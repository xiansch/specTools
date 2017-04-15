from obspy.io.sac.sacpz import attach_paz
from obspy.signal.invsim import corn_freq_2_paz
from obspy import read, Stream, Trace, read_events
from obspy.signal.trigger import ar_pick
from obspy.signal.detrend import spline
from obspy.signal.util import smooth
import numpy as np
import matplotlib as mpl
from matplotlib import pylab as plt
from matplotlib import gridspec
from obspy.signal.spectral_estimation import fft_taper
from matplotlib.mlab import psd
from matplotlib.mlab import magnitude_spectrum
import glob
import sys, traceback, logging

def process_multitaper(stream, station, picksfile,debug=0,winlength_p=5.0,winlength_s=8.0,n_pwins=3,n_swins=5, pshift=2, sshift=4,coda=True,coda_delay=8,n_codawins=5,coda_shift=4,time_bandwidth=2.5,minsnr=2,ppicks=True,intype='vel',outtype='vel',channel='U*',remove_resp=False,statistics=False):
	"""
	returns the spectra of a seismic record in msfile format.
	:param msfile: : class:'string'
	name of the mseed file containing the record.
	:param station: : class:'string'
	name of the station.
	:param winlength: : class: 'float'
	length in seconds of the spectral window.
	:param plotylim: : class: tuple
	ylimits of the plot.
	:param type: : class: 'string'
	'disp' or 'vel' or 'acc' (currently 'acc' not supported)
	returns: p spectra at station, 
	"""
	from mtspec import mtspec
	from mtspec.util import _load_mtdata
	pspecs = []
	pjks = []
	sspecs = []
	sjks = []
	cspecs = []
	streamcopy = stream.copy()
	utrace = streamcopy.select(channel=channel)[0]
	if remove_resp:
		paz_1hz = corn_freq_2_paz(1.0, damp=0.707)
		paz_1hz['sensitivity'] = 1.0
		attach_paz(utrace,'sacpz/N.'+station+'.'+channel+'.SAC_PZ')
	utrace.detrend('demean')
	df = utrace.stats.sampling_rate
	cat = read_events(picksfile)
	if debug > 0:
		print('we have ' + str(len(cat[0].picks)) + ' picks in file')
		print('the station is ' + station)
	spick = [pick for pick in cat[0].picks if pick.phase_hint == 'S' and pick.waveform_id.station_code == station][0].time
	#not all stations have p-pick, so check this
	try:
		ppick = [pick for pick in cat[0].picks if pick.phase_hint == 'P' and pick.waveform_id.station_code == station][0].time
	except:
		print('s pick only')
		ppicks=False
		ppick=0
	cpick = spick + coda_delay		
	cstartwin = cpick
	if debug > 0:
		print('ppick: ' + str(ppick))
		print('spick: ' + str(spick))
		print('starttime: ' + str(utrace.stats.starttime))
		print('coda: ' + str(cpick))
	#use trace.simulate to remove station response (result will be integrated displacement trace) and remove trend
	if remove_resp or outtype == 'disp':
		utrace.simulate(paz_remove = utrace.stats.paz, paz_simulate=paz_1hz)
		spline(utrace.data,order=2,dspline=1000,plot=False)
#	select noise window for SNR calculation before P arrival
	pbeforewin = ppick-10
	if not ppicks:
		pbeforewin = utrace.stats.starttime
		freqsp=0
	pbefore2 = utrace.copy()
	pbefore2.trim(starttime=pbeforewin, endtime=pbeforewin+winlength_s)
	specb, freqs, jackknifeb, _, _ = mtspec(data=pbefore2.data, delta=pbefore2.stats.delta, time_bandwidth=time_bandwidth, statistics=True)
	#multi-windowed average of s arrival
	swins = []
	for i2 in range(1,n_swins+1):
		swin=utrace.copy()
		sstartwin=spick+(i2-1)*sshift
		sendwin = sstartwin+winlength_s
		swin.trim(sstartwin, sendwin)
		specs, freqss, jackknifes, _, _ = mtspec(data=swin.data, delta=swin.stats.delta, time_bandwidth=time_bandwidth, statistics=True)
		ssnr = np.sum(specs)/np.sum(specb)
		print('SSNR: ' + str(ssnr))
		del sstartwin
		if i2 == 1 and ssnr < minsnr:
			print('the s spectrum does not exceed SNR ' + str(minsnr) + '!')
			return
		elif ssnr < minsnr:
			print('s lower than min SNR')
			break
		sspecs.append(np.sqrt(specs))
		sjks.append(jackknifes)
	if ppicks:
		pbefore =utrace.copy()
		pbefore.trim(starttime=pbeforewin, endtime=pbeforewin+winlength_p)
		specbp, freqsbp, jackknifebp, _, _ = mtspec(data=pbefore.data, delta=pbefore.stats.delta, time_bandwidth=time_bandwidth, statistics=True)
		for ip in range(1,n_pwins+1):
			pwin = utrace.copy()
			pstartwin = ppick+(ip-1)*pshift
			pwin.trim(starttime=pstartwin, endtime=pstartwin+winlength_p)
			specp, freqsp, jackknifep, _, _ = mtspec(data=pwin.data, delta=pwin.stats.delta, time_bandwidth=time_bandwidth, statistics=True)
			psnr = np.sum(specp)/np.sum(specbp)
			if psnr < minsnr:
				print('p pick lower than min SNR')
				break
			pspecs.append(np.sqrt(specp))
			pjks.append(jackknifep)
	for i1 in range(1, n_codawins+1):
		cwin=utrace.copy()
		cstartwin=cpick+(i1-1)*coda_shift
		cwin.trim(starttime=cstartwin, endtime=cstartwin+winlength_s)
		specc, freqsc, jackknifec, _, _ = mtspec(data=cwin.data, delta = cwin.stats.delta, time_bandwidth=time_bandwidth, statistics=True)	
		csnr = np.sum(specc)/np.sum(specb)
		if csnr < minsnr:
			print('ignoring coda due to SNR')
			break
		cspecs.append(np.sqrt(specc))
	#if we converted to disp to remove station response, convert back to vel
	if outtype == 'vel' and remove_resp:
		sspecs = [2*np.pi*freqss*specs for specs in sspecs]
		pspecs = [2*np.pi*freqsp*specp for specp in pspecs]
		cspecs = [2*np.pi*freqsc*specc for specc in cspecs]
	return freqss, sspecs, sjks,freqsp, pspecs, pjks, cspecs


