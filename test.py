import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from obspy import read, Stream
from obspy.signal.trigger import classic_sta_lta, plot_trigger
from obspy.signal.trigger import trigger_onset
import glob
from argparse import Namespace
#from streamPick import *
from obspy.core.event.origin import Pick
from obspy.core.event.base import WaveformStreamID
from spec_rat0 import *

data=glob.glob('*1')
#data name is of the format NNNNNNYYMMDDHHMM.CC1
#NNNNNN 6 digit station name, e.g. HRSH09 (strong motion meter associated with Hi-net
#station HRSH, CC 2 digit channel, the number 1 indicates down borehole (as opposed
#to 2 which is at surface.
stations=[dat[0:6] for dat in data]
stations=list(set(stations)) #get a list of all stations
channels=['EW1', 'NS1']
args=Namespace(F=None, ft='decon', W=8.0, S=1.5, P='S', N=1, lb=100, C='NN', R=None, sm=True, snr=3.0, snrtype='Pcoda',xcorr=0.4)
for station in stations:
	for channel in channels:
		specratios=[]
		if channel == 'EW1':
			args.altC=['EW1', 'EW', 'EW2']
		else:
			args.altC=['NS1', 'NS', 'NS2']
		data_subset=[dat for dat in data if station in dat and channel in dat]
		data_subset=[read(dat) for dat in data_subset]
		data_magnitudes=[stream[0].stats.knet.mag for stream in data_subset]
		if not any([m for m in data_magnitudes if m > 6.0]):
			print('the master event was not found in this station or channel')
			continue		
		master_event=[i for i,j in zip(data_subset, data_magnitudes) if j > 6.0][0]
		egf_event=[i for i,j in zip(data_subset, data_magnitudes) if not j > 6.0]
		master_event.trim(master_event[0].stats.starttime, master_event[0].stats.starttime+100)		
		df=master_event[0].stats.sampling_rate
		cft=classic_sta_lta(master_event[0].data,int(2*df), int(15*df))
		times=trigger_onset(cft,0.9*max(cft),0.5*max(cft),max_len=1600)
		picktime=times[0][1]
		print('picktime is ' + str(picktime) + 'samples')
		picktime=master_event[0].stats.starttime+(picktime*master_event[0].stats.delta)
		print(master_event[0].stats.starttime)
		print(master_event[0].stats.sampling_rate)
#	print(picktime)
		picks=[]
		picks.append(Pick(time=picktime, waveform_id=WaveformStreamID(station_code=master_event[0].stats.station, channel_code=master_event[0].stats.channel), phase_hint='S'))
		master_data=Namespace(pick=picks,la=master_event[0].stats.knet.evla, lo=master_event[0].stats.knet.evlo, dp=master_event[0].stats.knet.evdp, stla=master_event[0].stats.knet.stla, stlo=master_event[0].stats.knet.stlo, name=master_event[0].stats.knet.evot.strftime('%Y%m%dT%H%M%S')+'_M'+str(master_event[0].stats.knet.mag))	
		for stream in egf_event:
			stream[0].trim(stream[0].stats.starttime, stream[0].stats.starttime+100)		
			df=stream[0].stats.sampling_rate
			cft=classic_sta_lta(stream[0].data,int(2*df), int(15*df))
			times=trigger_onset(cft,0.9*max(cft),0.5*max(cft),max_len=1600)
			picktime=times[0][0]
			picktime=stream[0].stats.starttime+picktime*stream[0].stats.delta
			picks=[]
			picks.append(Pick(time=picktime, waveform_id=WaveformStreamID(network_code=stream[0].stats.network, station_code=stream[0].stats.station, channel_code=stream[0].stats.channel), phase_hint='S'))
			egf_data=Namespace(pick=picks,la=stream[0].stats.knet.evla, lo=stream[0].stats.knet.evlo, dp=stream[0].stats.knet.evdp, stla=stream[0].stats.knet.stla, stlo=stream[0].stats.knet.stlo, name=stream[0].stats.knet.evot.strftime('%Y%m%dT%H%M%S')+'_M'+str(stream[0].stats.knet.mag))	
#	print(picktime)
			specratio=get_spec_ratio((stream,egf_data), master_event, master_data, station, args)
			specratios.append(specratio)
		plot(specratios, savefile='geiyo_'+station+'_'+channel+'.png')
