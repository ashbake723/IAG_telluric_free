# Store all fxns for fitting lines here


import astropy.io.fits as fits
import matplotlib.pylab as plt
from matplotlib import gridspec
import numpy as np
import scipy.optimize as opt
from scipy.signal import fftconvolve
from scipy.interpolate import interp1d
from scipy import interpolate
from scipy.interpolate import splev, BSpline
import os,sys
from astropy.time import Time
import glob,scipy, urllib, json
import ephem

from utils import *

sys.path.append('read_opus/')
import opus2py as op

plt.ion()

class data():
	"""
	Main storage object for data files
	
	TO DO: 
	shift each spectrum according to iodine so in lab rest frame then remove iodine
	"""
	def __init__(self,
				input_path,
				night,
				v0=9000,
				vf=20000,
				nload=None,
				lon='9.9158',
				lat= '51.5413',
				elevation=100,
				plot_data=False):
	
		datapath = input_path + 'FTS/%s/' %night
		cal_file = input_path + 'PREFITS/test_%s.txt' %night
		tau_file = input_path + 'TAU/tau_%s.txt'   %night

		self.input_path = input_path
		# Load calibration data
		self.load_cal_file(cal_file)
		self.load_tau_file(tau_file)
		# load raw iodine file
		self.load_iodine()
		# load data dictionary
		self.load_data(datapath,v0=v0,vf=vf,nload=nload,lon=lon,lat=lat,elevation=elevation)
	
		# Define some things
		self.v0 = v0
		self.vf = vf
		self.nspec    = len(self.data['airmass'])
		self.specnums = self.data['s'].keys()
		self.specnums.sort()
		self.nwave    = len(self.data['s'][self.specnums[0]])
	
	def get_airmass(self,t,lon='9.9158', lat='51.5413', elevation=100):
		"""
		Get airmass of sun from time, long and lat and elevation of site
		assumes Gottingen parameters
	
		# source for airmass calc : http://www.ftexploring.com/solar-energy/air-mass-and-insolation2.htm#fn2
	
		inputs:
		-------
		t - time in JD
		lon - longitude of site (str)
		lat - latitude of site  (str)
		elevation - elevation   (int)
		"""
		gottingen = ephem.Observer()
		gottingen.lon = lon
		gottingen.lat = lat
		gottingen.elevation = elevation # units? m?
		gottingen.date = Time(t, format='jd', scale='utc').iso
		sun = ephem.Sun(gottingen)
		altitude = sun.alt
	
		ZA = np.pi/2 - sun.alt
		airmass = 1/(np.cos(ZA) + (0.50572 * (96.07995 - ZA*180/np.pi)**-1.6364))
	
		return airmass
	
	
	def get_time(self,time,date):
		"""
		Given the time and date of the FTS file 
		(from meta data loaded by opus2py), 
		return sjd as Time object
		"""
		sts = date[6:] + '-' + date[3:5] + '-' + date[0:2] + ' ' + time[0:12]
		gmtplus = float(time[18])
		sjd = Time(sts, format='iso', scale='utc').jd - gmtplus/24.0 # subtract +1 hr
		return sjd
	
	def get_cont(self,vFTS,spec,nstep=10000):
		"""
		given v, s from FTS data, determine continuum approximated by 1d interpolation
	
		inputs:
		------
		vFTS : v of FTS (units in wavenumber)
		spec : spectrum 
	
		outputs:
		-------
		spl - continuum interpolation object to be called continuum  = spl(v')
		"""
		x_cont = []
		y_cont = []
		for i in range(len(spec)/nstep):
			# Find local maximum
			local_max = np.where(spec[i*nstep:(i+1)*nstep] == np.max(spec[i*nstep:(i+1)*nstep]))[0]
			# Append to x,y arrays
			if vFTS[local_max[0] + i*nstep] < 15793 or vFTS[local_max[0] + i*nstep] > 15799:
				x_cont.append(vFTS[local_max[0] + i*nstep])
				y_cont.append(spec[local_max[0] + i*nstep])
	
		# Eliminate crazy point that's in FTS spectrum
		igd = np.where(np.array(y_cont) < 3*np.mean(y_cont))[0]
	
		# interpolate final result:
		spl = interp1d(np.array(x_cont)[igd][::-1],np.array(y_cont)[igd][::-1],\
			bounds_error=False,fill_value='extrapolate')
	
		return spl
		
	def load_cal_file(self,cal_file):
		"""
		Load prefit data into dictionary
		
		can do: 
		t = [cal[key]['t'] for key in cal.keys()]
		dnu = [cal[key]['dnu'] for key in cal.keys()]
		"""
		cal = {}
		if os.path.exists(cal_file):
			f = np.loadtxt(cal_file)
			for i in range(len(f)):
				specnum = '{num:04d}'.format(num=int(f[:,0][i]))
				cal[specnum]  =   {}
				cal[specnum]['t']         = f[:,1][i]
				cal[specnum]['eph']       = f[:,3][i]
				cal[specnum]['dnu']       = f[:,4][i]
				cal[specnum]['dnu_err']   = f[:,5][i]
				cal[specnum]['vel']       = f[:,6][i]
				cal[specnum]['vel_err']   = f[:,7][i]
				cal[specnum]['pfit_chi2'] = f[:,8][i]
		
		self.cal = cal
	
	def load_tau_file(self,tau_file):
		"""
		Load prefit tau data into dictionary
		
		can do: 
		t = [cal[key]['t'] for key in cal.keys()]
		dnu = [cal[key]['dnu'] for key in cal.keys()]
		"""
		tau = {}
		if os.path.exists(tau_file):
			f = np.loadtxt(tau_file)
			for i in range(len(f)):
				specnum = '{num:04d}'.format(num=int(f[:,0][i]))
				tau[specnum]         = f[:,1][i]
		
		ff = open(tau_file,'r')
		lines = ff.readlines()
		ff.close()

		self.tau = tau
		self.CD_H2O = float(lines[1].split()[-1])

	def load_iodine(self):
		"""
		Shift data according to iodine best fit
		
		files in prefit data
		"""
		# Load Iodine File
		iodine_file = self.input_path + 'IODINE/iodine.fits'
		f_iod = fits.open(iodine_file)
		s_iod = f_iod[0].data
		f_iod.close()
		
		iod_continuum = self.get_cont(np.arange(len(s_iod)),s_iod,nstep=800)
		sflat         = s_iod/iod_continuum(np.arange(len(s_iod)))
		sflat[np.where(sflat < 0)] = 1e-10

		self.iodine = sflat
	
	def load_data(self,datapath,v0=9000,vf=20000,nload=None,lon='9.9158',lat='51.5413',elevation=100):
		"""
		load fts data
	
		inputs:
		-------
		datapath : path to ze data
		v0, vf   : wavenumber defined boundaries to take subspec
	
		useful thing for dictionary manipulation: 
		t_arr = [data['t'][key] for key in specnums]
	
		outputs:
		--------
		data : dictionary of data, keys: v,s,unc,airmass,t
	
		"""
		speclist  = glob.glob('%s/*I2.0*' %datapath)
		if len(speclist) == 0:
			speclist  = glob.glob('%s/*0.0*' %datapath) #when I2 not present
		if len(speclist) == 0:
			speclist  = glob.glob('%s/*I2*.0*' %datapath) #when no halogens
	
		speclist = np.sort(speclist)
		
		# determine number to load
		nload = len(speclist) if nload == None else nload

		# load files to dictionary
		data = {}
		data['s'], data['v'], data['t'], data['unc'], data['tau'],data['airmass'], data['iodine'] = {},{},{},{},{},{},{}
		data['dnu'], data['vel'] = {}, {}
		for specname in speclist[0:nload]:
			# load spectrum
			specnum      = specname.split('.')[-1]
			raw_spec     = np.array(op.read_spectrum(specname))
		
			meta = op.get_spectrum_metadata(specname)      # meta data for first file
			v    = np.linspace(meta.xAxisStart,meta.xAxisEnd,meta.numPoints)
			isub = np.where( (v > v0) & (v < vf) )[0]
			
			# Skip completely if there are nans
			if len(np.where(np.isnan(raw_spec))[0]) > 0:
				print 'skipping ' + str(specnum)
				continue # skip if there are nans
		
			# Flatten data and get uncertainty - reject outlier datas
			continuum =  self.get_cont(v,raw_spec,nstep=10000)
			flat_spec =  raw_spec/continuum(v)
			flat_spec[np.where(flat_spec < 0)[0]] = np.min(flat_spec[np.where(flat_spec > 0)[0]])
			
			unc = np.ones(len(raw_spec)) * np.std(flat_spec[693750:694631]) # ash hand picked this region
			unc[np.where(flat_spec < 0.05)] = np.mean(unc)*2 # saturated points should have higher errors
			unc[np.where(flat_spec < 0.03)] = 1 # saturated points should be 
			unc[np.where(flat_spec < 0.01)] = 7 # saturated points should be 
		
			try:
				dnu = self.cal[specnum]['dnu']
				vel = self.cal[specnum]['vel']
			except KeyError:
				dnu = 0
				vel = 0

			dnu = 0 if np.isnan(dnu) else dnu # some cal files are incomplete
			vel = 0 if np.isnan(vel) else vel # check 04029, 0612 nights

			try:
				tau = self.tau[specnum]
			except KeyError:
				tau = 1.0

			# Interpolate iodine shifted to dnu
			tck_iod = interpolate.splrep(v[::-1]*(1-dnu/3.0e5),self.iodine[::-1], k=2, s=0)

			# Store into data dictionary
			data['unc'][specnum]  = unc[isub]
			data['v'][specnum]    = v[isub]
			data['s'][specnum]    = -1.0*np.log(flat_spec[isub])
			data['t'][specnum]    = self.get_time(meta.time,meta.date)
			data['airmass'][specnum] = self.get_airmass(data['t'][specnum],lon,lat,elevation)
			data['iodine'][specnum]  = -1.0*np.log(interpolate.splev(v[isub],tck_iod,der=0))
			data['dnu'][specnum]     = dnu
			data['vel'][specnum]     = vel
			data['tau'][specnum]     = tau

			print specnum # replace with loading bar
		
		
		# Remove data whose uncertainties are higher than 2*median of all files
		specnums = np.array(data['s'].keys())
		unc_sums = [np.median(data['unc'][key]) for key in specnums]
		inoisy   = specnums[np.where(unc_sums > np.median(unc_sums)*2)[0]]
		for specnum in inoisy:
			del data['unc'][specnum]
			del data['v'][specnum]
			del data['s'][specnum]
			del data['t'][specnum]
			del data['airmass'][specnum]
			del data['iodine'][specnum]
			del data['dnu'][specnum]
			del data['vel'][specnum]

		
		# save things
		self.v    = v[isub] # Save random wavenumber..should be the same
		self.date = meta.date
		self.data = data
		self.specnums = data['v'].keys

	def plot_data(self,fignum=-10):
		"""
		Plot the data
		"""
		plt.figure(fignum)
		for i,key in enumerate(self.specnums):
			plt.plot(self.data['v'][key],self.data['s'][key])
		
		plt.xlabel('Wavenumber (cm$^{-1}$)')
		plt.ylabel('Absorbance')
		plt.title('Date: ' + str(self.date))



if __name__=='__main__':
	datapath = '../inputs/FTS/%s/' %night
	cal_file = '../inputs/PREFITS/test_%s.txt' %night
	tau_file = '../inputs/TAU/tau_%s.txt'   %night
	
	dat = data(datapath,cal_file,v0=15000,vf=15100)
	
	

    
