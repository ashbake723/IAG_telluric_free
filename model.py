# Store all fxns for fitting lines here

"""
This version works assuming rv and tau are not in params

instead just have one rv shift
"""

import astropy.io.fits as fits
import matplotlib.pylab as plt
import numpy as np
import lmfit # new fitting package
from scipy.signal import fftconvolve
from scipy.interpolate import interp1d
from scipy.interpolate import splev, BSpline
import os,sys, glob, scipy
from astropy.time import Time
import scipy.optimize as opt


from hapi import db_begin, db_commit, tableList, select, fetch_by_ids, getColumn, getHelp
from utils import *


plt.ion()


class model():
	"""
	Model stuff - defines:
	minimization function
	model structure
	parameter starting values
	bounds for optimization
	Loading of hitran parameters
	"""
	def __init__(self,
				data,
				input_path,
				v0,
				vf,
				threshhold=1e-28,
				solar_type = "Kurucz",
				tel_fxn = "Lorentzian"
				):
	
		# Define species using
		self.input_path = input_path
		self.fit_threshholds = [1e-25,1e-27] # limit on strengths for lines in hitran units for individual viogt fits
		self.species         = ['H2O','O2']
		self.column_densities= [1.2e23,1e25] # starting values for column densities, tau absorbs rest
		self.c = 2.99792458e5 # km/s
		self.solar_type = solar_type
		self.tel_fxn    = tel_fxn

		# Load hitran parameters
		self.load_hitran(v0,vf,threshhold=threshhold)

		# Take things from data_class
		self.specnums = np.array(data['airmass'].keys())
		self.nspec    = len(self.specnums)

		self.airmass  = np.array([data['airmass'][key] for key in self.specnums])
		self.time     = np.array([data['t'][key]   for key in self.specnums])
		self.dnu      = np.array([data['dnu'][key] for key in self.specnums])
		self.vel      = np.array([data['vel'][key] for key in self.specnums])
		self.tau      = np.array([data['tau'][key] for key in self.specnums])

		self.v        = data['v'][self.specnums[0]] # take last key defined
		self.logv     = np.log(self.v)
		self.npoints  = len(self.v)

		# initialize data arrays
		self.s_arr    = np.zeros((self.nspec,self.npoints))
		self.v_arr    = np.zeros((self.nspec,self.npoints))
		self.iod_arr  = np.zeros((self.nspec,self.npoints))
		self.unc_arr2  = np.zeros((self.nspec,self.npoints))
		for i in range(self.nspec): # fill data arrays for each spectrum
			self.v_arr[i]   = data['v'][self.specnums[i]]
			self.s_arr[i]   = data['s'][self.specnums[i]]
			self.iod_arr[i] = data['iodine'][self.specnums[i]]
			self.unc_arr2[i] = data['unc'][self.specnums[i]]**2

		# Get knot values
		self.get_knots()
		if vf-v0 < 50:
			self.get_coeff0()
		else:
			print "Warning..too wide a wavelength range for spline fit"

	def telluric(self,p, fxn='Lorentzian'):
		'''
		Given the input data (line centers and line strengths) for a molecule, digitize the result into a spectrum of
		absorption cross-section in units of cm^2.
		Parameters
		----------
		p: paramater object from lmfit
			must contain taus, pres, temp, and column density parameters
		fxn : str, optional
			A string describing what function to assume. Choices available are:
			{'Voigt','Lorentzian'}.
		Returns
		-------
		xsec : array_like (nspec, npoints)
			The mean absorption cross-section (in cm^2) per molecule, evaluated at the wavelengths given by input `waves`.
	
		edited by ABaker march 5th
		'''
		# Concatenation 1.0 to tau array for first index (otherwise degenerate with column density parameter)
		#taus = np.array([p['tau_%s'%i] for i in self.specnums])
		taus  = self.tau
		pres = p['pres'].value

		linedata     = self.hitran_params  # shorten hitran param name
		tel = {}          # Initialize tel dictionary (keys are molecules)
		
		## Loop through lines, calculate telluric for each molecule
		if fxn=='Voigt': # needs to be updated! -AB 3/14/19
	 		for i, mol in enumerate(self.species):
				tel[mol] = np.zeros(self.npoints)
				nlines = len(linedata[mol]['nu'])
				for i in range(nlines):
					linecenter   = linedata[mol]['nu'][i]
					linestrength = linedata[mol]['sw'][i]
					gam_lorentz  = linedata[mol]['gamma_air'][i]
					gam_doppler  = 0.07
					shift0       = 0
					
					Ls  = PROFILE_VOIGT(linecenter + shift0,gam_doppler,gam_lorentz,self.v)
					L = p['CD_%i'%mol] * linestrength * Ls[0]/sum(Ls[0])
					
					tel[mol] += L
					
		if fxn=='Lorentzian':
			for i,mol in enumerate(self.species):
				tel[mol] = np.zeros(self.npoints)

				# Pull out per line fit params
				gam_factor  = p['gam_factor_%s' %mol].value
				CD          = p['CD_%s'%mol].value
				nfit        = len(self.istr[mol])
				nweak       = len(self.iwk[mol])
				nu_fit_arr  = np.array([p['nu_fit_%s_%s'%(mol,i)] for i in range(nfit)])
				gam_fit_arr = np.array([p['gam_fit_%s_%s'%(mol,i)] for i in range(nfit)])
				S_fit_arr   = np.array([p['S_fit_%s_%s'%(mol,i)] for i in range(nfit)])
				nu_wk_arr  = np.array([p['nu_wk_%s_%s'%(mol,i)] for i in range(nweak)])
				gam_wk_arr = np.array([p['gam_wk_%s_%s'%(mol,i)] for i in range(nweak)])
				S_wk_arr   = np.array([p['S_wk_%s_%s'%(mol,i)] for i in range(nweak)])

				# Weak lines
				nu       = np.array(linedata[mol]['nu']) + pres * np.array(linedata[mol]['delta_air'])
				gam      = np.array(linedata[mol]['gamma_air']) * gam_factor # better to just have gam_factor
				S        = np.array(linedata[mol]['sw'])        * 10**CD

				# Strong Lines
				nu[self.istr[mol]]    += nu_fit_arr
				gam[self.istr[mol]]   *= gam_fit_arr
				S[self.istr[mol]]     *= S_fit_arr
				nu[self.iwk[mol]]    += nu_wk_arr
				gam[self.iwk[mol]]   *= gam_wk_arr
				S[self.iwk[mol]]     *= S_wk_arr
				
				tel[mol] =  np.sum(lorentzian_profile(np.array([self.v]).T, S, gam, nu),axis=1)

		# Tile tel per molecule to multiply by tau
		xsec = np.zeros((self.nspec,self.npoints))
		for mol in self.species:
			if mol =='H2O':
				xsec += np.tile(tel[mol],(self.nspec,1)) * np.tile(self.airmass,(self.npoints,1)).T * np.tile(taus,(self.npoints,1)).T
			else:
				xsec += np.tile(tel[mol],(self.nspec,1)) * np.tile(self.airmass,(self.npoints,1)).T
			
		self.tel_arr = xsec

		
	def solar(self,p,type='Kurucz',coeff=None):
		"""
		Generate stellar model
		
		type: 'Kurucz' or 'Spline'
		"""
		# make v increasing order
		v    = self.v
		logv = self.logv
				
		# Extract RVs from param array - make first index 0
		#rvshifts = np.array([p['rv_%s'%i] for i in self.specnums])
		rvshift   = p['rv_shift']
		rvshifts  = np.log(1 - (-1*(self.vel+ rvshift))/self.c) 

		# Create all_v array , log space so can add
		all_logv =  np.tile(logv,(self.nspec,1)) + \
				 np.tile(rvshifts,(self.npoints,1)).T + \
				 np.tile(np.log(1 - self.dnu/self.c),(self.npoints,1)).T

		all_v    =  np.tile(v,(self.nspec,1)) * \
				 np.tile(np.exp(rvshifts),(self.npoints,1)).T * \
				 np.tile(1 - self.dnu/self.c,(self.npoints,1)).T
		
		if type=='Kurucz':
			try:
				vstel = self.vstel
				stel = self.stel
				
			except AttributeError:
				# Load stellar but only do this once
				stellar = fits.open(self.input_path + 'STELLAR/irradthuwn.fits')
				stel_v  = stellar[1].data['wavenumber']
				stel_s  = stellar[1].data['irradiance']
				stellar.close()
				
				vlo, vhi  = np.min(v), np.max(v)
				isub      = np.where( (stel_v > vlo - 0.2) & (stel_v < vhi+0.2) )[0]
				stel      = -1*np.log(stel_s[isub]/np.max(stel_s[isub]))

				self.vstel = stel_v[isub]
				self.stel  = -1*np.log(stel_s[isub]/np.max(stel_s[isub]))
				
			# resample spec at each stel
			stell_arr = np.zeros((self.nspec,self.npoints))
			for i in range(len(rvshifts)):
				tck   = scipy.interpolate.splrep(self.vstel,self.stel, s=0)
				stell_arr[i] = scipy.interpolate.splev(all_v[i],tck,der=0,ext=1)
			
			# Return, flip back array if had to flip v order
			self.stell_arr = stell_arr#.T[::-1].T if flipped else stell_arr

		if type == 'Spline':
			try:
				self.knots
			except AttributeError:
				self.get_knots()

			coeffs = np.array([p['coeff_%i'%i].value for i in range(len(self.knots))])

			spl = BSpline(self.knots,coeffs,3)
			
			# Generate stellar spec shifted
			stell_arr   =  spl(all_v)

			self.stell_arr = stell_arr#.T[::-1].T if flipped else stell_arr # no longer need to flip but now vel's go opposite way


 # shape (nspec, npoints)
		
		
	def continuum(self,p):
		"""
		Generate continuum given x,y points at ends
		
		outputs continuum in shape (nspec,npoints)
		"""
		cont_l = p['cont_l'].value
		cont_r = p['cont_r'].value

		continuum_slope = (cont_l - cont_r)/(self.v[0] - self.v[-1])
		continuum       = continuum_slope * (self.v - self.v[0])  + cont_l
		
		self.cont_arr = np.tile(continuum,(self.nspec,1))
	
	
	def load_hitran(self,v0,vf,threshhold=1e-28,hit_path = '../inputs/HITRAN/'):
		"""
		Load hitran data file into dictionary

		Relies on hapi.py from hitran.org

		# To download new hitran data and get table summary: 
		fetch_by_ids('H2O',[1,2],9000,20000,ParameterGroups=['Voigt_Air'])) # first two isotopologues
		fetch_by_ids('O2',[36,37],9000,20000,ParameterGroups=['Voigt_Air'])) # first two isotopologues
	
		tableList()
		describeTable('H2O')
		"""

		# Define parameters to include
		pnames  = ['nu','sw','gamma_air','n_air','delta_air','a']
		hitparams  = {} # dictionary of hitran parameters, keys by molecule then pnames
		istr = {}       # dictionary of indices of lines that are strong enough to fit
		iwk  = {}

		# Load subset of hitran data already downloaded
		self.subhitpath = hit_path + 'v0_%s_vf_%s/' %(int(v0),int(vf))
		# Load subfile if exists else make directory to store file
		if os.path.exists(self.subhitpath):
			# Load pre downloaded sub table
			pass
		else:
			os.system('mkdir ' + self.subhitpath)

			db_begin(hit_path)
			table = tableList()

			for i,molecule in enumerate(self.species):
				if molecule in table:
					subhitname = '%s_v0_%s_vf_%s.par' %(molecule,int(v0),int(vf))
					subhitfile = self.subhitpath + subhitname
					# Save sub table into list 
					select(molecule,
							Conditions=('AND',('between','nu',v0-5.0,vf+5.0),('>=','sw',threshhold)),
							DestinationTableName=subhitname,File=subhitfile)
#					db_commit() # dont think i need this anymore
				else:
					# Download so can work offline ..no idea where this stores shit and i htink it fails so just d/l it yourself
					print 'Downloading Hitran data for ' + molecule
					fetch_by_ids(molecule,[1,2,3],9000,20000,ParameterGroups=['Voigt_Air']) # first two isotopologues

		db_begin(self.subhitpath)
		table = tableList()

		# with table loaded, Loop through molecules again and save parameters into dics
		for i,molecule in enumerate(self.species):
			subhitname = '%s_v0_%s_vf_%s.par' %(molecule,int(v0),int(vf))
			# Get columns need for Voigt parameters
			hitparams[molecule] = {}
			istr[molecule] = {}
			iwk[molecule]  = {}
			for parameter in pnames:
				hitparams[molecule][parameter] = getColumn(subhitname.strip('.par'), parameter)
		
			istr[molecule] = np.where(np.array(hitparams[molecule]['sw']) > self.fit_threshholds[i])[0]
			iwk[molecule]  = np.where((np.array(hitparams[molecule]['sw']) < self.fit_threshholds[i])\
										& (np.array(hitparams[molecule]['sw']) > 0.1*self.fit_threshholds[i]))[0]


		self.hitran_params       = hitparams # save hitran parameters
		self.istr                = istr
		self.iwk                 = iwk
	
	def get_knots(self):
		"""
		Get knot point positions for modeling the star
		
		Have more knot points where stel is strongers 
		how will rv shift + defined based on kurucz affect this? overall shift?
		
		inputs
		------
		v - wavenumber array (not log)
		stel_start  - starting stellar guess/kurucz model in absorbance
		"""
		# Check that v is in order
		v = self.v

		# load Kurucz model (this is repetitive meh)
		try:
			stel     = self.stel	
			vstel    = self.vstel		
		except AttributeError:
			# Load stellar but only do this once
			stellar = fits.open(self.input_path + 'STELLAR/irradthuwn.fits')
			stel_v  = stellar[1].data['wavenumber']
			stel_s  = stellar[1].data['irradiance']
			stellar.close()
			
			vlo, vhi  = np.min(v), np.max(v)
			isub      = np.where( (stel_v > vlo-0.2) & (stel_v < vhi+0.2) )[0]
			vstel = stel_v[isub]
			stel = -1*np.log(stel_s[isub]/np.max(stel_s[isub]))
			self.vstel, self.stel = vstel, stel
		# Where stellar absorption is stronger than 10%, add more knot points
		where_stel_strong = np.where(stel > -1*np.log10(0.9))[0]
		kinks             = np.where(-1*(where_stel_strong - np.roll(where_stel_strong,-1)) > 1)[0]
		kinks 			  = np.concatenate((np.zeros(1,dtype=int),kinks,np.array([len(where_stel_strong)-1])))
	
		# Fill in knots points
		if len(where_stel_strong) == 0:
			knots = np.arange(vstel[0],vstel[-1],0.1)
		else:
			knots  = np.arange(vstel[0],vstel[where_stel_strong[kinks[0]]],0.1)
			for i in range(0,len(kinks)-1):
				knots = np.concatenate(( knots, np.arange(vstel[where_stel_strong[kinks[i]+1]],vstel[where_stel_strong[kinks[i+1]]],0.05) ))
				if i < len(kinks)-2:
					knots = np.concatenate(( knots, np.arange(vstel[where_stel_strong[kinks[i+1]]],vstel[where_stel_strong[kinks[i+1]+1]],0.1) ))

			knots = np.concatenate((knots, np.arange(vstel[where_stel_strong[-1]],vstel[-1],0.1)))

		self.knots = knots

	def get_coeff0(self):
		"""
		Get starting coefficients guess for stellar spline fit
		*finish this
		"""
		v    = self.v
		vstel = self.vstel
		stel = self.stel
		err  = .01*self.stel + 0.001
		# interpolate tel on stel grid which is slightly wider
		#tck   = scipy.interpolate.splrep(v[::-1],self.tel_arr[0][::-1], s=0)
		#tel = scipy.interpolate.splev(vstel,tck,der=0,ext=1)

		knots = self.knots

		def stel_pararr(narr,lo,hi):
			# For each, define name and value
			plo = np.ones(narr)*lo
			phi = np.ones(narr)*hi

			bounds = []
			
			for i in range(len(plo)):
				bounds.append([plo[i],phi[i]])

			return bounds

		def fit_func(coeffs,knots,v,stel,err,S=0,mode='sum'):
			"""
			Take stellar spectra data (each night's average) and fit spline to them
			S : Smoothing term, adds penalty for wiggliness around where telluric lines are
			"""			
			spl = BSpline(knots,coeffs,3)
			y = spl(v)
			#curvy  = y - np.roll(y,5)
			
			if mode =='sum':
				return np.sum(((y - stel)**2)/err)/100 #+ S*np.sum(curvy[bad_tel]**2)
			elif mode == 'get_model':
				return y

		S      = 100
		coeffs = np.ones(len(knots))*0.01
		bounds = stel_pararr(len(knots),0,2)
		#bad_tel = np.where(((tel > 1.5*np.mean(tel)) & (stel < 0.002)))[0]
		out    = opt.minimize(fit_func,coeffs,\
		                        args=(knots,vstel,stel,err,S),\
		    					method="SLSQP",bounds=bounds,options={'maxiter' : 100}) 

		p0        = out['x']
		model     = fit_func(p0,knots,v,stel,err,S=S,mode='get_model')
		self.coeffs0 = p0



if __name__=='__main__':
	v0 = 15000
	vf = 15100
	
	datapath = '../inputs/FTS/20150702/'
	cal_file = './calibration_data/test_20150702.txt'
	
	dat = data(datapath,cal_file,v0=v0,vf=vf)

	mod = model(v0,vf)


	
	
	

	
	

    
