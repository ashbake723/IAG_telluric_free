# Store all fxns for fitting lines here
# 3/21 notes
"""
This version takes out fit on tau and rv and fits first 10 then extends it to whole night

note to self..i think the dictionary for the parameter handling is slowing things down
return to scipy optimize. feed only the necessary parameters to the model but
keep the initial dictionary formatting for calling the bounds
"""

import astropy.io.fits as fits
import matplotlib.pylab as plt
import numpy as np
import scipy.optimize as opt
 
from joblib import Parallel, delayed
import lmfit
import astropy.io.fits as fits
import data as data_class
import model as model_class
from utils import * 
import time,sys, os
import pickle

#################################################
###### STRT USER INPUT ARGUMENTS ##############
#np.where(v_lims == 10610)[0][1]
nload = 100
output_path = '../outputs/'
params_folder='params/'

if len(sys.argv) > 1:
	inight = int(sys.argv[1]) - 1  # subtract 1 bc 0 isnt valid job id
	iv_lim = int(sys.argv[2]) - 1 
else:
	inight = 0   # range: 0-22 (23 nights)
	iv_lim = 160  # range: 0 - 1097 (1098 10 cm-1 ranges) #408,409

######## END USER INPUT ARGUMENTS ###############
#################################################
def check_fit(result):
	"""
	Extract fit params and histogram to see if edges are near bounds
	"""
	p = result.params
	mol = 'H2O'
	nfit = len(mod.istr[mol])
	nweak = len(mod.iwk[mol])
	# line fits
	nu_fit_arr  = np.array([p['nu_fit_%s_%s'%(mol,i)] for i in range(nfit)])
	gam_fit_arr = np.array([p['gam_fit_%s_%s'%(mol,i)] for i in range(nfit)])
	S_fit_arr   = np.array([p['S_fit_%s_%s'%(mol,i)] for i in range(nfit)])
	nu_wk_arr  = np.array([p['nu_wk_%s_%s'%(mol,i)] for i in range(nweak)])
	gam_wk_arr = np.array([p['gam_wk_%s_%s'%(mol,i)] for i in range(nweak)])
	S_wk_arr   = np.array([p['S_wk_%s_%s'%(mol,i)] for i in range(nweak)])

	rvshifts = np.array([p['rv_%s'%i] for i in mod.specnums])
	coeffs   = np.array([p['coeff_%s'%i] for i in range(len(mod.knots))])
	taus     = np.array([p['tau_%s'%i] for i in mod.specnums])
	
	dair = mod.hitran_params['H2O']['delta_air'][mod.istr['H2O']]
	plt.hist(S_fit_arr)
	vals, mins, maxs = [], [], []
	for key in p.keys:
		vals.append(p[key].value)
		mins.append(p[key].min)
		maxs.append(p[key].max)

def set_params(mod,dat,rv=False,cont=True,strong=True,weak=False,coeff=False):
	params = lmfit.Parameters()

	### PRESS PARAMETERS
	params.add('pres',     value=1.0, min=0.2, max=1.1)

	### RV SHIFT
	if rv:
		params.add('rv_shift', value=0.0, min=-0.05, max=0.05)
	else:
		params.add('rv_shift', value=0.0, vary=False)

	### COLUMN DENSITIES ####
	for i,mol in enumerate(mod.species):
		logCD = np.log10(mod.column_densities[i]) # one CD per species so only need tau for water
		if mol =='H2O':
			logCD = dat.CD_H2O
		params.add('CD_%s' %mol, value = logCD, min = logCD - 1.0, max = logCD + 1.0)

	### LINE PARAMS
	for mol in mod.species:
		params.add('gam_factor_%s' %mol, value=0.8, min=0.5, max=1.5)

		for i in range(len(mod.istr[mol])): # Loop through # of strong lines
			if strong==True:
				params.add('nu_fit_%s_%s'%(mol,i),  value = 0.0, min=-5.0e-3, max=5.e-3)
				params.add('gam_fit_%s_%s'%(mol,i), value = 1.0, min= 0.8,    max=1.2)
				params.add('S_fit_%s_%s'%(mol,i),   value = 1.0, min= 0.8,    max=1.2)
			else:
				params.add('nu_fit_%s_%s'%(mol,i),  value = 0.0, vary=False)
				params.add('gam_fit_%s_%s'%(mol,i), value = 1.0, vary=False)
				params.add('S_fit_%s_%s'%(mol,i),   value = 1.0, vary=False)
		for i in range(len(mod.iwk[mol])): # Loop through # of strong lines
			if weak==True:
				params.add('nu_wk_%s_%s'%(mol,i),  value = 0.0,  min=-5.0e-3, max=5.e-3)
				params.add('gam_wk_%s_%s'%(mol,i), value = 1.0 , min= 0.8,    max=1.2)
				params.add('S_wk_%s_%s'%(mol,i),   value = 1.0 , min= 0.8,    max=1.2)
			else:
				params.add('nu_wk_%s_%s'%(mol,i),  value = 0.0, vary=False)
				params.add('gam_wk_%s_%s'%(mol,i), value = 1.0, vary=False)
				params.add('S_wk_%s_%s'%(mol,i),   value = 1.0, vary=False)

	### CONTINUUM ###
	if cont == True:
		params.add('cont_l', value=0, min=-0.1, max=0.1)
		params.add('cont_r', value=0, min=-0.1, max=0.1)
	else:
		params.add('cont_l', value=0, vary=False) 
		params.add('cont_r', value=0, vary=False) 


	#### SPLINE COEFFICIENTS FOR STELLAR ####
	for i,knot in enumerate(mod.knots): # have specnum labels
		if coeff:
			params.add('coeff_%s' %i,value=mod.coeffs0[i], min=mod.coeffs0[i]*.8, max=1.2*mod.coeffs0[i])
		else:
			params.add('coeff_%s' %i,value=mod.coeffs0[i],vary=False)

	return params


def redo_params(params,rv=False,cont=False,strong=False,weak=False,CD=False,coeff=False,gamma=False):
	"""
	Take params and 
	"""
	params_new = lmfit.Parameters()

	params_new.add('pres',     value=params['pres'].value, vary=False)

	### RV SHIFT
	if rv:
		params_new.add('rv_shift', value=params['rv_shift'], min=-0.05, max=0.05)
	else:
		params_new.add('rv_shift', value=0.0, vary=False)

	### COLUMN DENSITIES ####
	for i,mol in enumerate(mod.species):
		if CD==True:
			params_new.add('CD_%s' %mol, value = params['CD_%s' %mol].value, min =  params['CD_%s' %mol] - 1.0, max = params['CD_%s' %mol] + 1.0)
		else:
			params_new.add('CD_%s' %mol, value = params['CD_%s' %mol].value, vary=False)

	### LINE PARAMETERS
	# Split up into two bc otherwise takes too long
	for mol in mod.species:
		if gamma:
			params_new.add('gam_factor_%s' %mol, value=params['gam_factor_%s' %mol].value, min=0.5, max=1.5)
		else:
			params_new.add('gam_factor_%s' %mol, value=params['gam_factor_%s' %mol].value, vary=False)#min=0.5, max=.9)

		for i in range(len(mod.istr[mol])): # Loop through # of strong lines
			nu  = params['nu_fit_%s_%s'%(mol,i)].value
			S   = params['S_fit_%s_%s'%(mol,i)].value
			gam = params['gam_fit_%s_%s'%(mol,i)].value
			if strong==True:
				params_new.add('nu_fit_%s_%s'%(mol,i),  value = nu,  min=-0.01,   max=0.01)
				params_new.add('gam_fit_%s_%s'%(mol,i), value = gam, min= gam*0.8,  max=gam*1.2)
				params_new.add('S_fit_%s_%s'%(mol,i),   value = S,   min= S*0.8,    max=S*1.2)
			else:
				params_new.add('nu_fit_%s_%s'%(mol,i),  value = nu, vary=False)
				params_new.add('gam_fit_%s_%s'%(mol,i), value = gam, vary=False)
				params_new.add('S_fit_%s_%s'%(mol,i),   value = S, vary=False)
		for i in range(len(mod.iwk[mol])): # Loop through # of strong lines
			nu     = params['nu_wk_%s_%s'%(mol,i)].value
			S      = params['S_wk_%s_%s'%(mol,i)].value
			gam    = params['gam_wk_%s_%s'%(mol,i)].value
			if weak==True:
				params_new.add('nu_wk_%s_%s'%(mol,i),  value = nu,  min= -0.01,   max=0.01)
				params_new.add('gam_wk_%s_%s'%(mol,i), value = gam, min= gam*0.8,  max=gam*1.2)
				params_new.add('S_wk_%s_%s'%(mol,i),   value = S,   min= S*0.8,    max=S*1.2)
			else:
				params_new.add('nu_wk_%s_%s'%(mol,i),  value = nu,  vary=False)
				params_new.add('gam_wk_%s_%s'%(mol,i), value = gam, vary=False)
				params_new.add('S_wk_%s_%s'%(mol,i),   value = S,   vary=False)

	### CONTINUUM ###
	contl = params['cont_l'].value
	contr = params['cont_r'].value
	if cont == True:
		params_new.add('cont_l', value=contl, min=contl-0.05, max=contl+0.05)
		params_new.add('cont_r', value=contr, min=contr-0.05, max=contr+0.05)
	else:
		params_new.add('cont_l', value=contl, vary=False) 
		params_new.add('cont_r', value=contr, vary=False) 

	#### SPLINE COEFFICIENTS FOR STELLAR ####
	for i,knot in enumerate(mod.knots): # have specnum labels
		if coeff:
			if params['coeff_%s' %i].value > 1e-5:
				#params_new.add('coeff_%s' %i, value=params['coeff_%s' %i].value, min=params['coeff_%s' %i] * 0.7, max=params['coeff_%s' %i] * 1.3)
				params_new.add('coeff_%s' %i, value=params['coeff_%s' %i].value, min=params['coeff_%s' %i] * 0, max=params['coeff_%s' %i]*0+0.5)
			else:
				params_new.add('coeff_%s' %i, value=params['coeff_%s' %i].value, min=1e-15, max=5e-5)

		else: 
			params_new.add('coeff_%s' %i, value=params['coeff_%s' %i].value,vary=False)

	return params_new


def minimizing_function(params,mod,s,stel_type='Kurucz'):
	"""
	define function
	"""
	# Reset param array based on pararr
	mod.continuum(params)
	
	# Define solar
	mod.solar(params,type=stel_type)

	# Define telluric
	mod.telluric(params)

	# define model
	model = mod.iod_arr + mod.tel_arr + mod.stell_arr + mod.cont_arr 

	# Minimization
	return (model - mod.s_arr)/np.sqrt(mod.unc_arr2)


#corner.corner(result.flatchain, labels=restul.var_names, truths=list(result.params.valuesdict().values()))

def run_fit(fxn,p,mod,stel_type='Kurucz',maxfunevals=300):
	"""
	Inputs 
	------
	fxn - function to minimize
	p   - parameters to optimize

	Outputs
	------
	result - return object from lmfit
	model  - 

	"""
	m = lmfit.Minimizer(fxn, p, 
			fcn_args=(mod,0,stel_type),calc_covar=False)

	result = m.minimize(method='ampgo',local='SLSQP',maxfunevals=maxfunevals,glbtol=1e-7)#,eps2=0.01,eps1=0.001,disp=True)	
	
	#result = m.minimize(method='emcee',nwalkers=700,progress=True)
	
	#lmfit.report_fit(result)
	print 'reduced chi 2 = %s' %result.redchi

	# Redefine final model
	mod.continuum(result.params)
	mod.solar(result.params,type=stel_type)
	mod.telluric(result.params)
	model = mod.stell_arr + mod.tel_arr + mod.cont_arr + mod.iod_arr

	return result, model


def plot_fit(mod,model,night,savename):
	"""
	make some plots
	"""
	plt.figure()
	for i in range(mod.nspec):
		plt.plot(mod.v,mod.s_arr[i])
		plt.plot(mod.v,model[i],'k--')
		plt.plot(mod.v,model[i] - mod.s_arr[i],'r')

	plt.xlabel('Wavenumber (cm$^{-1}$)')
	plt.ylabel('Log Normalized Flux')
	plt.title(night)

	# Plot where weak lines
	for mol in mod.species:
		nus = np.array(mod.hitran_params[mol]['nu'])
		plt.plot(nus[mod.istr[mol]], np.zeros(len(mod.istr[mol])) - 0.03,'s')
		plt.plot(nus[mod.iwk[mol]], np.zeros(len(mod.iwk[mol])) - 0.03,'*')

	plt.savefig(output_path + 'plots/%s.pdf' %savename)

def plot_fit_trans(mod,model,night,savename):
	"""
	make some plots
	"""
	plt.figure()
	for i in range(mod.nspec):
		plt.plot(mod.v,np.exp(-1*mod.s_arr[i]))
		plt.plot(mod.v,np.exp(-1*model[i]),'k--')
		plt.plot(mod.v,np.exp(-1*model[i]) - np.exp(-1*mod.s_arr[i]),'r')

	plt.xlabel('Wavenumber (cm$^{-1}$)')
	plt.ylabel('Log Normalized Flux')
	plt.title(night)

	# Plot where weak lines
	for mol in mod.species:
		nus = np.array(mod.hitran_params[mol]['nu'])
		plt.plot(nus[mod.istr[mol]], np.ones(len(mod.istr[mol])) + 0.03,'s')
		plt.plot(nus[mod.iwk[mol]], np.ones(len(mod.iwk[mol])) + 0.03,'*')

	plt.savefig(output_path + 'plots/%s.png' %savename)

def save_fit(result,savename):
	"""
	Save fit file
	"""
	# Save fit parameters
	pickle_out = open(output_path + 'params/' + savename + '.pickle',"wb")
	pickle.dump(result.params, pickle_out)
	pickle_out.close()

def save_data_txt(p,mod,savename):
	"""
	Save stuff from data
	p: final parameters
	mod, dat are loaded model and data classes
	"""
	taus = mod.tau #np.array([p['tau_%s'%i] for i in mod.specnums])
	vels = mod.vel #np.array([p['rv_%s'%i] for i in mod.specnums])

	f =  open(output_path + 'supplementary/%s.txt' %savename,'w')

	f.write('# supplementary data outputs for night '+ night + '\n')
	f.write('# hitran sub data file: %s\n' %mod.subhitfile)
	f.write('# threshhold values: %s\n' %mod.fit_threshholds)

	f.write('# specnum time airmass taus vels\n')
	for i,specnum in enumerate(mod.specnums):
		f.write('%s %s %s %s %s\n' %(specnum, mod.time[i], mod.airmass[i], taus[i], vels[i]))

	f.close()

def save_data(result,savename):
 	"""
	Save stuff from data
	assumes final param and fit, v0, vf, iv_lim, and mod from global variables
	"""
	p = result.params
	taus = mod.tau
	vels = mod.vel

	tbhdu = fits.BinTableHDU.from_columns(
       [fits.Column(name='specnum', format='4A', array=np.array(mod.specnums)),
        fits.Column(name='time',format='D',array=np.array(mod.time)),
        fits.Column(name='airmass', format='E', array=np.array(mod.airmass)),
        fits.Column(name='taus', format='E', array=np.array(taus)),
        fits.Column(name='vels', format='E', array=np.array(vels)) ])
	
	# Header info
	hdr = fits.Header()
	hdr['NIGHT']     = night
	for i in range(len(mod.species)):
		hdr['T_%s' %mod.species[i]] = mod.fit_threshholds[i]
		hdr['T_%s' %mod.species[i]] = mod.fit_threshholds[i]
	hdr['REDCHI']   = round(result.redchi,5)
	hdr['HITFILE']  = mod.subhitpath
	hdr['V0']  = v0
	hdr['VF']  = vf
	hdr['IV_LIM']  = iv_lim
	hdr['NSPEC']  = mod.nspec
	primary_hdu = fits.PrimaryHDU(header=hdr)

	hdu = fits.HDUList([primary_hdu, tbhdu])
	filename = output_path + 'supplementary/%s.fits' %savename
	if os.path.exists(filename):
		os.system('rm %s' % filename)
		hdu.writeto(filename)
	else:
		hdu.writeto(filename)

def iterate_params(p_start):
	#refnight = night#'20150702'
	result0,model0 = run_fit(minimizing_function,p_start,mod,maxfunevals=400,stel_type = 'Kurucz')

	params_new      = redo_params(result0.params,rv=False,cont=True,strong=False,weak=False,CD=True,coeff=False,gamma=True)
	result1, model1 = run_fit(minimizing_function, params_new, mod,maxfunevals=400,stel_type = 'Kurucz')

	if result1.redchi < result0.redchi:
		result = result1
	else:
		result = result0

	# Run 2 strong lines
	if len(mod.istr['H2O']) + len(mod.istr['O2']) > 0:
		params_new2 = redo_params(result.params,rv=False,cont=False,strong=True,weak=False,CD=False)
		result2, model2 = run_fit(minimizing_function, params_new2,mod,maxfunevals=300,stel_type = 'Kurucz')			
		if result2.redchi < result.redchi:
			result = result2
		
		params_new2 = redo_params(result.params,rv=False,cont=True,strong=True,weak=False,CD=False)
		result2, model2 = run_fit(minimizing_function, params_new2,mod,stel_type = 'Kurucz')			
		if result2.redchi < result.redchi:
			result = result2


	# Run 3 weak lines
	if len(mod.iwk['H2O']) + len(mod.iwk['O2']) > 0:
		params_new3 = redo_params(result.params,rv=False,cont=False,strong=False,weak=True,CD=False,coeff=False)
		result3, model3 = run_fit(minimizing_function, params_new3,mod,maxfunevals=500)
		if result3.redchi < result.redchi:
			result = result3

	params_new2 = redo_params(result.params,rv=False,cont=True,strong=False,weak=False,CD=False)
	result2, model2 = run_fit(minimizing_function, params_new2,mod,stel_type = 'Kurucz')			
	if result2.redchi < result.redchi:
		result = result2

	# both stron and weak
	if len(mod.iwk['H2O']) + len(mod.iwk['O2']) > 0:
		params_new3 = redo_params(result.params,rv=False,cont=False,strong=True,weak=True,CD=False,coeff=False)
		result3, model3 = run_fit(minimizing_function, params_new3,mod,maxfunevals=400)
		if result3.redchi < result.redchi:
			result = result3

	return result

def iterate_params_spline(p_start):
	#refnight = night#'20150702'
	result0,model0 = run_fit(minimizing_function,p_start,mod,maxfunevals=400,stel_type = 'Kurucz')

	params_new      = redo_params(result0.params,rv=False,cont=True,strong=False,weak=False,CD=True,coeff=False,gamma=True)
	result1, model1 = run_fit(minimizing_function, params_new, mod,maxfunevals=400,stel_type = 'Kurucz')

	if result1.redchi < result0.redchi:
		result = result1
	else:
		result = result0

	# Run 2 strong lines
	if len(mod.istr['H2O']) + len(mod.istr['O2']) > 0:
		params_new2 = redo_params(result.params,rv=False,cont=True,strong=True,weak=False,CD=False)
		result2, model2 = run_fit(minimizing_function, params_new2,mod,stel_type = 'Kurucz')
		if (result2.redchi > 50) & (result2.redchi < result.redchi) : # fit longer if chi2 sux
			params_new2 = redo_params(result2.params,rv=False,cont=True,strong=True,weak=False,CD=False)
			result2, model2 = run_fit(minimizing_function, params_new2,mod,maxfunevals=800,stel_type = 'Kurucz')
		elif (result2.redchi > 50): # run without cont varying if got worse
			params_new2 = redo_params(result.params,rv=False,cont=False,strong=True,weak=False,CD=False)
			result2, model2 = run_fit(minimizing_function, params_new2,mod,stel_type = 'Kurucz')			
		if result2.redchi < result.redchi:
			result = result2
		else:
			params_new2 = redo_params(result.params,rv=False,cont=False,strong=True,weak=False,CD=False)
			result2, model2 = run_fit(minimizing_function, params_new2,mod,maxfunevals=300,stel_type = 'Spline')			
			if result2.redchi < result.redchi:
				result = result2
				params_new2 = redo_params(result.params,rv=False,cont=True,strong=True,weak=False,CD=False)
				result2, model2 = run_fit(minimizing_function, params_new2,mod,stel_type = 'Kurucz')			
				if result2.redchi < result.redchi:
					result = result2

	params_new2 = redo_params(result.params,rv=False,cont=True,strong=False,weak=False,CD=False)
	result2, model2 = run_fit(minimizing_function, params_new2,mod,stel_type = 'Kurucz')			
	if result2.redchi < result.redchi:
		result = result2

	# Run 3 weak lines
	if len(mod.iwk['H2O']) + len(mod.iwk['O2']) > 0:
		params_new3 = redo_params(result.params,rv=False,cont=False,strong=False,weak=True,CD=False,coeff=False)
		result3, model3 = run_fit(minimizing_function, params_new3,mod,maxfunevals=500)
		if result3.redchi < result.redchi:
			result = result3

	# Switch to Spline
	params_new4 = redo_params(result.params,rv=False,cont=False,strong=False,weak=False,CD=False,coeff=True)
	result4, model4 = run_fit(minimizing_function, params_new4,mod,maxfunevals=500,stel_type='Spline')

	if result4.redchi < result.redchi:
		result = result4

	# Refit varying telluric and stellar, reduce range for strong lines
	params_new5 = redo_params(result.params,rv=False,cont=False,strong=False,weak=False,CD=False,coeff=True)
	for mol in mod.species:
		for i in range(len(mod.istr[mol])): # Loop through # of strong lines
			params_new5.add('nu_fit_%s_%s'%(mol,i),  value = params_new5['nu_fit_%s_%s'%(mol,i)].value,  vary=False)
			params_new5.add('gam_fit_%s_%s'%(mol,i), value = params_new5['gam_fit_%s_%s'%(mol,i)].value, min= 0.9*params_new5['gam_fit_%s_%s'%(mol,i)].value,    max=1.1*params_new5['gam_fit_%s_%s'%(mol,i)].value)
			params_new5.add('S_fit_%s_%s'%(mol,i),   value = params_new5['S_fit_%s_%s'%(mol,i)].value,   min= 0.9*params_new5['S_fit_%s_%s'%(mol,i)].value,    max=1.1*params_new5['S_fit_%s_%s'%(mol,i)].value)
		for i in range(len(mod.iwk[mol])): # Loop through # of strong lines
			params_new5.add('nu_wk_%s_%s'%(mol,i),  value = params_new5['nu_wk_%s_%s'%(mol,i)].value,  vary=False)
			params_new5.add('gam_wk_%s_%s'%(mol,i), value = params_new5['gam_wk_%s_%s'%(mol,i)].value, min= 0.9*params_new5['gam_wk_%s_%s'%(mol,i)].value,  max=1.1*params_new5['gam_wk_%s_%s'%(mol,i)].value)
			params_new5.add('S_wk_%s_%s'%(mol,i),   value = params_new5['S_wk_%s_%s'%(mol,i)].value,   min= 0.9*params_new5['S_wk_%s_%s'%(mol,i)].value,    max=1.1*params_new5['S_wk_%s_%s'%(mol,i)].value)

	result5, model5 = run_fit(minimizing_function, params_new5,mod,maxfunevals=600,stel_type='Spline')
	
	if result5.redchi < result.redchi:
		result = result5

	return result, model5

if __name__=='__main__':
	night_list, v_lims = define_stuff()

	# Load data, pick data path and calibration file
	night  = night_list[inight]
	v0, vf = v_lims[iv_lim]
	print 'Starting Fits for Night %s in range %s to %s cm-1' %(night,v0,vf)

	savename = 'night_%s_v0_%s_vf_%s' %(night,int(v0),int(vf))

	# Name paths to input files
	input_path = '../inputs/'

	# Load Data and Model
	dat = data_class.data(input_path,night,v0=v0,vf=vf,nload=10)
	mod = model_class.model(dat.data,input_path,v0,vf)

	####################
	## FIT SEQUENCE ##
	####################
	refnight=night
	try:
		p_old        = pickle.load( open(output_path + params_folder + refnight +'/' + 'night_%s_v0_%s_vf_%s' %(refnight,int(v0),int(vf))+ '.pickle'   , "rb" ) )
		p_start      = redo_params(p_old,rv=True,cont=False,strong=False,weak=False,CD=True,coeff=False,gamma=True)
	except IOError:
	# save v's still and make model 1.1 to stand out
		p_start        = set_params(mod,dat,cont=False,rv=True,strong=False,weak=False,coeff=False)
	except KeyError:
	# save v's still and make model 1.1 to stand out
		p_start        = set_params(mod,dat,cont=False,rv=True,strong=False,weak=False,coeff=False)

	niter= 0
	result = iterate_params(p_start)
	while (result.redchi > 1.5) & (niter < 2):
		result = iterate_params(result.params)
		niter += 1

	save_fit(result,savename)
	save_data(result,savename)
	#plot_fit_trans(mod,model,night,savename)

	logfile = '../outputs/logs/%s.log' %night
	if os.path.exists(logfile):
		f = open(logfile,'a')
	else:
		f = open(logfile,'w')
	
	f.write('%s\t%s\n'%(iv_lim,result.redchi))
	f.close()


	# ReLoad Data and Model for all spectra
#	dat = data_class.data(datapath,cal_file,tau_file,v0=v0,vf=vf,nload=None)
#	mod = model_class.model(dat.data,v0,vf)

#	result6, model6 = run_fit(minimizing_function, result5.params,mod,maxfunevals=400,stel_type='Spline')


	


