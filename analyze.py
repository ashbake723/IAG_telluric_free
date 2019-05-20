# Store all fxns for fitting lines here
import matplotlib.pylab as plt
import numpy as np

import time,sys,glob,os
import cPickle as pickle

from utils import * 
import data as data_class
import model as model_class


output_path = '../outputs/'
params_folder='params/'

nload=10

if len(sys.argv) > 1:
	inight = int(sys.argv[1]) - 1  # subtract 1 bc 0 isnt valid job id
	iv_lim = int(sys.argv[2]) - 1 
else:
	inight = 0  # range: 0-22 (23 nights)
	iv_lim = 200 # range: 0 - 1097 (1098 10 cm-1 ranges)


def read_sup(filename):
	"""
	read in supplementary data
	"""
	nhead = 4 # change this if you change read out!

	f = open(filename,'r')
	lines= f.readlines()
	f.close()

	nlines = len(lines)
	# Save header things
	night = lines[0].split(' ')[-1].strip()
	hit_file   = lines[1].split(' ')[-1].strip()
	threshvals = [lines[2].split(' ')[-2].strip('[').strip(','),lines[2].split(' ')[-1].strip('[').strip(',')]

	sup_dat = {}
	keys = [lines[3].split(' ')[i].strip() for i in range(1,len(lines[3].split(' ')))]
	for key in keys:
		sup_dat[key] = np.zeros(nlines - nhead)
	sup_dat['specnum'] = np.zeros(nlines - nhead,dtype='|S4')

	for i in range(nhead,nlines):
		sup_dat[keys[0]][i-nhead] = '{num:04d}'.format(num=int(lines[i].split()[0]))
		sup_dat[keys[1]][i-nhead] = float(lines[i].split()[1])
		sup_dat[keys[2]][i-nhead] = float(lines[i].split()[2])
		sup_dat[keys[3]][i-nhead] = float(lines[i].split()[3])
		sup_dat[keys[4]][i-nhead] = float(lines[i].split()[4])

	return night, sup_dat, hit_file, threshvals


def get_subdata(dat,v0,vf):
	subdata = {}
	isub = np.where( (dat.v > v0) & (dat.v < vf) )[0]
	for key in dat.data.keys():
		subdata[key] = {}
		for specnum in dat.specnums:
			if type(dat.data[key][dat.specnums[0]]) == np.ndarray:
				subdata[key][specnum] = dat.data[key][specnum][isub]
			else:
				subdata[key] = dat.data[key]

	return subdata

def open_fit(savename):
	"""
	Save fit file
	"""
	dat = {}
	# Save fit parameters
	for night in night_list:
		_, temp_sup,_,_ = read_sup(output_path + 'supplementary/' + savename%(night,int(v0),int(vf))+'.txt' )
		temp  =  pickle.load( open(output_path + 'params/' + savename %(night,int(v0),int(vf))+ '.pickle'   , "rb" ) )
		dat[night] = {}
		dat[night]['p_dic'] = temp
		for key in temp_sup.keys():
			dat[night][key]  = temp_sup[key]

		dat[night]['taus']  = np.array([temp['tau_%s'%i] for i in dat[night]['specnum']])

		print 'loaded %s' %night

	return dat


def plot_fit_trans(v_mod,model,s_mod,night):
	"""
	make some plots
	"""
	plt.figure()
	for i in range(len(mod_all)):
		plt.plot(v_mod,np.exp(-1*s_mod[i]))
		plt.plot(v_mod,np.exp(-1*model[i]),'k--')
		plt.plot(v_mod,np.exp(-1*model[i]) - np.exp(-1*s_mod[i]),'r')

	plt.ylim(0,1)
	plt.xlabel('Wavenumber (cm$^{-1}$)')
	plt.ylabel('Transmission')
	plt.title(night)


def load_sub_model(iv_lim):
	v0, vf = v_lims[iv_lim]
	print 'Loading Nights for range %s to %s cm-1' %(v0,vf)
	savename = 'night_%s_v0_%s_vf_%s'
	filename = output_path + 'params/' + savename %(night,int(v0),int(vf))+ '.pickle'
	if os.path.isfile(filename):
		params_temp  =  pickle.load(filename)
	else:
		print 'missing file for iv_lim %s' %iv_lim

	# Load model structure
	mod = model_class.model(dat,v0,vf)

def save_full_model(night):
	"""
	Save fit file
	"""
	mod_dic = {}
	mod_dic['v']    = v_mod
	mod_dic['iod']  = iod_mod
	mod_dic['tel']  = tel_mod
	mod_dic['cont'] = cont_mod
	mod_dic['stel'] = stel_mod
	mod_dic['s_mod']= s_mod

	# Save fit parameters
	pickle_out = open(output_path + 'stel_final/' + '%s_full_model' %night + '.pickle',"wb")
	pickle.dump(mod_dic, pickle_out)
	pickle_out.close()

def load_full_model(filename):
	"""
	Load pickle file containing arrays of the best fit iod, stel, tel, cont models
	for the specified file
	"""
	mod_dic =  pickle.load(open(filename, "rb" ))
	v      = mod_dic['v_mod']
	iod    = mod_dic['iod_mod']
	tel    = mod_dic['tel_mod']
	cont   = mod_dic['cont_mod']
	stel   = mod_dic['stel_mod']
	data   = mod_dic['s_mod'] # just saved the first ten spectra

	model = iod + tel + cont + stel # sum components (these are all in absorbance)

	td = np.exp(-1*data)  # turn data to transmission
	tm = np.exp(-1*model) # turn model to transmission from absorbance

	# plot
	plt.plot(v,td[0])
	plt.plot(v,tm[0],'k--')
	plt.plot(v,td[0]-tm[0],'r')

	plt.xlabel('Wavenumber (cm$^{-1}$')
	plt.ylabel('Transmission')



if __name__=='__main__':
	night_list, v_lims = define_stuff()
	night = night_list[inight]

	# Load full data
	input_path = '../inputs/'
	dat = data_class.data(input_path,night,v0=9000,vf=20000,nload=nload)

	# Process subset of data
	for iv_lim in range(100,1098):
		v0, vf = v_lims[iv_lim]
		savename = 'night_%s_v0_%s_vf_%s'

		# make subset data
		subdata = get_subdata(dat,v0,vf)

		try:
			p_final = pickle.load( open(output_path + params_folder + night +'/' + savename %(night,int(v0),int(vf))+ '.pickle'   , "rb" ) )
			#p_final = pickle.load( open(output_path + params_folder +'/' + savename %(night,int(v0),int(vf))+ '.pickle'   , "rb" ) )			
		except IOError:
			# save v's still and make model 1.1 to stand out
			redo_file = '../outputs/logs/runagain_%s.log' %night
			if os.path.exists(redo_file):
				f = open(redo_file,'a')
			else:
				f = open(redo_file,'w')
			f.write('%s\t%s\n'%(iv_lim,'0'))
			f.close()
			print 'Skipped %s' %iv_lim
			continue # Skip this iv_lim
	
		# Load model for subdata
		mod = model_class.model(subdata,input_path,v0,vf)

		# Load best fit parameters for sub v range
		mod.continuum(p_final)
		mod.solar(p_final)
		mod.telluric(p_final)

		# extend model and components .. might have to concatenate .. or print to file
		try:
			v_mod    = np.concatenate((v_mod,mod.v))
			iod_mod  = np.concatenate((iod_mod,mod.iod_arr),axis=1)
			tel_mod  = np.concatenate((tel_mod,mod.tel_arr),axis=1)
			cont_mod = np.concatenate((cont_mod,mod.cont_arr),axis=1)
			stel_mod = np.concatenate((stel_mod,mod.stell_arr),axis=1)
			s_mod    = np.concatenate((s_mod,mod.s_arr),axis=1)
		except NameError:
			v_mod, iod_mod, tel_mod, cont_mod, stel_mod, s_mod= mod.v,mod.iod_arr,mod.tel_arr,mod.cont_arr,mod.stell_arr, mod.s_arr
			print 'defining model variables'


	mod_all = iod_mod + tel_mod + cont_mod + stel_mod
	#mod     = model_class.model(dat.data,dat.v0,dat.vf) # full model

	# Save full model
	#plot_fit_trans(v_mod,mod_all,s_mod,night)

	isort = np.argsort(v_mod)
	plt.figure()
	plt.plot(v_mod[isort],np.exp(-1*s_mod[0][isort]))
	plt.plot(v_mod[isort],np.exp(-1*mod_all[0][isort]),'k--')
	plt.plot(v_mod[isort],np.exp(-1*s_mod[0][isort]) - np.exp(-1*mod_all[0][isort]),'r')

	plt.plot(v_mod[isort],np.exp(-1*stel_mod[0][isort]))





	CD_H2O, pres, gamma_H2O, v0s = [],[],[],[]
	for iv_lim in range(0,1098):
		v0, vf = v_lims[iv_lim]
		savename = 'night_%s_v0_%s_vf_%s'

		# make subset data
		#subdata = get_subdata(dat,v0,vf)

		try:
			p_final = pickle.load( open(output_path + params_folder + night +'/' + savename %(night,int(v0),int(vf))+ '.pickle'   , "rb" ) )
		except IOError:
			# save v's still and make model 1.1 to stand out
			print 'skipped %s' %iv_lim
			continue # Skip this iv_lim

		#mod = model_class.model(subdata,v0,vf)

		# Save CD pres and gamma
		CD_H2O.append(p_final['CD_H2O'])
		#CD_O2  = p_final['CD_O2']
		pres.append(p_final['pres'])
		gamma_H2O.append(p_final['gam_factor_H2O'])
		#gamma_O2  = p_final['gam_factor_O2']
		v0s.append((v0+vf)/2.)


		print iv_lim
		


	
