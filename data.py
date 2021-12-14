#!/usr/bin/env python
# coding: utf-8

# ###########specs#########

# In[1]:


import scipy.io as sio 
import matplotlib.pyplot as plt 
import numpy as np
import matplotlib.pyplot as plt
import h5py
import matplotlib.pyplot as plt
import numpy as np
import os
import math
from glob import glob
import pandas as pd
import random as rdm
from tqdm import tqdm
from scipy.fftpack import fft,ifft


###讀檔案近來
data = h5py.File('sim_press_te30_3T_2000_2048.mat')
metabo_names = ['Ala','Asp','Cr','GABA','Glc','Gln','Glu','GPC','PCh','Lac','Ins','NAA','NAAG','Scyllo','Tau']
###########################################Ala
specs = (data['sim_meta']['specs'][0][0])
specs = np.array(data[specs])[0]
Ala_r=[]
Ala_i=[]
for i in range(2048):
    Ala_r.append(specs[i][0])
for i in range(2048):
    Ala_i.append(specs[i][1])
Ala=np.array(Ala_r)+np.array(Ala_i)*1j
############################################Asp
specs = (data['sim_meta']['specs'][1][0])
specs = np.array(data[specs])[0]
Asp_r=[]
Asp_i=[]
for i in range(2048):
    Asp_r.append(specs[i][0])
for i in range(2048):
    Asp_i.append(specs[i][1])
Asp=np.array(Asp_r)+np.array(Asp_i)*1j
###########################################Cr
specs = (data['sim_meta']['specs'][2][0])
specs = np.array(data[specs])[0]
Cr_r=[]
Cr_i=[]
for i in range(2048):
    Cr_r.append(specs[i][0])
for i in range(2048):
    Cr_i.append(specs[i][1])
Cr=np.array(Cr_r)+np.array(Cr_i)*1j
#########################################GABA
specs = (data['sim_meta']['specs'][5][0])
specs = np.array(data[specs])[0]
GABA_r=[]
GABA_i=[]
for i in range(2048):
    GABA_r.append(specs[i][0])
for i in range(2048):
    GABA_i.append(specs[i][1])
GABA=np.array(GABA_r)+np.array(GABA_i)*1j
##########################################Glc
specs = (data['sim_meta']['specs'][16][0])
specs = np.array(data[specs])[0]
Glc_r=[]
Glc_i=[]
for i in range(2048):
    Glc_r.append(specs[i][0])
for i in range(2048):
    Glc_i.append(specs[i][1])
Glc=np.array(Glc_r)+np.array(Glc_i)*1j
#########################################Gln
specs = (data['sim_meta']['specs'][6][0])
specs = np.array(data[specs])[0]
Gln_r=[]
Gln_i=[]
for i in range(2048):
    Gln_r.append(specs[i][0])
for i in range(2048):
    Gln_i.append(specs[i][1])
Gln=np.array(Gln_r)+np.array(Gln_i)*1j
#########################################Glu
specs = (data['sim_meta']['specs'][7][0])
specs = np.array(data[specs])[0]
Glu_r=[]
Glu_i=[]
for i in range(2048):
    Glu_r.append(specs[i][0])
for i in range(2048):
    Glu_i.append(specs[i][1])
Glu=np.array(Glu_r)+np.array(Glu_i)*1j
##########################################GPC
specs = (data['sim_meta']['specs'][18][0])
specs = np.array(data[specs])[0]
GPC_r=[]
GPC_i=[]
for i in range(2048):
    GPC_r.append(specs[i][0])
for i in range(2048):
    GPC_i.append(specs[i][1])
GPC=np.array(GPC_r)+np.array(GPC_i)*1j
##########################################PCh
specs = (data['sim_meta']['specs'][4][0])
specs = np.array(data[specs])[0]
PCh_r=[]
PCh_i=[]
for i in range(2048):
    PCh_r.append(specs[i][0])
for i in range(2048):
    PCh_i.append(specs[i][1])
PCh=np.array(PCh_r)+np.array(PCh_i)*1j
##########################################Lac
specs = (data['sim_meta']['specs'][11][0])
specs = np.array(data[specs])[0]
Lac_r=[]
Lac_i=[]
for i in range(2048):
    Lac_r.append(specs[i][0])
for i in range(2048):
    Lac_i.append(specs[i][1])
Lac=np.array(Lac_r)+np.array(Lac_i)*1j
############################################mI
specs = (data['sim_meta']['specs'][10][0])
specs = np.array(data[specs])[0]
mI_r=[]
mI_i=[]
for i in range(2048):
    mI_r.append(specs[i][0])
for i in range(2048):
    mI_i.append(specs[i][1])
mI=np.array(mI_r)+np.array(mI_i)*1j
###########################################NAA
specs = (data['sim_meta']['specs'][12][0])
specs = np.array(data[specs])[0]
NAA_r=[]
NAA_i=[]
for i in range(2048):
    NAA_r.append(specs[i][0])
for i in range(2048):
    NAA_i.append(specs[i][1])
NAA=np.array(NAA_r)+np.array(NAA_i)*1j
###########################################NAAG
specs = (data['sim_meta']['specs'][17][0])
specs = np.array(data[specs])[0]
NAAG_r=[]
NAAG_i=[]
for i in range(2048):
    NAAG_r.append(specs[i][0])
for i in range(2048):
    NAAG_i.append(specs[i][1])
NAAG=np.array(NAAG_r)+np.array(NAAG_i)*1j
############################################Scyllo
specs = (data['sim_meta']['specs'][13][0])
specs = np.array(data[specs])[0]
Scyllo_r=[]
Scyllo_i=[]
for i in range(2048):
    Scyllo_r.append(specs[i][0])
for i in range(2048):
    Scyllo_i.append(specs[i][1])
Scyllo=np.array(Scyllo_r)+np.array(Scyllo_i)*1j
############################################Tau
specs = (data['sim_meta']['specs'][14][0])
specs = np.array(data[specs])[0]
Tau_r=[]
Tau_i=[]
for i in range(2048):
    Tau_r.append(specs[i][0])
for i in range(2048):
    Tau_i.append(specs[i][1])
Tau=np.array(Tau_r)+np.array(Tau_i)*1j
############################################Pcr
specs = (data['sim_meta']['specs'][3][0])
specs = np.array(data[specs])[0]
Pcr_r=[]
Pcr_i=[]
for i in range(2048):
    Pcr_r.append(specs[i][0])
for i in range(2048):
    Pcr_i.append(specs[i][1])
Pcr=np.array(Pcr_r)+np.array(Pcr_i)*1j

metabo_names = ['Ala','Asp','Cr','GABA','Glc','Gln','Glu','GPC','PCh','Lac','mI','NAA','NAAG','Scyllo','Tau']
decode_basis_god=np.empty(shape=[0, 2048])
decode_basis_god=np.vstack([decode_basis_god,Ala,Asp,Cr,GABA,Glc,Gln,Glu,GPC,PCh,Lac,mI,NAA,NAAG,Scyllo,Tau,Pcr])
ppm = (data['sim_meta']['ppm'][0][0])
ppm = np.array(data[ppm])
ppm=ppm[:,0]
ppm1 = np.linspace(12.4751303, -3.1751303, 1024)
ppm2 = np.linspace(4.5, 0.5, 1024)



def measure_width_hz(ppm, spectrum_data):
    if (spectrum_data.ndim == 2):
        spectrum_data = spectrum_data[:,0]
    tNAA_pos = np.where((ppm>=1.8) & (ppm<=2.2))
    tNAA_crop = spectrum_data[tNAA_pos]
    #tNAA_FWHM_val = max(tNAA_crop)*0.2
    tNAA_FWHM_val = max(tNAA_crop)*(1-0.707)
    tNAA_FWHM = np.where(tNAA_crop > tNAA_FWHM_val)
    tNAA_FWHM_width_ppm = max(ppm[tNAA_pos][tNAA_FWHM])-min(ppm[tNAA_pos][tNAA_FWHM])
    #ppm to Hz, (BW)/(total ppm = BW/B) * width_ppm = B*width_ppm
    tNAA_FWHM_width_hz = round(123.177*tNAA_FWHM_width_ppm, 3)
    return tNAA_FWHM_width_hz
metabo_names = ['Ala','Asp','Cr','GABA','Glc','Gln','Glu','GPC','PCh','Lac','mI','NAA','NAAG','Scyllo','Tau','Pcr']#GSH = PCh?
##brain_metabos_conc_lower = np.array([0.1,1.0,4.5,1.0,1.0,3.0,6.0,0.5,0.5,0.2,4.0,7.5,0.5,0,2.0])
##brain_metabos_conc_upper = np.array([1.5,2.0,10.5,2.0,2.0,6.0,12.5,2.0,2.0,1.0,9.0,17,2.5,0,6.0])
brain_metabos_conc_lower = np.array([0.1,
                                    1.0,
                                     1.5,
                                     1.0,
                                     1.0,
                                     3.0,
                                     6.0,
                                     0.3,
                                     0.2,
                                     0.2,
                                     4.0,
                                     7.5,
                                     0.5,
                                     0.2,
                                     2.0,
                                    3.0]
                                   )
brain_metabos_conc_upper = np.array([1.5,
                                     2.0,
                                     5.0,
                                     2.0,
                                     2.0,
                                     6.0,
                                     12.5,
                                     1.5,
                                     1.0,
                                     1.0,
                                     9.0,
                                     17,
                                     2.5,
                                     0.5,
                                     6.0,
                                    5.5]
                                   )
#var_brain_metabos_conc_set = rdm.uniform(brain_metabos_conc_lower,brain_metabos_conc_upper)
pos = np.where((ppm>=0.5) & (ppm<=4.2))
N_specta = 1

SHUFFLE_var_brain_metabos_conc_set = np.zeros([len(metabo_names), N_specta], dtype=np.float)
for i in range(len(metabo_names)):
    for t in range(N_specta):
        SHUFFLE_var_brain_metabos_conc_set[i,t] = rdm.uniform(brain_metabos_conc_lower[i],brain_metabos_conc_upper[i])
    rdm.shuffle(SHUFFLE_var_brain_metabos_conc_set[i])
AU = np.array([0.72,0.28,0.38,0.05,0.45,0.36,0.36,0.04,0.2,0.11,0.64,0.07,1])
SHUFFLE_var_AU_set = np.zeros([len(AU), N_specta], dtype=np.float)
for i in range(len(AU)):
    for t in range(N_specta):
        SHUFFLE_var_AU_set[i,t] = rdm.uniform(AU[i]*0.9,AU[i]*1.1)
    rdm.shuffle(SHUFFLE_var_AU_set[i])
FWHM=np.array([21.2,19.16,15.9,7.5,29.03,20.53,17.89,5.3,14.02,17.89,33.52,11.85,37.48])
SHUFFLE_var_FWHM_set = np.zeros([len(FWHM), N_specta], dtype=np.float)
for i in range(len(FWHM)):
    for t in range(N_specta):
        SHUFFLE_var_FWHM_set[i,t] = rdm.uniform(FWHM[i]*0.8,FWHM[i]*1.2)
    rdm.shuffle(SHUFFLE_var_FWHM_set[i])


working_dir = os.getcwd()
#basis_filename = 'GAVA_press_te35_3T_test.basis'#16 kinds of metabo
basis_filename = 'press_te30_3t_01a.basis'#15 kinds of metabo
#basis_filename = 'gamma_press_te35_123mhz_149.basis'

basis_path = os.path.join(working_dir, basis_filename)
f_basis = open(basis_path,'r')
BASIS = f_basis.read()
###############Read basis info####################

SPLIT_BASIS = BASIS.split()
indices = [i for i, x in enumerate(SPLIT_BASIS) if x == "METABO"]
NMUSED_indices = [i for i, x in enumerate(SPLIT_BASIS) if x == "$NMUSED"]
conc_indices = [i for i, x in enumerate(SPLIT_BASIS) if x == "CONC"]
ISHIFT_indices = [i for i, x in enumerate(SPLIT_BASIS) if x == "ISHIFT"]
PPMAPP_indices = [i for i, x in enumerate(SPLIT_BASIS) if x == "PPMAPP"]

BADELT = np.array(SPLIT_BASIS[SPLIT_BASIS.index('BADELT')+2].split(","))[0].astype(float)
###測試@########


B = float((SPLIT_BASIS[SPLIT_BASIS.index("HZPPPM")+2]).split(",")[0])
###########################################################################
basis_title = []
for i in range(len(indices)):
    idx = indices[i]
    #print(SPLIT_BASIS[idx:idx+3],i)
    meta_bolite_title = SPLIT_BASIS[idx:idx+3]
    basis_title.append(meta_bolite_title)#拿每種metabo的名字
###########################################################################
data_idx = NMUSED_indices
basis_set = []
for i in range(len(data_idx)):
    if (i <len(data_idx)-1):
        idx = data_idx[i+1]
        meta_bolite_basis = SPLIT_BASIS[idx-4096*2:idx]
        basis_set.append(meta_bolite_basis)#拿每種metabo的basis
        #print("i",i)
    else:
        idx = len(SPLIT_BASIS)
        meta_bolite_basis = SPLIT_BASIS[idx-4096*2:idx]
        #print("i final",i)        
        basis_set.append(meta_bolite_basis)
basis_set = np.array(basis_set)#shape = Nx(sample points*2), N = number of metabo

con_set = []
for i in range(len(conc_indices)):
    idx = SPLIT_BASIS[conc_indices[i]+2]
    idx = idx.split(",")
    con_set.append(idx[0])#拿種每種metabo 的濃度 concetraition
con_set = np.array(con_set).astype(float)


##############################設定一下dump的路徑########################################
dump_folder_name = 'yayayaya1'
dump_folder_path = os.path.join(working_dir,dump_folder_name)

if not os.path.isdir(dump_folder_path):
    os.makedirs(dump_folder_path)
other_parameters_path = os.path.join(dump_folder_path,'other_parameters')
brain_betabo_conc_table_df_path = os.path.join(dump_folder_path,'brain_betabo_conc_table_df')
MM_table_df_path = os.path.join(dump_folder_path,'MM_table_df')
####如過沒有這些目錄 就建起來###
if not os.path.isdir(other_parameters_path):
    os.makedirs(other_parameters_path)
if not os.path.isdir(brain_betabo_conc_table_df_path):
    os.makedirs(brain_betabo_conc_table_df_path)
if not os.path.isdir(MM_table_df_path):
    os.makedirs(MM_table_df_path)

###############################################################    
for steps in tqdm(range(N_specta)):
    var_AU_set = []
    var_brain_metabos_conc_set = []
    var_FWHM_set = []
    #代謝物濃度
    metabo_names = ['Ala','Asp','Cr','GABA','Glc','Gln','Glu','GPC','PCh','Lac','mI','NAA','NAAG','Scyllo','Tau','Pcr']#GSH = PCh?
    var_brain_metabos_conc_set = SHUFFLE_var_brain_metabos_conc_set[:,steps]
    #print('var_brain_metabos_conc_set',var_brain_metabos_conc_set)

    #baseline magnitude
    AU = np.array([0.72,0.28,0.38,0.05,0.45,0.36,0.36,0.04,0.2,0.11,0.64,0.07,1])
    var_AU_set = SHUFFLE_var_AU_set[:,steps]
    #print('var_AU_set',var_AU_set)

    #baseline linewidth
    FWHM=np.array([21.2,19.16,15.9,7.5,29.03,20.53,17.89,5.3,14.02,17.89,33.52,11.85,37.48])
    var_FWHM_set = SHUFFLE_var_FWHM_set[:,steps]
    #print('var_FWHM_set',var_FWHM_set)

    #baseline total magnitude
    #The var_MM_amp config was in paragraph
    var_MM_amp = 0 #2500000
    #boarden parameter
    #var_boarden_t2 = rdm.randrange(200,500,50)
    var_boarden_t2 = rdm.randrange(50,350,30)
    
    #zero order shift parameter
    var_zero_shift = rdm.randint(-5,5)
    #AWGNoise db parameter
    #var_AWGN_db = rdm.randint(5,15)
    var_AWGN_db = rdm.randint(12,40)
    

        ############ Consider concertration################

    brain_betabo_conc_table = {
            "names": metabo_names,
            "conc": var_brain_metabos_conc_set
            }
    brain_betabo_conc_table_df = pd.DataFrame(brain_betabo_conc_table)
    
    #8 = PCh
    
    add_conc_decode_basis = np.zeros(decode_basis_god[0].size).astype(float)
    real_add_conc_decode_basis = np.zeros(decode_basis_god[0].size).astype(float)
    imag_add_conc_decode_basis = np.zeros(decode_basis_god[0].size).astype(float)
    
    #plt.figure()
    unit_decode_basis_set = np.zeros([len(decode_basis_god),len(decode_basis_god[0])],dtype=np.csingle)
    #plt.figure(figsize=(15,10))
    for i in range(len(decode_basis_god)):
        #print('basis_title[i][2]',basis_title[i][2])
        unit_decode_basis = decode_basis_god[i]
        unit_decode_basis_set[i] = unit_decode_basis
        conc_decode_basis = unit_decode_basis*brain_betabo_conc_table_df['conc'][i]
        #plt.figure()
        #plt.title(str(basis_title[i][2]))
        #plt.plot(crop_ppm,conc_decode_basis[pos], label=str(basis_title[i][2]))
        #plt.xlim(4.5,0.5)
        #plt.ylim(-1000,50000000)
        #plt.legend(loc='upper right')
        real_add_conc_decode_basis += conc_decode_basis.real
        imag_add_conc_decode_basis += conc_decode_basis.imag
    #############答案, netabolite only spectrum#############
    add_conc_decode_basis = real_add_conc_decode_basis + 1j*imag_add_conc_decode_basis
    ##################MM Baseline###################
    def g(x, A, μ, σ):
        return (A / (σ * math.sqrt(2 * math.pi))) * np.exp(-(x-μ)**2 / (2*σ**2))
    
    group_name = ['MM09','MM12','MM14','MM16','MM20','MM21','MM23','MM26','MM30','MM31','MM37','MM38','MM40']
    AU = var_AU_set
    freq = ['0.9','1.21','1.38','1.63','2.01','2.09','2.25','2.61','2.96','3.11','3.67','3.8','3.96']
    FWHM = var_FWHM_set
    MM_table = {
            "name": group_name,
            "AU":AU,
            "freq":freq,
            "FWHM":FWHM
            }
    MM_table_df = pd.DataFrame(MM_table)
    
    #plt.figure(figsize=(10,5))
    #var_MM_amp = rdm.randrange(200000*3,400000*3,30000) #2500000
    max_pure_metabo_peak = float(max(add_conc_decode_basis.real))
    
    #var_MM_amp = rdm.randrange(int(max_pure_metabo_peak*0.002), int(max_pure_metabo_peak*0.004), int(max_pure_metabo_peak*0.001))
    var_MM_amp = rdm.uniform(float(max_pure_metabo_peak*0.01), float(max_pure_metabo_peak*0.03))
    #var_MM_amp=5
    add_MM = np.zeros(ppm.size).astype(float)
    add_MM1 = np.zeros(ppm.size).astype(float)
    for i in range(len(MM_table_df)):
        amp = float(MM_table_df['AU'][i])*var_MM_amp#2500000
        fre_ppm = float(MM_table_df['freq'][i])
        w = (float(MM_table_df['FWHM'][i])/B)/2.355#FWHM to sigma
        mmm = g(ppm,amp,fre_ppm,w)
        #print("mmm",mmm.real)
        #plt.figure()
        mmm1=g(ppm,amp,fre_ppm,w)
        '''
        plt.title(str(MM_table_df.loc[i]))
        plt.plot(crop_ppm,mmm.real)
        plt.xlim((4.5,0.5))
        '''
        add_MM += mmm.real
        add_MM1 += mmm1.real
    
    component_sum = add_conc_decode_basis+add_MM#2500000
    #################Boarden linewith###############

    filted_tdata_ori1 = np.fft.fft(component_sum)

    filted_tdata_ori=filted_tdata_ori1[0:1024]
    
    filted_tdata_ori2 = np.fft.ifft(filted_tdata_ori)
    
    
    x_t = np.arange(0,len(filted_tdata_ori))
    # 個案測試
    #var_boarden_t2_list = [300,300,1000,1000]
    
    #var_boarden_t2 = var_boarden_t2_list[steps]
    #var_boarden_t2 = 80
    exp_adop_filt = np.exp(-(x_t/var_boarden_t2))
    filted_tdata = filted_tdata_ori*exp_adop_filt
    #################Zero order correct###############    
    #zero_shift_filter = np.exp(var_zero_shift)    

    ##################Crop period################
    filted_sdata = np.fft.ifft(filted_tdata)
    filted_sdata_basic = np.fft.ifft(filted_tdata_ori)

    

    #pos = np.where((ppm>=0.5) & (ppm<=4.5))
    pos = np.where((ppm>=0.5) & (ppm<=4.2))
    crop_ppm = ppm[pos]
    
    

    crop_filted_sdata = filted_sdata

    pure_metabo_basis = add_conc_decode_basis

    crop_MM = add_MM[pos]
    
    ##############base basis set
    base_basis_set = np.zeros([len(unit_decode_basis_set),len(unit_decode_basis_set[0])])
    base_basis_set = np.array([cont for i,cont in enumerate(unit_decode_basis_set)])
    
    broaden_FWHM_nosnr = measure_width_hz(ppm1, crop_filted_sdata)
    
    
    ##################Add noise##################
    
    crop_filted_sdata_watts = crop_filted_sdata ** 2

    crop_filted_sdata_db = 10 * np.log10(crop_filted_sdata_watts)

    # Additive White Gaussian Noise (AWGN)
    # Set a target SNR
    # 個案測試
    #var_AWGN_db_list = [20,10,20,10]
    #var_AWGN_db = var_AWGN_db_list[steps]
    #steps = 0
    #var_AWGN_db = 12
    
    target_snr_db = var_AWGN_db
    # Calculate signal power and convert to dB 
    sig_avg_watts = np.mean(crop_filted_sdata_watts)

    sig_avg_db = 10 * np.log10(sig_avg_watts)

    # Calculate noise according to [2] then convert to watts
    noise_avg_db = sig_avg_db - target_snr_db

    noise_avg_watts = 10 ** (noise_avg_db / 10)


    mean_noise = 0
    noise_volts = np.random.normal(mean_noise, np.sqrt(noise_avg_watts), len(crop_filted_sdata_watts))

    # Noise up the original signal
    y_volts = crop_filted_sdata + noise_volts

    add_noise_crop_filted_sdata = y_volts

    broaden_FWHM = measure_width_hz(ppm1, add_noise_crop_filted_sdata)
    
    ##################View result##################
    ###################FID#########################
    gt=add_noise_crop_filted_sdata
    
    fid = fft(add_noise_crop_filted_sdata)
    tfid=fid.copy()
    tfid[16:1024]=0
    tspec=np.fft.ifft(tfid)
    train=tspec
    
    if steps < 5:
        #print('pure metabot peak AU:', max(pure_metabo_basis.real))
        #print('var_MM_amp ',var_MM_amp)
        #print('var_MM_amp/max(pure_metabo_basis.real)',var_MM_amp/max(pure_metabo_basis.real))
        
        
        plt.figure(figsize=(15,10))
        plt.subplot(4,1,1)
        plt.title('Pure metabolite spectrum data')
        plt.plot(ppm,add_conc_decode_basis)
        
        
        plt.figure(figsize=(15,10))
        plt.subplot(4,1,1)
        plt.title('Pure metabolite spectrum data')
        plt.plot(ppm,add_MM1)
        
        plt.figure(figsize=(15,10))
        plt.subplot(4,1,1)
        plt.title('Pure metabolite spectrum data')
        plt.plot(ppm,component_sum)
        #plt.xlim((4.5,0.5))
        #plt.xlim((4.2,0.5))
        #plt.ylim(0,0.4*10**8)
        
        
        plt.subplot(4,1,2)
        plt.title('brodeing')
        plt.plot(ppm1,crop_filted_sdata)
        #plt.xlim((4.5,0.5))
        #plt.xlim((4.2,0.5))
        #plt.ylim(0,0.4*10**8)
        #plt.ylim(0,0.4*10**8)
        
        plt.subplot(4,1,3)
        plt.title(f'GT:  SNR ={var_AWGN_db} linewidth:{broaden_FWHM} Hz ')
        plt.plot(ppm1, gt)
        #plt.xlim((4.2,0.5))
        #plt.xlim((4.5,0.5))
        #plt.ylim((0,0.4*10**8))
        plt.subplot(4,1,4)
        plt.title('train ')
        plt.plot(ppm1, train)
        plt.tight_layout()
        
        '''
        sav_filename = f'gen_{steps}.png'
        #print(f"save: f{sav_filename}")
        sav_filename_path = os.path.join(working_dir,'gen_folder', sav_filename)    
        plt.savefig(sav_filename)#Why I can't save it into sub folder
        '''
    
    # NO NORMALIZED _ less noise
    np.savez(os.path.join(dump_folder_path, f'generate_basis_{steps}'), X=train, Y=gt, ppm = ppm1,SNR=var_AWGN_db,lw=broaden_FWHM_nosnr)
   # np.savez(os.path.join(dump_folder_path, f'generate_basis_{steps}'), X=add_noise_crop_filted_sdata, Y=pure_metabo_basis, ppm = crop_ppm)
   # np.savez(os.path.join(other_parameters_path ,f'other_parameters_{steps}'), var_MM_amp = var_MM_amp, var_boarden_t2 = var_boarden_t2, var_zero_shift= var_zero_shift, var_AWGN_db = var_AWGN_db,broaden_FWHM_nosnr = broaden_FWHM_nosnr,broaden_FWHM = broaden_FWHM)
    brain_betabo_conc_table_df.to_pickle(os.path.join(brain_betabo_conc_table_df_path,f'brain_betabo_conc_table_df_{steps}'))
    MM_table_df.to_pickle(os.path.join(MM_table_df_path,f'MM_table_df_{steps}'))
    
    if steps == 0:
        #np.savez(os.path.join(working_dir,'base_basis_set'),data = base_basis_set)
        np.savez(os.path.join(working_dir,'42ppm_base_basis_set'),data = base_basis_set, ppm = crop_ppm)
    #base_basis_set

    print("Current steps: ", steps)

    


# ##################
