# -*- coding: utf-8 -*-
"""
a Script to download HelioMAS output and compute the change in solar wind 
structure from CR ot CR


Created on Mon Dec 20 11:48:49 2021

@author: mathewjowens
"""

import httplib2
import urllib
import os
from pyhdf.SD import SD, SDC  
import numpy as np
import astropy.units as u
import datetime
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
from scipy.stats import pearsonr, spearmanr

import helio_time as htime
import ReadICMElist_CaneRichardson as ICMElist

plt.rcParams.update({'font.size': 16})
downloadnow = False #flag to determine whether HelioMAS data needs to be obtained. Set to False after first use

datadir = 'D:\\Dropbox\\python_repos\\SolarWindVariability\data\\'
heliomasdir = datadir + 'HelioMAS\\'
#heliomasdir = os.environ['DBOX'] + 'Data\\HelioMAS\\'

# <codecell> functions

def get_helioMAS_output(cr=np.NaN, observatory='', runtype='', runnumber='', masres=''):
    """
    A function to grab the  Vr and Br boundary conditions from MHDweb. An order
    of preference for observatories is given in the function. Checks first if
    the data already exists in the HUXt boundary condition folder

    Parameters
    ----------
    cr : INT
        Carrington rotation number 
    observatory : STRING
        Name of preferred observatory (e.g., 'hmi','mdi','solis',
        'gong','mwo','wso','kpo'). Empty if no preference and automatically selected 
    runtype : STRING
        Name of preferred MAS run type (e.g., 'mas','mast','masp').
        Empty if no preference and automatically selected 
    runnumber : STRING
        Name of preferred MAS run number (e.g., '0101','0201').
        Empty if no preference and automatically selected    

    Returns
    -------
    flag : INT
        1 = successful download. 0 = files exist, -1 = no file found.

    """
    
    assert(np.isnan(cr) == False)
    
    # The order of preference for different MAS run results
    if not masres:
        masres_order = ['high','medium']
    else:
        masres_order = [str(masres)]
           
    if not observatory:
        observatories_order = ['hmi', 'mdi', 'solis', 'gong', 'mwo', 'wso', 'kpo']
    else:
        observatories_order = [str(observatory)]
               
    if not runtype:
        runtype_order = ['masp', 'mas', 'mast']
    else:
        runtype_order = [str(runtype)]
           
    if not runnumber:
        runnumber_order = ['0201', '0101']
    else:
        runnumber_order = [str(runnumber)]
           
    # Get the HUXt boundary condition directory
    #dirs = H._setup_dirs_()
    #_boundary_dir_ = dirs['boundary_conditions'] 
      
    # Example URL: http://www.predsci.com/data/runs/cr2010-medium/mdi_mas_mas_std_0101/helio/br_r0.hdf
    # https://shadow.predsci.com/data/runs/cr2000-medium/mdi_mas_mas_std_0101/helio/vr002.hdf
    heliomas_url_front = 'https://shadow.predsci.com/data/runs/cr'
    heliomas_url_end = '002.hdf'
    
    vrfilename = 'vr002.hdf'
    brfilename = 'br002.hdf'
    inputfilename = 'br_r0.hdf'


    # Search MHDweb for a HelioMAS run, in order of preference
    h = httplib2.Http(disable_ssl_certificate_validation=True)
    foundfile = False
    for res in masres_order:
        for masob in observatories_order:
            for masrun in runtype_order:
                for masnum in runnumber_order:
                    urlbase = (heliomas_url_front + str(int(cr)) + '-' + 
                               res + '/' + masob + '_' +
                               masrun + '_mas_std_' + masnum + '/helio/')
                    url = urlbase + 'br' + heliomas_url_end
                    
                    coronal_urlbase = (heliomas_url_front + str(int(cr)) + '-' + 
                               res + '/' + masob + '_' +
                               masrun + '_mas_std_' + masnum + '/corona/')

                    # See if this br file exists
                    resp = h.request(url, 'HEAD')
                    if int(resp[0]['status']) < 400:
                        foundfile = True
                                            
                    # Exit all the loops - clumsy, but works
                    if foundfile: 
                        break
                if foundfile:
                    break
            if foundfile:
                break
        if foundfile:
            break
        
    if foundfile == False:
        print('No data available for given CR and observatory preferences')
        return -1
    
    # Download teh vr and br files
    import ssl
    ssl._create_default_https_context = ssl._create_unverified_context
    print('Downloading from: ', urlbase)
    urllib.request.urlretrieve(urlbase + 'br' + heliomas_url_end,
                               os.path.join(brfilename))
    urllib.request.urlretrieve(urlbase + 'vr' + heliomas_url_end,
                               os.path.join(vrfilename))
    
    #also grab the input Br
    urllib.request.urlretrieve(coronal_urlbase + 'br_r0.hdf',
                               os.path.join(inputfilename))
        
    return 1

#Data reader functions
def LoadSSN(filepath='null'):
    #(dowload from http://www.sidc.be/silso/DATA/SN_m_tot_V2.0.csv)
    if filepath == 'null':
        filepath= os.environ['DBOX'] + 'Data\\SN_m_tot_V2.0.txt'
        
    col_specification =[(0, 4), (5, 7), (8,16),(17,23),(24,29),(30,35)]
    ssn_df=pd.read_fwf(filepath, colspecs=col_specification,header=None)
    dfdt=np.empty_like(ssn_df[0],dtype=datetime)
    for i in range(0,len(ssn_df)):
        dfdt[i] = datetime(int(ssn_df[0][i]),int(ssn_df[1][i]),15)
    #replace the index with the datetime objects
    ssn_df['datetime']=dfdt
    ssn_df['ssn']=ssn_df[3]
    ssn_df['mjd'] = htime.datetime2mjd(dfdt)
    #delete the unwanted columns
    ssn_df.drop(0,axis=1,inplace=True)
    ssn_df.drop(1,axis=1,inplace=True)
    ssn_df.drop(2,axis=1,inplace=True)
    ssn_df.drop(3,axis=1,inplace=True)
    ssn_df.drop(4,axis=1,inplace=True)
    ssn_df.drop(5,axis=1,inplace=True)
    
    #add the 13-month running smooth
    window = 13*30
    temp = ssn_df.rolling(str(window)+'D', on='datetime').mean()
    ssn_df['smooth'] = np.interp(ssn_df['mjd'],temp['mjd'],temp['ssn'],
                                              left =np.nan, right =np.nan)
    
    #add in a solar activity index, which normalises the cycle magnitude
    #approx solar cycle length, in months
    nwindow = int(11*12)
    
    #find maximum value in a 1-solar cycle bin centred on current time
    ssn_df['rollingmax'] = ssn_df.rolling(nwindow, center = True).max()['smooth']
    
    #fill the max value at the end of the series
    fillval = ssn_df['rollingmax'].dropna().values[-1]
    ssn_df['rollingmax'] = ssn_df['rollingmax'].fillna(fillval) 
    
    #create a Solar Activity Index, as SSN normalised to the max smoothed value in
    #1-sc window centred on current tim
    ssn_df['sai'] = ssn_df['smooth']/ssn_df['rollingmax']
    
    return ssn_df

def read_HelioMAS(filepath):
    """
    A function to read in the HelioMAS data cubes

    Parameters
    ----------
    directory_path : INT
        Carrington rotation number

    Returns
    -------
    MAS_vr : NP ARRAY (NDIM = 2)
        Solar wind speed at 30rS, in km/s
    MAS_vr_Xa : NP ARRAY (NDIM = 1)
        Carrington longitude of Vr map, in rad
    MAS_vr_Xm : NP ARRAY (NDIM = 1)
        Latitude of Vr as angle down from N pole, in rad
    MAS_vr_Xr : NP ARRAY (NDIM = 1)
        Radial distance of Vr, in solar radii

    """
    
    assert os.path.exists(filepath)
    
    file = SD(filepath, SDC.READ)
        
    sds_obj = file.select('fakeDim0')  # select sds
    MAS_vr_Xa = sds_obj.get()  # get sds data
    sds_obj = file.select('fakeDim1')  # select sds
    MAS_vr_Xm = sds_obj.get()  # get sds data
    sds_obj = file.select('fakeDim2')  # select sds
    MAS_vr_Xr = sds_obj.get()  # get sds data
    sds_obj = file.select('Data-Set-2')  # select sds
    MAS_vr = sds_obj.get()  # get sds data
    
    # # Convert from model to physicsal units
    # MAS_vr = MAS_vr*481.0 * u.km/u.s
    MAS_vr_Xa = MAS_vr_Xa * u.rad
    MAS_vr_Xm = MAS_vr_Xm * u.rad
    MAS_vr_Xr = MAS_vr_Xr * u.solRad
    
    
    return MAS_vr, MAS_vr_Xa, MAS_vr_Xm, MAS_vr_Xr


# <codecell> load and process the OMNI data
#data_dir = os.environ['DBOX'] + 'Data_hdf5\\'

CRstart = int(np.ceil(htime.mjd2crnum(htime.datetime2mjd(datetime(1994,11,21))))) # 1625
CRstop = 2232

omni_1hour = pd.read_hdf(datadir + 'omni_1hour.h5')

ssn_df = LoadSSN(datadir + 'SN_m_tot_V2.0.txt')

#remove ICMEs
icmes = ICMElist.ICMElist(datadir + 'List of Richardson_Cane ICMEs Since January1996_2022.csv') 
omni_nocmes = omni_1hour.copy()
for i in range(0,len(icmes)):
    mask = ((omni_nocmes['datetime'] >= icmes['Shock_time'][i])
            & (omni_nocmes['datetime'] < icmes['ICME_end'][i]) )
    omni_nocmes.loc[mask,'V'] = np.nan
    omni_nocmes.loc[mask,'Bx_gse'] = np.nan



# <codecell> Example CR
CR = 1970

#find the two CRs to be compared
smjd = htime.crnum2mjd(CR)
fmjd = smjd + 27.27
mask_thisCR = (omni_1hour['mjd'] >= smjd) & (omni_1hour['mjd'] < fmjd)
smjd = htime.crnum2mjd(CR+1)
fmjd = smjd + 27.27
mask_nextCR = (omni_1hour['mjd'] >= smjd) & (omni_1hour['mjd'] < fmjd)

v_thisCR = omni_1hour['V'][mask_thisCR].values
v_nextCR = omni_1hour['V'][mask_nextCR].values

Br_thisCR = omni_1hour['Bx_gse'][mask_thisCR].values
Br_nextCR = omni_1hour['Bx_gse'][mask_nextCR].values

v_nocmes_thisCR = omni_nocmes['V'][mask_thisCR].values
v_nocmes_nextCR = omni_nocmes['V'][mask_nextCR].values

Br_nocmes_thisCR = omni_nocmes['Bx_gse'][mask_thisCR].values
Br_nocmes_nextCR = omni_nocmes['Bx_gse'][mask_nextCR].values

fig = plt.figure()

time = omni_1hour['mjd'][mask_thisCR].values
time = time - time[0]

vlims = (250,750)
blims = (-10,10)

ax = plt.subplot(321)
ax.plot(time, v_thisCR, 'k', label = 'ICME')
ax.plot(time, v_nocmes_thisCR, 'r', label = 'Solar wind')
ax.get_xaxis().set_ticklabels([])
ax.set_title('CR' +str(CR))
ax.set_ylim((250,750))
#ax.legend(loc = 'upper right')
ax.set_ylabel(r'$V$ [km/s]')
ax.text(0.05,0.9,'(a)', fontsize = 14, transform=ax.transAxes, backgroundcolor = 'w')


ax = plt.subplot(323)
ax.plot(time, v_nextCR, 'k', label = 'ICME')
ax.plot(time, v_nocmes_nextCR, 'r', label = 'Solar wind')
ax.get_xaxis().set_ticklabels([])
ax.set_title('CR' +str(CR+1))
ax.set_ylim((250,750))
ax.legend(loc = 'upper right')
ax.set_ylabel(r'$V$ [km/s]')
ax.text(0.05,0.9,'(c)', fontsize = 14, transform=ax.transAxes, backgroundcolor = 'w')

# ax = plt.subplot(425)
# ax.plot(time, v_nextCR - v_thisCR , 'k', label = 'ICME')
# ax.plot(time, v_nocmes_nextCR - v_nocmes_thisCR, 'r', label = 'Solar wind')
# ax.get_xaxis().set_ticklabels([])
# ax.set_ylim((-350,350))
# #ax.legend(loc = 'upper right')
# ax.set_ylabel(r'$\Delta V$ [km/s]', fontsize = 16)
# ax.plot([0, 27.27],[0, 0],'k--')
# ax.text(0.05,0.9,'(a)', fontsize = 14, transform=ax.transAxes, backgroundcolor = 'w')

ax = plt.subplot(325)
ax.plot(time, abs(v_nextCR - v_thisCR) , 'k', label = 'ICME')
ax.plot(time, abs(v_nocmes_nextCR - v_nocmes_thisCR), 'r', label = 'Solar wind')
ax.set_ylim((0,400))
#ax.legend(loc = 'upper right')
ax.set_ylabel(r'$|\Delta V|$ [km/s]', fontsize = 16)
ax.text(0.05,0.9,'(e)', fontsize = 14, transform=ax.transAxes, backgroundcolor = 'w')


ax = plt.subplot(322)
ax.plot(time, -Br_thisCR, 'k', label = 'All data')
ax.plot(time, -Br_nocmes_thisCR, 'r', label = 'No ICMEs')
ax.get_xaxis().set_ticklabels([])
ax.set_title('CR' +str(CR))
ax.set_ylim((-15,15))
#ax.legend(loc = 'upper right')
ax.set_ylabel(r'$B_R$ [nT]')
ax.yaxis.tick_right(); ax.yaxis.set_label_position("right")
ax.plot([0, 27.27],[0, 0],'k--')
ax.text(0.05,0.9,'(b)', fontsize = 14, transform=ax.transAxes, backgroundcolor = 'w')

ax = plt.subplot(324)
ax.plot(time, -Br_nextCR, 'k', label = 'All data')
ax.plot(time, -Br_nocmes_nextCR, 'r', label = 'No ICMEs')
ax.get_xaxis().set_ticklabels([])
ax.set_title('CR' +str(CR+1))
ax.set_ylim((-15,15))
#ax.legend(loc = 'upper right')
ax.set_ylabel(r'$B_R$ [nT]')
ax.yaxis.tick_right(); ax.yaxis.set_label_position("right")
ax.plot([0, 27.27],[0, 0],'k--')
ax.text(0.05,0.9,'(d)', fontsize = 14, transform=ax.transAxes, backgroundcolor = 'w')

# ax = plt.subplot(426)
# ax.plot(time, Br_nextCR - Br_thisCR , 'k', label = 'All data')
# ax.plot(time, Br_nocmes_nextCR - Br_nocmes_thisCR, 'r', label = 'No ICMEs')
# ax.get_xaxis().set_ticklabels([])
# ax.set_ylim((-15,15))
# #ax.legend(loc = 'upper right')
# ax.set_ylabel(r'$\Delta B_R$ [nT]', fontsize = 16)
# ax.yaxis.tick_right(); ax.yaxis.set_label_position("right")
# ax.plot([0, 27.27],[0, 0],'k--')

ax = plt.subplot(326)
ax.plot(time, abs(Br_nextCR - Br_thisCR) , 'k', label = 'All data')
ax.plot(time, abs(Br_nocmes_nextCR - Br_nocmes_thisCR), 'r', label = 'No ICMEs')
ax.set_ylim((0,20))
#ax.legend(loc = 'upper right')
ax.set_ylabel(r'$|\Delta B_R|$ [nT]', fontsize = 16)
ax.yaxis.tick_right(); ax.yaxis.set_label_position("right")
ax.text(0.05,0.9,'(f)', fontsize = 14, transform=ax.transAxes, backgroundcolor = 'w')



#add a shared x-axis
ax = fig.add_axes( [0., 0., 1, 1] )
ax.set_axis_off()
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.text( 
    .5, 0.05, "Time through CR [days]", rotation='horizontal',
    horizontalalignment='center', verticalalignment='center'
)

#copute the metrics
print('<dV> all: ' + str(np.nanmean(v_nextCR - v_thisCR)))
print('<dV> NoCME: ' + str(np.nanmean(v_nocmes_nextCR - v_nocmes_thisCR)))
print('<dBr> all: ' + str(np.nanmean(Br_nextCR - Br_thisCR)))
print('<dBr> NoCME: ' + str(np.nanmean(Br_nocmes_nextCR - Br_nocmes_thisCR)))
print(' ')
print('<|dV|> all: ' + str(np.nanmean(abs(v_nextCR - v_thisCR))))
print('<|dV|> NoCME: ' + str(np.nanmean(abs(v_nocmes_nextCR - v_nocmes_thisCR))))
print('<|dBr|> all: ' + str(np.nanmean(abs(Br_nextCR - Br_thisCR))))
print('<|dBr|> NoCME: ' + str(np.nanmean(abs(Br_nocmes_nextCR - Br_nocmes_thisCR))))





# <codecell> Process all OMNI data

dB_omni = np.ones((CRstop-CRstart,2))*np.nan
dV_omni = np.ones((CRstop-CRstart,2))*np.nan
dB_nocmes = np.ones((CRstop-CRstart,2))*np.nan
dV_nocmes = np.ones((CRstop-CRstart,2))*np.nan
nCR_omni = np.ones((CRstop-CRstart))*np.nan
ssn_omni = np.ones((CRstop-CRstart))*np.nan
sai_omni = np.ones((CRstop-CRstart))*np.nan

counter = 0 
for CR in range(CRstart,CRstop):
    
    nCR_omni[counter] = CR + 0.5
    
    #take average sunspot number over both CRs
    smjd = htime.crnum2mjd(CR)
    fmjd = htime.crnum2mjd(CR+2)
    mask = (ssn_df['mjd'] >= smjd) & (ssn_df['mjd'] < fmjd)
    ssn_omni[counter] = np.nanmean(ssn_df.loc[mask,'ssn'])
    sai_omni[counter] = np.nanmean(ssn_df.loc[mask,'sai'])
    
    #find the two CRs to be compared
    smjd = htime.crnum2mjd(CR)
    fmjd = smjd + 27.27
    mask_thisCR = (omni_1hour['mjd'] >= smjd) & (omni_1hour['mjd'] < fmjd)
    smjd = htime.crnum2mjd(CR+1)
    fmjd = smjd + 27.27
    mask_nextCR = (omni_1hour['mjd'] >= smjd) & (omni_1hour['mjd'] < fmjd)
    
    v_thisCR = omni_1hour['V'][mask_thisCR].values
    v_nextCR = omni_1hour['V'][mask_nextCR].values
    
    Br_thisCR = omni_1hour['Bx_gse'][mask_thisCR].values
    Br_nextCR = omni_1hour['Bx_gse'][mask_nextCR].values
    
    v_nocmes_thisCR = omni_nocmes['V'][mask_thisCR].values
    v_nocmes_nextCR = omni_nocmes['V'][mask_nextCR].values
    
    Br_nocmes_thisCR = omni_nocmes['Bx_gse'][mask_thisCR].values
    Br_nocmes_nextCR = omni_nocmes['Bx_gse'][mask_nextCR].values
    
    #force vectors of the same length
    if len(v_thisCR) > len(v_nextCR):
        v_thisCR = np.delete(v_thisCR,-1)
        Br_thisCR = np.delete(Br_thisCR,-1)
        
        v_nocmes_thisCR = np.delete(v_nocmes_thisCR,-1)
        Br_nocmes_thisCR = np.delete(Br_nocmes_thisCR,-1)
    elif len(v_thisCR) < len(v_nextCR):
        v_nextCR = np.delete(v_nextCR,-1)
        Br_nextCR = np.delete(Br_nextCR,-1)
        
        v_nocmes_nextCR = np.delete(v_nocmes_nextCR,-1)
        Br_nocmes_nextCR = np.delete(Br_nocmes_nextCR,-1)
        
        
           
    dV_omni[counter,0] = np.nanmean(abs(v_thisCR - v_nextCR))
    dB_omni[counter,0] = np.nanmean(abs(Br_thisCR - Br_nextCR))
    
    dV_nocmes[counter,0] = np.nanmean(abs(v_nocmes_thisCR - v_nocmes_nextCR))
    dB_nocmes[counter,0] = np.nanmean(abs(Br_nocmes_thisCR - Br_nocmes_nextCR))
    
    #standard error on mean
    Nv = len(np.isfinite(v_thisCR - v_nextCR))
    dV_omni[counter,1] = np.nanstd(abs(v_thisCR - v_nextCR))/np.sqrt(Nv-1)
    Nb = len(np.isfinite(Br_thisCR - Br_nextCR))
    dB_omni[counter,1] = np.nanstd(abs(Br_thisCR - Br_nextCR))/np.sqrt(Nb-1)
    
    Nv = len(np.isfinite(v_nocmes_thisCR - v_nocmes_nextCR))
    dV_nocmes[counter,1] = np.nanstd(abs(v_nocmes_thisCR - v_nocmes_nextCR))/np.sqrt(Nv-1)
    Nb = len(np.isfinite(Br_nocmes_thisCR - Br_nocmes_nextCR))
    dB_nocmes[counter,1] = np.nanstd(abs(Br_nocmes_thisCR - Br_nocmes_nextCR))/np.sqrt(Nb-1)
        
          
    counter = counter + 1
 
mask_max = ((sai_omni >= 0.5))
mask_min = ((sai_omni < 0.5))    

Nv_all = len(np.isfinite(dV_omni[:,0]))
Nv_min = len(np.isfinite(dV_omni[mask_min,0]))
Nv_max = len(np.isfinite(dV_omni[mask_max,0]))
Nb_all = len(np.isfinite(dB_omni[:,0]))
Nb_min = len(np.isfinite(dB_omni[mask_min,0]))
Nb_max = len(np.isfinite(dB_omni[mask_max,0]))

#compute the metrics
print('1995-present OMNI data')
print('<|dV|> all: ' + str(np.nanmean(dV_omni[:,0])) + 
      ' +/- ' + str(round(np.nanstd(dV_omni[:,0])/np.sqrt(Nv_all),2)))
print('<|dV|> NoCME: ' + str(np.nanmean(dV_nocmes[:,0])) + 
      ' +/- ' + str(round(np.nanstd(dV_nocmes[:,0])/np.sqrt(Nv_all),2)))
print('<|dBr|> all: ' + str(np.nanmean(dB_omni[:,0])) + 
      ' +/- ' + str(round(np.nanstd(dB_omni[:,0])/np.sqrt(Nb_all),2)))
print('<|dBr|> NoCME: ' + str(np.nanmean(dB_nocmes[:,0])) + 
      ' +/- ' + str(round(np.nanstd(dB_nocmes[:,0])/np.sqrt(Nb_all),2)))
print('')

print('<|dV|> all (low SAI): ' + str(np.nanmean(dV_omni[mask_min,0])) + 
      ' +/- ' + str(round(np.nanstd(dV_omni[mask_min,0])/np.sqrt(Nv_min),2)))
print('<|dV|> all (high SAI): ' + str(np.nanmean(dV_omni[mask_max,0])) + 
      ' +/- ' + str(round(np.nanstd(dV_omni[mask_max,0])/np.sqrt(Nv_max),2)))
print('<|dBr|> all (low SAI): ' + str(np.nanmean(dB_omni[mask_min,0])) + 
      ' +/- ' + str(round(np.nanstd(dB_omni[mask_min,0])/np.sqrt(Nb_min),2)))
print('<|dBr|> all (high SAI): ' + str(np.nanmean(dB_omni[mask_max,0])) + 
      ' +/- ' + str(round(np.nanstd(dB_omni[mask_max,0])/np.sqrt(Nb_max),2)))
print('')

print('<|dV|> NoCME (low SAI): ' + str(np.nanmean(dV_nocmes[mask_min,0])) + 
      ' +/- ' + str(round(np.nanstd(dV_nocmes[mask_min,0])/np.sqrt(Nv_min),2)))
print('<|dV|> NoCME (high SAI): ' + str(np.nanmean(dV_nocmes[mask_max,0])) + 
      ' +/- ' + str(round(np.nanstd(dV_nocmes[mask_max,0])/np.sqrt(Nv_max),2)))
print('<|dBr|> NoCME (low SAI): ' + str(np.nanmean(dB_nocmes[mask_min,0])) + 
      ' +/- ' + str(round(np.nanstd(dB_nocmes[mask_min,0])/np.sqrt(Nb_min),2)))
print('<|dBr|> NoCME (high SAI): ' + str(np.nanmean(dB_nocmes[mask_max,0])) + 
      ' +/- ' + str(round(np.nanstd(dB_nocmes[mask_max,0])/np.sqrt(Nb_max),2)))



# <codecell> plot the OMNI results

    
#time series plots
plt.figure()

ax = plt.subplot(311)
plt.plot(htime.mjd2datetime(htime.crnum2mjd(nCR_omni)),ssn_omni/200, 'k', label = 'SSN/250')
plt.plot(htime.mjd2datetime(htime.crnum2mjd(nCR_omni)),sai_omni, 'r', label = 'SAI')
plt.ylabel('SSN')
plt.legend(fontsize = 16)
ax.text(0.05,0.9,'(a)', fontsize = 14, transform=ax.transAxes, backgroundcolor = 'w')

ax = plt.subplot(312)
plt.plot(htime.mjd2datetime(htime.crnum2mjd(nCR_omni)),
         dV_omni[:,0], 'b', label = 'All data')
plt.plot(htime.mjd2datetime(htime.crnum2mjd(nCR_omni)),
         dV_nocmes[:,0], 'r', label = 'No ICMEs')
plt.legend(fontsize = 16, loc = 'upper right')
plt.ylabel(r'$<|\Delta V|>_{CR}$ [km/s]', fontsize = 16)
#ax.set_ylim((0,150))
ax.text(0.05,0.9,'(b)', fontsize = 14, transform=ax.transAxes, backgroundcolor = 'w')

ax = plt.subplot(313)
plt.plot(htime.mjd2datetime(htime.crnum2mjd(nCR_omni)), 
         dB_omni[:,0], 'b', label = 'All data')
plt.plot(htime.mjd2datetime(htime.crnum2mjd(nCR_omni)), 
         dB_nocmes[:,0], 'r', label = 'No ICMEs')
plt.legend(fontsize = 16, loc = 'upper right')
plt.ylabel(r'$<|\Delta B_R|>_{CR}$ [nT]', fontsize = 16)
#ax.set_ylim((0,6))
ax.text(0.05,0.9,'(c)', fontsize = 14, transform=ax.transAxes, backgroundcolor = 'w')


#scatter plots
plt.figure()

ax = plt.subplot(221)
ax.plot(ssn_omni,dV_omni[:,0],'k.')
ax.set_ylabel(r'$<|\Delta V|>_{CR}$ [km/s]', fontsize = 16)
#ax.set_xlabel('SSN', fontsize = 16)
ax.set_title('All Data', fontsize = 16)
ax.set_ylim((0,200))
ax.set_xlim((0,250))
pos = np.isfinite(dV_omni[:,0])
r_i = pearsonr(ssn_omni[pos],dV_omni[pos,0])
r_s = spearmanr(ssn_omni[pos],dV_omni[pos,0])
ax.text(0.05, 0.9, r'(a) $r_L$ = {:0.2f}'.format(r_i[0]) + r'; $r_S$ = {:0.2f}'.format(r_s[0]), 
     transform = ax.transAxes)


ax = plt.subplot(223)
ax.plot(ssn_omni,dB_omni[:,0],'k.')
ax.set_ylabel(r'$<|\Delta B_R|>_{CR}$ [nT]', fontsize = 16)
ax.set_xlabel('SSN', fontsize = 16)
ax.set_xlim((0,250))
ax.set_ylim((0,6))
pos = np.isfinite(dB_omni[:,0])
r_i = pearsonr(ssn_omni[pos],dB_omni[:,0][pos])
r_s = spearmanr(ssn_omni[pos],dB_omni[:,0][pos])
ax.text(0.05, 0.9, r'(c) $r_L$ = {:0.2f}'.format(r_i[0]) + r'; $r_S$ = {:0.2f}'.format(r_s[0]), 
     transform = ax.transAxes)

ax = plt.subplot(222)
ax.plot(ssn_omni,dV_nocmes[:,0],'k.')
#ax.set_ylabel(r'$\Delta$ V [km/s]', fontsize = 16)
#ax.set_xlabel('SSN', fontsize = 16)
ax.set_title('No ICMEs', fontsize = 16)
ax.set_ylim((0,200))
ax.set_xlim((0,250))
pos = np.isfinite(dV_nocmes[:,0])
r_i = pearsonr(ssn_omni[pos],dV_nocmes[pos,0])
r_s = spearmanr(ssn_omni[pos],dV_nocmes[pos,0])
ax.text(0.05, 0.9, r'(b) $r_L$ = {:0.2f}'.format(r_i[0]) + r'; $r_S$ = {:0.2f}'.format(r_s[0]), 
     transform = ax.transAxes)

ax = plt.subplot(224)
ax.plot(ssn_omni,dB_nocmes[:,0],'k.')
#ax.set_ylabel(r'$\Delta B_R$ [nT]', fontsize = 16)
ax.set_xlabel('SSN', fontsize = 16)
ax.set_xlim((0,250))
ax.set_ylim((0,6))
pos = np.isfinite(dB_nocmes[:,0])
r_i = pearsonr(ssn_omni[pos],dB_nocmes[:,0][pos])
r_s = spearmanr(ssn_omni[pos],dB_nocmes[:,0][pos])
ax.text(0.05, 0.9, r'(d) $r_L$ = {:0.2f}'.format(r_i[0]) + r'; $r_S$ = {:0.2f}'.format(r_s[0]), 
     transform = ax.transAxes)



#scatter plots
plt.figure()

ax = plt.subplot(221)
ax.plot(sai_omni,dV_omni[:,0],'k.')
ax.set_ylabel(r'$<|\Delta V|>_{CR}$ [km/s]', fontsize = 16)
#ax.set_xlabel('SSN', fontsize = 16)
ax.set_title('All Data', fontsize = 16)
ax.set_ylim((0,200))
ax.set_xlim((0,1))
pos = ( np.isfinite(dV_omni[:,0]) & np.isfinite(sai_omni))
r_i = pearsonr(sai_omni[pos],dV_omni[pos,0])
r_s = spearmanr(sai_omni[pos],dV_omni[pos,0])
ax.text(0.05, 0.9, r'$r_L$ = {:0.2f}'.format(r_i[0]) + r'; $r_S$ = {:0.2f}'.format(r_s[0]), 
     transform = ax.transAxes)

ax = plt.subplot(223)
ax.plot(sai_omni,dB_omni[:,0],'k.')
ax.set_ylabel(r'$<|\Delta B_R|>_{CR}$ [nT]', fontsize = 16)
ax.set_xlabel('SAI', fontsize = 16)
ax.set_xlim((0,1))
ax.set_ylim((0,6))
pos = (np.isfinite(dB_omni[:,0]) & np.isfinite(sai_omni))
r_i = pearsonr(sai_omni[pos],dB_omni[:,0][pos])
r_s = spearmanr(sai_omni[pos],dB_omni[:,0][pos])
ax.text(0.05, 0.9, r'$r_L$ = {:0.2f}'.format(r_i[0]) + r'; $r_S$ = {:0.2f}'.format(r_s[0]), 
     transform = ax.transAxes)

ax = plt.subplot(222)
ax.plot(sai_omni,dV_nocmes[:,0],'k.')
#ax.set_ylabel(r'$\Delta$ V [km/s]', fontsize = 16)
#ax.set_xlabel('SSN', fontsize = 16)
ax.set_title('No ICMEs', fontsize = 16)
ax.set_ylim((0,200))
ax.set_xlim((0,1))
pos = (np.isfinite(dV_nocmes[:,0]) & np.isfinite(sai_omni))
r_i = pearsonr(sai_omni[pos],dV_nocmes[pos,0])
r_s = spearmanr(sai_omni[pos],dV_nocmes[pos,0])
ax.text(0.05, 0.9, r'$r_L$ = {:0.2f}'.format(r_i[0]) + r'; $r_S$ = {:0.2f}'.format(r_s[0]), 
     transform = ax.transAxes)

ax = plt.subplot(224)
ax.plot(sai_omni,dB_nocmes[:,0],'k.')
#ax.set_ylabel(r'$\Delta B_R$ [nT]', fontsize = 16)
ax.set_xlabel('SAI', fontsize = 16)
ax.set_xlim((0,1))
ax.set_ylim((0,6))
pos = (np.isfinite(dB_nocmes[:,0]) & np.isfinite(sai_omni))
r_i = pearsonr(sai_omni[pos],dB_nocmes[:,0][pos])
r_s = spearmanr(sai_omni[pos],dB_nocmes[:,0][pos])
ax.text(0.05, 0.9, r'$r_L$ = {:0.2f}'.format(r_i[0]) + r'; $r_S$ = {:0.2f}'.format(r_s[0]), 
     transform = ax.transAxes)

print('N = ', str(len(pos)))

# <codecell> box'n'whiskers plots

plt.figure()
colors = ['k', 'b', 'r']

ax = plt.subplot(221)
box = ax.boxplot([dV_omni[:,0], dV_omni[mask_min,0], dV_omni[mask_max,0]],
           notch=True, patch_artist=True,showfliers=False,whis=1.5)
for patch, median, color in zip(box['boxes'], box['medians'], colors):
    patch.set(facecolor = color)
    patch.set(color = color)
    median.set(color='w') 
ax.set_ylim((0,125))
ax.set_ylabel(r'$<|\Delta V|>_{CR}$ [km/s]', fontsize = 16)
ax.get_xaxis().set_ticks([])
ax.set_title('ICMEs included', fontsize = 16)
ax.text(0.03, 0.05, r'(a)', transform = ax.transAxes)



ax = plt.subplot(222)
box = ax.boxplot([dV_nocmes[:,0], dV_nocmes[mask_min,0], dV_nocmes[mask_max,0]],
           notch=True, patch_artist=True,showfliers=False,whis=1.5)
for patch, median, color in zip(box['boxes'], box['medians'], colors):
    patch.set(facecolor = color)
    patch.set(color = color)
    median.set(color='w')
ax.set_ylim((0,125))
ax.get_xaxis().set_ticks([])
ax.set_title('ICMEs removed', fontsize = 16)
ax.text(0.03, 0.05, r'(b)', transform = ax.transAxes)

ax = plt.subplot(223)
box = ax.boxplot([dB_omni[:,0], dB_omni[mask_min,0], dB_omni[mask_max,0]],
           notch=True, patch_artist=True,showfliers=False,whis=1.5)
for patch, median, color in zip(box['boxes'], box['medians'], colors):
    patch.set(facecolor = color)
    patch.set(color = color)
    median.set(color='w')
ax.set_ylim((0,4.7))
ax.set_ylabel(r'$<|\Delta B_R|>_{CR}$ [nT]', fontsize = 16)
ax.get_xaxis().set_ticks([])
ax.text(0.03, 0.05, r'(c)', transform = ax.transAxes)

ax.legend([box["boxes"][0], box["boxes"][1],box["boxes"][2]], 
          ["All data", "SAI < 0.5", "SAI > 0.5"], 
          loc='upper center', bbox_to_anchor=(0.5,1.3), framealpha = 1)  

    
ax = plt.subplot(224)
box = ax.boxplot([dB_nocmes[:,0], dB_nocmes[mask_min,0], dB_nocmes[mask_max,0]],
           notch=True, patch_artist=True,showfliers=False,whis=1.5)
for patch, median, color in zip(box['boxes'], box['medians'], colors):
    patch.set(facecolor = color)
    patch.set(color = color)
    median.set(color='w')
ax.set_ylim((0,4.7))
ax.get_xaxis().set_ticks([])
ax.text(0.03, 0.05, r'(d)', transform = ax.transAxes)

# <codecell> download HelioMAS data for each observatory
#heliomasdir = 'D:\\Dropbox\\Data\\HelioMAS\\'


crstart = 1625
crend = 2232
observatories = ['hmi', 'mdi', 'solis', 'gong', 'mwo', 'wso', 'kpo']

#observatories = ['mdi', 'solis',  'mwo', 'wso']

if downloadnow:
    for obs in observatories:
        #move to the appropriate directory
        MYDIR = heliomasdir + obs
        CHECK_FOLDER = os.path.isdir(MYDIR)
        if not CHECK_FOLDER:
            os.makedirs(MYDIR)
            print("created folder : ", MYDIR)
    
        else:
            print(MYDIR, "folder already exists.")
        os.chdir(MYDIR)
        
        
        for cr in range(crstart, crend):
            #move to the appropriate directory
            CRDIR = MYDIR + '\\CR' + str(cr)
            CHECK_FOLDER = os.path.isdir(CRDIR)
            if not CHECK_FOLDER:
                os.makedirs(CRDIR)
                print("created folder : ", CRDIR)
    
            else:
                print(CRDIR, "folder already exists.")
            os.chdir(CRDIR)
            
            get_helioMAS_output(cr=cr, observatory=obs)




# <codecell> Display some example data
CR = 2100
r_obs = 30
bmax = 100
obs = 'hmi'

brfilepath = heliomasdir + obs + '\\CR' + str(CR) + '\\br002.hdf'
vrfilepath = heliomasdir + obs + '\\CR' + str(CR) + '\\vr002.hdf'
br_next_filepath = heliomasdir + obs + '\\CR' + str(CR+1) + '\\br002.hdf'
vr_next_filepath = heliomasdir + obs + '\\CR' + str(CR+1) + '\\vr002.hdf'
#load the Vr data
Vr, VXa, VXm, VXr = read_HelioMAS(vrfilepath)
Vr_next, VXa_next, VXm_next, VXr_next = read_HelioMAS(vr_next_filepath)
Vr = Vr * 481.0
Vr_next = Vr_next * 481.0

#load the Br data
Br, BXa, BXm, BXr = read_HelioMAS(brfilepath) 
Br_next, BXa_next, BXm_next, BXr_next = read_HelioMAS(br_next_filepath)
Br = Br *  2.2e5
Br_next = Br_next *  2.2e5

#take the slice at r = 215 rS
id_r_v = np.argmin(abs(VXr - r_obs*u.solRad))

id_r_b = np.argmin(abs(BXr - r_obs*u.solRad))

#find lats around teh ecliptic
delta_lat = 7.25 * np.pi/180 *u.rad
mask_v = ((VXm > np.pi/2*u.rad - delta_lat) & (VXm < np.pi/2*u.rad + delta_lat))
mask_b = ((BXm > np.pi/2*u.rad - delta_lat) & (BXm < np.pi/2*u.rad + delta_lat))

#compute the difference globally
dV = np.nanmean(abs(Vr[:,:,id_r_v] - Vr_next[:,:,id_r_v]))
dB = np.nanmean(abs(Br[:,:,id_r_b] - Br_next[:,:,id_r_b]))

#compute the difference at the ecliptic
dV_eclip = np.nanmean(abs(Vr[:, mask_v, id_r_v] 
                                   - Vr_next[:, mask_v, id_r_v]))
dB_eclip = np.nanmean(abs(Br[:, mask_b, id_r_b] 
                                   - Br_next[:, mask_b, id_r_b]))




fig = plt.figure()

ax = plt.subplot(321)
im = ax.pcolor(VXa.value*180/np.pi, 90 - VXm.value*180/np.pi, np.flipud(Vr[:,:,id_r_v].T))
ax.set_yticks([-90, 0, 90])
ax.set_xticks([0, 90, 180, 270, 360])
ax.get_xaxis().set_ticklabels([])
#ax.set_title('CR' +str(CR))
ax.text(0.05,0.9,'(a)' + ' CR' +str(CR), fontsize = 14, transform=ax.transAxes, backgroundcolor = 'w')
ax.set_xlim((0,360))
ax.plot([0, 360],[7.5, 7.5],'w--'); ax.plot([0, 360],[-7.5, -7.5],'w--');
cb = plt.colorbar(im); cb.ax.tick_params(labelsize=12)
cb.ax.set_title(r'$V$ [km/s]', fontsize = 14)

ax = plt.subplot(323)
im = ax.pcolor(VXa.value*180/np.pi, 90 - VXm.value*180/np.pi, np.flipud(Vr_next[:,:,id_r_v].T))
ax.set_yticks([-90, 0, 90])
ax.set_xticks([0, 90, 180, 270, 360])
ax.get_xaxis().set_ticklabels([])
#ax.set_title('CR' +str(CR+1))
ax.text(0.05,0.9,'(c)' + ' CR' +str(CR+1), fontsize = 14, transform=ax.transAxes, backgroundcolor = 'w')
ax.set_xlim((0,360))
ax.plot([0, 360],[7.5, 7.5],'w--'); ax.plot([0, 360],[-7.5, -7.5],'w--');
cb = plt.colorbar(im); cb.ax.tick_params(labelsize=12)
cb.ax.set_title(r'$V$ [km/s]', fontsize = 14)

# ax = plt.subplot(325)
# im = ax.pcolor(VXa.value*180/np.pi, 90 - VXm.value*180/np.pi, 
#           -np.flipud(Vr[:,:,id_r_v].T - Vr_next[:,:,id_r_v].T), cmap='RdBu',
#           norm=plt.Normalize(-200,200))
# ax.set_yticks([-90, 0, 90])
# ax.set_xticks([0, 90, 180, 270, 360])
# ax.get_xaxis().set_ticklabels([])
# ax.set_xlim((0,360))
# ax.plot([0, 360],[7.5, 7.5],'k--'); ax.plot([0, 360],[-7.5, -7.5],'k--');
# cb = plt.colorbar(im); cb.ax.tick_params(labelsize=12)
# cb.ax.set_title(r'$\Delta V$ [km/s]', fontsize = 14)

ax = plt.subplot(325)
im = ax.pcolor(VXa.value*180/np.pi, 90 - VXm.value*180/np.pi, 
          np.flipud(abs(Vr[:,:,id_r_v].T - Vr_next[:,:,id_r_v].T)), norm=plt.Normalize(0,200))
ax.set_yticks([-90, 0, 90])
ax.set_xticks([0, 90, 180, 270, 360])
ax.set_xlim((0,360))
ax.plot([0, 360],[7.5, 7.5],'w--'); ax.plot([0, 360],[-7.5, -7.5],'w--');
cb = plt.colorbar(im); cb.ax.tick_params(labelsize=12)
cb.ax.set_title(r'$|\Delta V|$ [km/s]', fontsize = 14)
ax.text(0.05,0.9,'(e)', fontsize = 14, transform=ax.transAxes, backgroundcolor = 'w')


ax = plt.subplot(322)
im = ax.pcolor(BXa.value*180/np.pi, 90 - BXm.value*180/np.pi, 
          np.flipud(Br[:,:,id_r_v].T),cmap='RdBu',norm=plt.Normalize(-bmax,bmax))
ax.set_yticks([-90, 0, 90])
ax.set_xticks([0, 90, 180, 270, 360])
ax.get_xaxis().set_ticklabels([])
ax.get_yaxis().set_ticklabels([])
#ax.set_title('CR' +str(CR))
ax.set_xlim((0,360))
ax.plot([0, 360],[7.5, 7.5],'k--'); ax.plot([0, 360],[-7.5, -7.5],'k--');
cb = plt.colorbar(im); cb.ax.tick_params(labelsize=12)
cb.ax.set_title(r'$B_R$ [nT]', fontsize = 14)
ax.text(0.05,0.9,'(b)' + ' CR' +str(CR), fontsize = 14, transform=ax.transAxes, backgroundcolor = 'w')

ax = plt.subplot(324)
im = ax.pcolor(BXa.value*180/np.pi, 90 - BXm.value*180/np.pi, 
          np.flipud(Br_next[:,:,id_r_v].T),cmap='RdBu',norm=plt.Normalize(-bmax,bmax))
ax.set_yticks([-90, 0, 90])
ax.set_xticks([0, 90, 180, 270, 360])
ax.get_xaxis().set_ticklabels([])
ax.get_yaxis().set_ticklabels([])
#ax.set_title('CR' +str(CR+1))
ax.set_xlim((0,360))
ax.plot([0, 360],[7.5, 7.5],'k--'); ax.plot([0, 360],[-7.5, -7.5],'k--');
cb = plt.colorbar(im); cb.ax.tick_params(labelsize=12)
cb.ax.set_title(r'$B_R$ [nT]', fontsize = 14)
ax.text(0.05,0.9,'(d)' + ' CR' +str(CR+1), fontsize = 14, transform=ax.transAxes, backgroundcolor = 'w')

# ax = plt.subplot(426)
# im = ax.pcolor(BXa.value*180/np.pi, 90 - BXm.value*180/np.pi, 
#           -np.flipud(Br[:,:,id_r_v].T - Br_next[:,:,id_r_v].T), cmap='RdBu',
#           norm=plt.Normalize(-2,2))
# ax.set_yticks([-90, 0, 90])
# ax.set_xticks([0, 90, 180, 270, 360])
# ax.get_yaxis().set_ticklabels([])
# ax.get_xaxis().set_ticklabels([])
# ax.set_xlim((0,360))
# ax.plot([0, 360],[7.5, 7.5],'k--'); ax.plot([0, 360],[-7.5, -7.5],'k--');
# cb = plt.colorbar(im); cb.ax.tick_params(labelsize=12)
# cb.ax.set_title(r'$\Delta B_R$ [nT]', fontsize = 14)

ax = plt.subplot(326)
im = ax.pcolor(BXa.value*180/np.pi, 90 - BXm.value*180/np.pi, 
          np.flipud(abs(Br[:,:,id_r_v].T - Br_next[:,:,id_r_v].T)),
          norm=plt.Normalize(0,bmax))
ax.set_yticks([-90, 0, 90])
ax.set_xticks([0, 90, 180, 270, 360])
ax.get_yaxis().set_ticklabels([])
ax.set_xlim((0,360))
ax.plot([0, 360],[7.5, 7.5],'w--'); ax.plot([0, 360],[-7.5, -7.5],'w--');
cb = plt.colorbar(im); cb.ax.tick_params(labelsize=12)
cb.ax.set_title(r'$|\Delta B_R|$ [nT]', fontsize = 14)
ax.text(0.05,0.9,'(f)', fontsize = 14, transform=ax.transAxes, backgroundcolor = 'w')

#add a shared y-axis
ax = fig.add_axes( [0., 0., 1, 1] )
ax.set_axis_off()
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.text( 
    .05, 0.5, "Latitude [degrees]", rotation='vertical',
    horizontalalignment='center', verticalalignment='center'
)

#add a shared x-axis
ax = fig.add_axes( [0., 0., 1, 1] )
ax.set_axis_off()
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.text( 
    .5, 0.05, "Carrington longitude [degrees]", rotation='horizontal',
    horizontalalignment='center', verticalalignment='center'
)

delta_lat = 7.25 * np.pi/180 *u.rad
mask_v = ((VXm > np.pi/2*u.rad - delta_lat) & (VXm < np.pi/2*u.rad + delta_lat))
mask_b = ((BXm > np.pi/2*u.rad - delta_lat) & (BXm < np.pi/2*u.rad + delta_lat))

print('<dV> global: ' + str(np.nanmean(Vr_next[:,:,id_r_v] - Vr[:,:,id_r_v])))
print('<dBr> global: ' + str(np.nanmean(Br_next[:,:,id_r_v] - Br[:,:,id_r_v])))
print('<dV> eclip: ' + str(np.nanmean(Vr_next[:,mask_v,id_r_v] - Vr[:,mask_v,id_r_v])))
print('<dBr> eclip: ' + str(np.nanmean(Br_next[:,mask_b,id_r_v] - Br[:,mask_b,id_r_v])))
print(' ')

print('<|dV|> global: ' + str(np.nanmean(abs(Vr_next[:,:,id_r_v] - Vr[:,:,id_r_v]))))
print('<|dBr|> global: ' + str(np.nanmean(abs(Br_next[:,:,id_r_v] - Br[:,:,id_r_v]))))
print('<|dV|> eclip: ' + str(np.nanmean(abs(Vr_next[:,mask_v,id_r_v] - Vr[:,mask_v,id_r_v]))))
print('<|dBr|> eclip: ' + str(np.nanmean(abs(Br_next[:,mask_b,id_r_v] - Br[:,mask_b,id_r_v]))))


# <codecell> Process HelioMAS data

#loop through the HelioMAS solutions and compute the change from CR to CR
#HelioMASdir = 'D:\\Dropbox\\Data_archive\\MAS_helio\\'
HelioMASdir = os.environ['DBOX'] + 'Data\\HelioMAS\\'
ssn_df = LoadSSN()

CRstart = 1625
CRstop = 2232


observatories = ['hmi', 'mdi', 'solis', 'gong', 'mwo', 'wso', 'kpo']

df = pd.DataFrame()

lats = np.arange(2.5,180-2.499999,5) * np.pi/180
Nlats = len(lats)-1

for obs in observatories: 
    
    print('Processing ' + obs)
    
    dB = np.ones((CRstop-CRstart))*np.nan
    dV = np.ones((CRstop-CRstart))*np.nan
    dB_eclip = np.ones((CRstop-CRstart))*np.nan
    dV_eclip = np.ones((CRstop-CRstart))*np.nan
    dB_lat = np.ones((CRstop-CRstart,Nlats))*np.nan
    dV_lat = np.ones((CRstop-CRstart,Nlats))*np.nan
    nCR = np.ones((CRstop-CRstart))*np.nan
    ssn = np.ones((CRstop-CRstart))*np.nan
    sai = np.ones((CRstop-CRstart))*np.nan
    
    counter = 0 
    for CR in range(CRstart,CRstop):
        brfilepath = heliomasdir + obs + '\\CR' + str(CR) + '\\br002.hdf'
        vrfilepath = heliomasdir + obs + '\\CR' + str(CR) + '\\vr002.hdf'
        br_next_filepath = heliomasdir + obs + '\\CR' + str(CR+1) + '\\br002.hdf'
        vr_next_filepath = heliomasdir + obs + '\\CR' + str(CR+1) + '\\vr002.hdf'
        #brfilepath = HelioMASdir + 'CR' + str(CR) + '\HelioMAS_CR' + str(CR) + '_br002.hdf'
        #vrfilepath = HelioMASdir + 'CR' + str(CR) + '\HelioMAS_CR' + str(CR) + '_vr002.hdf'
        #br_next_filepath = HelioMASdir + 'CR' + str(CR+1) + '\HelioMAS_CR' + str(CR+1) + '_br002.hdf'
        #vr_next_filepath = HelioMASdir + 'CR' + str(CR+1) + '\HelioMAS_CR' + str(CR+1) + '_vr002.hdf'
        
        
        
        nCR[counter] = CR + 0.5
        
        smjd = htime.crnum2mjd(CR)
        fmjd = htime.crnum2mjd(CR+2)
        
        mask = (ssn_df['mjd'] >= smjd) & (ssn_df['mjd'] < fmjd)
        ssn[counter] = np.nanmean(ssn_df.loc[mask,'ssn'])
        sai[counter] = np.nanmean(ssn_df.loc[mask,'sai'])
        
        
        if ((os.path.exists(brfilepath) == True) &
            (os.path.exists(br_next_filepath) == True)):
            
            #load the Vr data
            Vr, VXa, VXm, VXr = read_HelioMAS(vrfilepath)
            Vr_next, VXa_next, VXm_next, VXr_next = read_HelioMAS(vr_next_filepath)
            Vr = Vr * 481.0
            Vr_next = Vr_next * 481.0
            
            #load the Br data
            Br, BXa, BXm, BXr = read_HelioMAS(brfilepath) 
            Br_next, BXa_next, BXm_next, BXr_next = read_HelioMAS(br_next_filepath)
            Br = Br *  2.2e5
            Br_next = Br_next *  2.2e5
            
            #check that consecutive runs are the same resolution
            if (Br.size == Br_next.size) & (Vr.size == Vr_next.size):
            
                #take the slice at r = 215 rS
                id_r_v = np.argmin(abs(VXr - 215*u.solRad))
                
                id_r_b = np.argmin(abs(BXr - 215*u.solRad))
                
                #find lats around teh ecliptic
                delta_lat = 7.25 * np.pi/180 *u.rad
                mask_v = ((VXm > np.pi/2*u.rad - delta_lat) & (VXm < np.pi/2*u.rad + delta_lat))
                mask_b = ((BXm > np.pi/2*u.rad - delta_lat) & (BXm < np.pi/2*u.rad + delta_lat))
                
                #compute the difference globally
                dV[counter] = np.nanmean(abs(Vr[:,:,id_r_v] - Vr_next[:,:,id_r_v]))
                dB[counter] = np.nanmean(abs(Br[:,:,id_r_b] - Br_next[:,:,id_r_b]))
                
                #compute the difference at the ecliptic
                dV_eclip[counter] = np.nanmean(abs(Vr[:, mask_v, id_r_v] 
                                                   - Vr_next[:, mask_v, id_r_v]))
                dB_eclip[counter] = np.nanmean(abs(Br[:, mask_b, id_r_b] 
                                                   - Br_next[:, mask_b, id_r_b]))
                
                
                #now find teh differences at each lat bin
                for ilat in range(0,Nlats):
                    mask_v = ((VXm >= lats[ilat]*u.rad) & (VXm < lats[ilat+1]*u.rad))
                    mask_b = ((BXm >= lats[ilat]*u.rad) & (BXm < lats[ilat+1]*u.rad))
                
                    #compute the difference at the given lat band
                    dV_lat[counter,ilat] = np.nanmean(abs(Vr[:, mask_v, id_r_v] 
                                                       - Vr_next[:, mask_v, id_r_v]))
                    dB_lat[counter,ilat] = np.nanmean(abs(Br[:, mask_b, id_r_b] 
                                                       - Br_next[:, mask_b, id_r_b]))
                
                
            
            else:
                dV[counter] = np.nan
                dB[counter] = np.nan
                dV_eclip[counter] = np.nan
                dB_eclip[counter] = np.nan  
                dV_lat[counter,:] = np.nan
                dB_lat[counter,:] = np.nan 
        else:
            dV[counter] = np.nan
            dB[counter] = np.nan
            dV_eclip[counter] = np.nan
            dB_eclip[counter] = np.nan
            dV_lat[counter,:] = np.nan
            dB_lat[counter,:] = np.nan 
    
        counter = counter + 1
    
    #add the data to the data frame
    df['CR'] = nCR
    df['SSN'] = ssn
    df['SAI'] = sai
    df[obs + '_dV_global'] = dV
    df[obs + '_dV_eclip'] = dV_eclip
    df[obs + '_dB_global'] = dB
    df[obs + '_dB_eclip'] = dB_eclip
    for ilat in range(0,Nlats):
        df[obs + '_dB_lat_' + str(ilat)] = dB_lat[:,ilat]
        df[obs + '_dV_lat_' + str(ilat)] = dV_lat[:,ilat]

#convert to datetime
df['datetime'] = htime.mjd2datetime(htime.crnum2mjd(df['CR'].values))

#take averages across all observatories
#observatories = ['hmi', 'mdi', 'solis', 'gong', 'mwo', 'wso', 'kpo']
df['dV'] = df[['hmi_dV_global','mdi_dV_global','solis_dV_global',
               'gong_dV_global','mwo_dV_global','wso_dV_global', 
               'kpo_dV_global']].mean(axis = 1, skipna=True)
df['dV_eclip'] = df[['hmi_dV_eclip','mdi_dV_eclip','solis_dV_eclip',
               'gong_dV_eclip','mwo_dV_eclip','wso_dV_eclip', 
               'kpo_dV_eclip']].mean(axis = 1, skipna=True)
df['dB'] = df[['hmi_dB_global','mdi_dB_global','solis_dB_global',
               'gong_dB_global','mwo_dB_global','wso_dB_global', 
               'kpo_dB_global']].mean(axis = 1, skipna=True)
df['dB_eclip'] = df[['hmi_dB_eclip','mdi_dB_eclip','solis_dB_eclip',
               'gong_dB_eclip','mwo_dB_eclip','wso_dB_eclip', 
               'kpo_dB_eclip']].mean(axis = 1, skipna=True)

for ilat in range(0,Nlats):
    l = str(ilat)
    df['dB_lat_' + str(ilat)] = df[['hmi_dB_lat_' + l,'mdi_dB_lat_' + l,'solis_dB_lat_' + l,
                   'gong_dB_lat_' + l,'mwo_dB_lat_' + l,'wso_dB_lat_' + l, 
                   'kpo_dB_lat_' + l]].mean(axis = 1, skipna=True)
    df['dV_lat_' + str(ilat)] = df[['hmi_dV_lat_' + l,'mdi_dV_lat_' + l,'solis_dV_lat_' + l,
                   'gong_dV_lat_' + l,'mwo_dV_lat_' + l,'wso_dV_lat_' + l, 
                   'kpo_dV_lat_' + l]].mean(axis = 1, skipna=True)
    


mask_max = ((df['SAI'] >= 0.5))
mask_min = ((df['SAI'] < 0.5)) 

Nall = len(np.isfinite(df['dV']))   
Nmin = len(np.isfinite(df.loc[mask_min,'dV']))
Nmax = len(np.isfinite(df.loc[mask_max,'dV']))

#compute the metrics
print('HelioMAS')
print('<|dV|> global: ' + str(round(np.nanmean(df['dV']),2)) + 
      ' +/- ' + str(round(np.nanstd(df['dV'])/np.sqrt(Nall),2)))
print('<|dV|> ecliptic: ' + str(round(np.nanmean(df['dV_eclip']),2)) + 
      ' +/- ' + str(round(np.nanstd(df['dV_eclip'])/np.sqrt(Nall),2)))

print('<|dBr|> global: ' + str(round(np.nanmean(df['dB']),2)) + 
      ' +/- ' + str(round(np.nanstd(df['dB'])/np.sqrt(Nall),2)))
print('<|dBr|> ecliptic: ' + str(round(np.nanmean(df['dB_eclip']),2)) + 
      ' +/- ' + str(round(np.nanstd(df['dB_eclip'])/np.sqrt(Nall),2)))
print('')

print('<|dV|> global (low SAI): ' + str(round(np.nanmean(df.loc[mask_min,'dV']),2)) + 
      ' +/- ' + str(round(np.nanstd(df.loc[mask_min,'dV'])/np.sqrt(Nmin),2)))
print('<|dV|> global (high SAI): ' + str(round(np.nanmean(df.loc[mask_max,'dV']),2)) + 
      ' +/- ' + str(round(np.nanstd(df.loc[mask_max,'dV'])/np.sqrt(Nmax),2)))
print('<|dBr|> global (low SAI): ' + str(round(np.nanmean(df.loc[mask_min,'dB']),2)) + 
      ' +/- ' + str(round(np.nanstd(df.loc[mask_min,'dB'])/np.sqrt(Nmin),2)))
print('<|dBr|> global (high SAI): ' + str(round(np.nanmean(df.loc[mask_max,'dB']),2)) + 
      ' +/- ' + str(round(np.nanstd(df.loc[mask_max,'dB'])/np.sqrt(Nmax),2)))
print('')

print('<|dV|> ecliptic (low SAI): ' + str(round(np.nanmean(df.loc[mask_min,'dV_eclip']),2)) + 
      ' +/- ' + str(round(np.nanstd(df.loc[mask_min,'dV_eclip'])/np.sqrt(Nmin),2)))
print('<|dV|> ecliptic (high SAI): ' + str(round(np.nanmean(df.loc[mask_max,'dV_eclip']),2)) + 
      ' +/- ' + str(round(np.nanstd(df.loc[mask_max,'dV_eclip'])/np.sqrt(Nmax),2)))
print('<|dBr|> ecliptic (low SAI): ' + str(round(np.nanmean(df.loc[mask_min,'dB_eclip']),2)) + 
      ' +/- ' + str(round(np.nanstd(df.loc[mask_min,'dB_eclip'])/np.sqrt(Nmin),2)))
print('<|dBr|> ecliptic (high SAI): ' + str(round(np.nanmean(df.loc[mask_max,'dB_eclip']),2)) + 
      ' +/- ' + str(round(np.nanstd(df.loc[mask_max,'dB_eclip'])/np.sqrt(Nmax),2)))
print('')

#post 1995 values?

#export the lat composite as an array
dB_lat = np.ones((CRstop-CRstart,Nlats))*np.nan
dV_lat = np.ones((CRstop-CRstart,Nlats))*np.nan

dV_lat_min = np.ones((Nlats,2))*np.nan
dV_lat_max = np.ones((Nlats,2))*np.nan
dV_lat_all = np.ones((Nlats,2))*np.nan

dB_lat_min = np.ones((Nlats,2))*np.nan
dB_lat_max = np.ones((Nlats,2))*np.nan
dB_lat_all = np.ones((Nlats,2))*np.nan

r_dV_lat = np.ones((Nlats,2))*np.nan
r_dB_lat = np.ones((Nlats,2))*np.nan

#compute the delat and correlation at each latitude
for ilat in range(0,Nlats):
    dB_lat[:,ilat] = df['dB_lat_' + str(ilat)].to_numpy()
    dV_lat[:,ilat] = df['dV_lat_' + str(ilat)].to_numpy()
    
    mask_max = ((df['SAI'] >= 0.5))
    mask_min = ((df['SAI'] < 0.5))
    
    dV_lat_all[ilat,1] = np.nanmean(dV_lat[:,ilat])
    dB_lat_all[ilat,1] = np.nanmean(dB_lat[:,ilat])
    
    dV_lat_max[ilat,1] = np.nanmean(dV_lat[mask_max,ilat])
    dB_lat_max[ilat,1] = np.nanmean(dB_lat[mask_max,ilat])
    
    dV_lat_min[ilat,1] = np.nanmean(dV_lat[mask_min,ilat])
    dB_lat_min[ilat,1] = np.nanmean(dB_lat[mask_min,ilat])
    
    #compute correlations
    pos = ((np.isfinite(dV_lat[:,ilat]) & (np.isfinite(df['SSN'].values))))
    r_i = pearsonr(df[pos]['SSN'].values, dV_lat[pos,ilat])
    r_s = spearmanr(df[pos]['SSN'].values, dV_lat[pos,ilat])
    r_dV_lat[ilat,0] = r_i[0]
    r_dV_lat[ilat,1] = r_s[0]
    pos = ((np.isfinite(dB_lat[:,ilat]) & (np.isfinite(df['SSN'].values))))
    r_i = pearsonr(df[pos]['SSN'].values, dB_lat[pos,ilat])
    r_s = spearmanr(df[pos]['SSN'].values, dB_lat[pos,ilat])
    r_dB_lat[ilat,0] = r_i[0]
    r_dB_lat[ilat,1] = r_s[0]
    
    
# <codecell> plot the HelioMAS results

#plt.rcParams.update({'font.size': 14})
from cycler import cycler
plt.rcParams['axes.prop_cycle'] = cycler(color='bgkymcr')

plot_observatories = ['hmi', 'mdi', 'solis', 'gong', 'mwo', 'wso', 'kpo']




#time series plots
fig = plt.figure()

ax = plt.subplot(311)
plt.plot(df['datetime'],df['SSN']/200, 'k', label = 'SSN/200')
plt.plot(df['datetime'],df['SAI'], 'r', label = 'SAI')
plt.ylabel('SSN')
plt.legend(fontsize = 16)
ax.text(0.05,0.9,'(a)', fontsize = 14, transform=ax.transAxes, backgroundcolor = 'w')

ax = plt.subplot(312)
for obs in plot_observatories: 
    plt.plot(df['datetime'], df[obs + '_dV_global'], label = obs)
#plt.legend(fontsize = 16)
ax.set_ylabel(r'$<|\Delta V|>_{CR}$ [km/s]' + '\n(Global)', fontsize = 16)
ax.set_ylim((0,230))
ax.text(0.05,0.9,'(b)', fontsize = 14, transform=ax.transAxes, backgroundcolor = 'w')

ax = plt.subplot(313)
for obs in plot_observatories: 
    plt.plot(df['datetime'], df[obs + '_dV_eclip'], label = obs)
#plt.legend(fontsize = 16)
ax.set_ylabel(r'$<|\Delta V|>_{CR}$ [km/s]' + '\n(Ecliptic)', fontsize = 16)
ax.set_ylim((0,230))
ax.text(0.05,0.9,'(c)', fontsize = 14, transform=ax.transAxes, backgroundcolor = 'w')

handles, labels = ax.get_legend_handles_labels()
fig.legend(handles, labels, loc='right', framealpha=1)


fig = plt.figure()

ax = plt.subplot(311)
plt.plot(df['datetime'],df['SSN']/200, 'k', label = 'SSN/200')
plt.plot(df['datetime'],df['SAI'], 'r', label = 'SAI')
plt.ylabel('SSN')
plt.legend(fontsize = 16)
ax.text(0.05,0.9,'(a)', fontsize = 14, transform=ax.transAxes, backgroundcolor = 'w')

ax = plt.subplot(312)
for obs in plot_observatories: 
    plt.plot(df['datetime'], df[obs + '_dB_global'], label = obs)
#plt.legend(fontsize = 16)
ax.set_ylabel(r'$<|\Delta$ $B_R|>_{CR}$ [nT]' + '\n(Global)', fontsize = 16)
ax.set_ylim((0,2))
ax.text(0.05,0.9,'(b)', fontsize = 14, transform=ax.transAxes, backgroundcolor = 'w')

ax = plt.subplot(313)
for obs in plot_observatories: 
    plt.plot(df['datetime'], df[obs + '_dB_eclip'], label = obs)
#plt.legend(fontsize = 16)
ax.set_ylabel(r'$<|\Delta$ $B_R|>_{CR}$ [nT]' + '\n(Ecliptic)', fontsize = 16)
ax.set_ylim((0,2))
ax.text(0.05,0.9,'(c)', fontsize = 14, transform=ax.transAxes, backgroundcolor = 'w')

handles, labels = ax.get_legend_handles_labels()
fig.legend(handles, labels, loc='right', framealpha=1)

#==============================================================================
#scatter plots
#==============================================================================
fig = plt.figure()

ax = plt.subplot(221)
for obs in plot_observatories: 
    ax.plot(df['SSN'], df[obs + '_dV_global'], '.', label = obs)
#ax.plot(ssn,dV,'k.')
ax.set_ylabel(r'$<|\Delta V|>_{CR}$ [km/s]', fontsize = 16)
ax.set_title('Global', fontsize = 16)
ax.set_ylim((0,250))
ax.set_xlim((0,250))
pos = np.isfinite(df['dV'].values)
r_i = pearsonr(df[pos]['SSN'].values,df[pos]['dV'].values)
r_s = spearmanr(df[pos]['SSN'].values,df[pos]['dV'].values)
ax.text(0.05, 0.9, r'(a) $r_L$ = {:0.2f}'.format(r_i[0]) + r'; $r_S$ = {:0.2f}'.format(r_s[0]), 
     transform = ax.transAxes)

ax = plt.subplot(223)
for obs in plot_observatories: 
    ax.plot(df['SSN'], df[obs + '_dB_global'], '.', label = obs)
ax.set_ylabel(r'$<|\Delta B_R|>_{CR}$ [nT]', fontsize = 16)
ax.set_xlabel('SSN', fontsize = 16)
ax.set_xlim((0,250))
ax.set_ylim((0,2.7))
pos = np.isfinite(df['dB'].values)
r_i = pearsonr(df[pos]['SSN'].values,df[pos]['dB'].values)
r_s = spearmanr(df[pos]['SSN'].values,df[pos]['dB'].values)
ax.text(0.05, 0.9, r'(c) $r_L$ = {:0.2f}'.format(r_i[0]) + r'; $r_S$ = {:0.2f}'.format(r_s[0]), 
     transform = ax.transAxes)

ax = plt.subplot(222)
for obs in plot_observatories: 
    ax.plot(df['SSN'], df[obs + '_dV_eclip'], '.', label = obs)
ax.set_title('Ecliptic', fontsize = 16)
ax.set_ylim((0,250))
ax.set_xlim((0,250))
pos = np.isfinite(df['dV_eclip'].values)
r_i = pearsonr(df[pos]['SSN'].values,df[pos]['dV_eclip'].values)
r_s = spearmanr(df[pos]['SSN'].values,df[pos]['dV_eclip'].values)
ax.text(0.05, 0.9, r'(b) $r_L$ = {:0.2f}'.format(r_i[0]) + r'; $r_S$ = {:0.2f}'.format(r_s[0]), 
     transform = ax.transAxes)

ax = plt.subplot(224)
for obs in plot_observatories: 
    ax.plot(df['SSN'], df[obs + '_dB_eclip'], '.', label = obs)
ax.set_xlabel('SSN', fontsize = 16)
ax.set_xlim((0,250))
ax.set_ylim((0,2.7))
pos = np.isfinite(df['dB_eclip'].values)
r_i = pearsonr(df[pos]['SSN'].values,df[pos]['dB_eclip'].values)
r_s = spearmanr(df[pos]['SSN'].values,df[pos]['dB_eclip'].values)
ax.text(0.05, 0.9, r'(d) $r_L$ = {:0.2f}'.format(r_i[0]) + r'; $r_S$ = {:0.2f}'.format(r_s[0]), 
     transform = ax.transAxes)

handles, labels = ax.get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', framealpha=1, ncol=4)

#scatter plots
fig = plt.figure()

ax = plt.subplot(221)
for obs in plot_observatories: 
    ax.plot(df['SAI'], df[obs + '_dV_global'], '.', label = obs)
#ax.plot(ssn,dV,'k.')
ax.set_ylabel(r'$\Delta$ V [km/s]', fontsize = 16)
ax.set_title('Global', fontsize = 16)
ax.set_ylim((0,250))
ax.set_xlim((0,1))
pos = ((np.isfinite(df['dV'].values)) & (np.isfinite(df['SAI'].values)))
r_i = pearsonr(df[pos]['SAI'].values,df[pos]['dV'].values)
r_s = spearmanr(df[pos]['SAI'].values,df[pos]['dV'].values)
ax.text(0.05, 0.9, r'$r_L$ = {:0.2f}'.format(r_i[0]) + r'; $r_S$ = {:0.2f}'.format(r_s[0]), 
     transform = ax.transAxes)

ax = plt.subplot(223)
for obs in plot_observatories: 
    ax.plot(df['SAI'], df[obs + '_dB_global'], '.', label = obs)
ax.set_ylabel(r'$\Delta B_R$ [nT]', fontsize = 16)
ax.set_xlabel('SAI', fontsize = 16)
ax.set_xlim((0,1))
ax.set_ylim((0,2.7))
pos = ((np.isfinite(df['dB'].values)) & (np.isfinite(df['SAI'].values)))
r_i = pearsonr(df[pos]['SAI'].values,df[pos]['dB'].values)
r_s = spearmanr(df[pos]['SAI'].values,df[pos]['dB'].values)
ax.text(0.05, 0.9, r'$r_L$ = {:0.2f}'.format(r_i[0]) + r'; $r_S$ = {:0.2f}'.format(r_s[0]), 
     transform = ax.transAxes)

ax = plt.subplot(222)
for obs in plot_observatories: 
    ax.plot(df['SAI'], df[obs + '_dV_eclip'], '.', label = obs)
ax.set_title('Ecliptic', fontsize = 16)
ax.set_ylim((0,250))
ax.set_xlim((0,1))
pos = ((np.isfinite(df['dV_eclip'].values)) & (np.isfinite(df['SAI'].values)))
r_i = pearsonr(df[pos]['SAI'].values,df[pos]['dV_eclip'].values)
r_s = spearmanr(df[pos]['SAI'].values,df[pos]['dV_eclip'].values)
ax.text(0.05, 0.9, r'$r_L$ = {:0.2f}'.format(r_i[0]) + r'; $r_S$ = {:0.2f}'.format(r_s[0]), 
     transform = ax.transAxes)

ax = plt.subplot(224)
for obs in plot_observatories: 
    ax.plot(df['SAI'], df[obs + '_dB_eclip'], '.', label = obs)
ax.set_xlabel('SAI', fontsize = 16)
ax.set_xlim((0,1))
ax.set_ylim((0,2.7))
pos = ((np.isfinite(df['dB_eclip'].values)) & (np.isfinite(df['SAI'].values)))
r_i = pearsonr(df[pos]['SAI'].values,df[pos]['dB_eclip'].values)
r_s = spearmanr(df[pos]['SAI'].values,df[pos]['dB_eclip'].values)
ax.text(0.05, 0.9, r'$r_L$ = {:0.2f}'.format(r_i[0]) + r'; $r_S$ = {:0.2f}'.format(r_s[0]), 
     transform = ax.transAxes)

handles, labels = ax.get_legend_handles_labels()
fig.legend(handles, labels, loc='right', framealpha=1)
# <codecell> box'n'whiskers plots

plt.figure()
colors = ['k', 'b', 'r']

ax = plt.subplot(221)
box = ax.boxplot([df['dV'].dropna().values, 
                  df.loc[mask_min,'dV'].dropna().values, 
                  df.loc[mask_max,'dV'].dropna().values],
           notch=True, patch_artist=True,showfliers=False,whis=1.5)
for patch, median, color in zip(box['boxes'], box['medians'], colors):
    patch.set(facecolor = color)
    patch.set(color = color)
    median.set(color='w') 
ax.set_ylim((0,200))
ax.set_ylabel(r'$<|\Delta V|>_{CR}$ [km/s]', fontsize = 16)
ax.get_xaxis().set_ticks([])
ax.set_title('Global', fontsize = 16)
ax.text(0.03, 0.05, r'(a)', transform = ax.transAxes)



ax = plt.subplot(222)
box = ax.boxplot([df['dV_eclip'].dropna().values, 
                  df.loc[mask_min,'dV_eclip'].dropna().values, 
                  df.loc[mask_max,'dV_eclip'].dropna().values],
           notch=True, patch_artist=True,showfliers=False,whis=1.5)
for patch, median, color in zip(box['boxes'], box['medians'], colors):
    patch.set(facecolor = color)
    patch.set(color = color)
    median.set(color='w')
ax.set_ylim((0,200))
ax.get_xaxis().set_ticks([])
ax.set_title('Ecliptic', fontsize = 16)
ax.text(0.03, 0.05, r'(b)', transform = ax.transAxes)

ax = plt.subplot(223)
box = ax.boxplot([df['dB'].dropna().values, 
                  df.loc[mask_min,'dB'].dropna().values, 
                  df.loc[mask_max,'dB'].dropna().values],
           notch=True, patch_artist=True,showfliers=False,whis=1.5)
for patch, median, color in zip(box['boxes'], box['medians'], colors):
    patch.set(facecolor = color)
    patch.set(color = color)
    median.set(color='w')
ax.set_ylim((0,2))
ax.set_ylabel(r'$<|\Delta B_R|>_{CR}$ [nT]', fontsize = 16)
ax.get_xaxis().set_ticks([])
ax.text(0.03, 0.05, r'(c)', transform = ax.transAxes)

ax.legend([box["boxes"][0], box["boxes"][1],box["boxes"][2]], 
          ["All data", "SAI < 0.5", "SAI > 0.5"], 
          loc='upper center', bbox_to_anchor=(0.5,1.1), framealpha = 1)  

    
ax = plt.subplot(224)
box = ax.boxplot([df['dB_eclip'].dropna().values, 
                  df.loc[mask_min,'dB_eclip'].dropna().values, 
                  df.loc[mask_max,'dB_eclip'].dropna().values],
           notch=True, patch_artist=True,showfliers=False,whis=1.5)
for patch, median, color in zip(box['boxes'], box['medians'], colors):
    patch.set(facecolor = color)
    patch.set(color = color)
    median.set(color='w')
ax.set_ylim((0,2))
ax.get_xaxis().set_ticks([])
ax.text(0.03, 0.05, r'(d)', transform = ax.transAxes)


# <codecell> lat plots
#==============================


lat_centres = 90 - lats[0:len(lats)-1]*180/np.pi -2.5


import matplotlib.gridspec as gridspec
fig = plt.figure()
gs = gridspec.GridSpec(3, 4)

ax = fig.add_subplot(gs[0, 0:3])
plt.plot(df['datetime'],df['SSN']/200, 'k', label = 'SSN/200')
plt.plot(df['datetime'],df['SAI'], 'r', label = 'SAI')
ax.get_xaxis().set_ticklabels([])
plt.ylabel('SSN')
plt.legend(fontsize = 14, bbox_to_anchor=(0.3, .85), framealpha = 1)
ax.text(0.02,.9,'(a)', fontsize = 14, transform=ax.transAxes, backgroundcolor = 'w')
ax.set_xlim((datetime(1975,1,1), datetime(2020,1,1)))
ax.set_ylim((0,1.5))

ax = fig.add_subplot(gs[1, 0:3])
im_v = ax.pcolor(df['datetime'], lat_centres, dV_lat.T, norm=plt.Normalize(0,150))
ax.set_yticks([-90, -45, 0, 45, 90])
ax.get_xaxis().set_ticklabels([])
ax.text(0.02,1.05,'(b)' + r'$<|\Delta V|>_{CR}$ [km/s]                                                           ',
        fontsize = 14, transform=ax.transAxes, backgroundcolor = 'w')
ax.set_xlim((datetime(1975,1,1), datetime(2020,1,1)))
ax.set_ylim((-90,90))
ax.set_ylabel('Latitude [deg]')

#sort out the bloody colourbar
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
axins = inset_axes(ax,
                    width="100%",  # width = 50% of parent_bbox width
                    height="10%",  # height : 5%
                    loc='upper right',
                    bbox_to_anchor=(0.45, 0.65, 0.5, 0.5),
                    bbox_transform=ax.transAxes,
                    borderpad=0,)

#ax.plot([0, 360],[7.5, 7.5],'w--'); ax.plot([0, 360],[-7.5, -7.5],'w--');

cb = fig.colorbar(im_v, cax = axins, orientation = 'horizontal',  pad = -0.1)
cb.ax.tick_params(labelsize=12)
#cb.ax.set_title(r'$<|\Delta V|>_{CR}$ [km/s]', fontsize = 14)


ax =  fig.add_subplot(gs[2, 0:3])
im_b = ax.pcolor(df['datetime'], lat_centres, dB_lat.T, norm=plt.Normalize(0,1.5))
ax.set_yticks([-90, -45, 0, 45, 90])
for tick in ax.get_xticklabels():
            tick.set_rotation(90)
ax.set_xlim((datetime(1975,1,1), datetime(2020,1,1)))
ax.set_ylim((-90,90))
ax.text(0.02,1.05,'(d)' + r'$<|\Delta B_R|>_{CR}$ [nT]                                                     ' , 
        fontsize = 14, transform=ax.transAxes, backgroundcolor = 'w')
ax.set_ylabel('Latitude [deg]')

axins = inset_axes(ax,
                    width="100%",  # width = 50% of parent_bbox width
                    height="10%",  # height : 5%
                    loc='upper right',
                    bbox_to_anchor=(0.45, 0.65, 0.5, 0.5),
                    bbox_transform=ax.transAxes,
                    borderpad=0,)

#ax.plot([0, 360],[7.5, 7.5],'w--'); ax.plot([0, 360],[-7.5, -7.5],'w--');

cb = fig.colorbar(im_b, cax = axins, orientation = 'horizontal',  pad = -0.1)
cb.ax.tick_params(labelsize=12)


ax = fig.add_subplot(gs[1, 3])
ax.plot(dV_lat_all[:,1], lat_centres, 'k', label = 'All data')
ax.plot(dV_lat_min[:,1], lat_centres, 'b', label = 'SAI < 0.5')
ax.plot(dV_lat_max[:,1], lat_centres, 'r', label = 'SAI >= 0.5')
ax.set_yticks([-90, -45, 0, 45, 90])
ax.get_yaxis().set_ticklabels([])
ax.set_xlabel(r'$<|\Delta V|>$ [km/s]')
ax.xaxis.tick_top(); ax.xaxis.set_label_position('top') 
ax.set_ylim((-90,90))
ax.set_xlim((0,150))
ax.text(0.05,0.9,'(c)', fontsize = 14, transform=ax.transAxes, backgroundcolor = 'w')
ax.legend(loc = 'upper right', fontsize = 14, bbox_to_anchor=(1.15, 1.8), framealpha = 1)

ax = fig.add_subplot(gs[2, 3])
ax.plot(dB_lat_all[:,1], lat_centres, 'k', label = 'All data')
ax.plot(dB_lat_min[:,1], lat_centres, 'b', label = 'SAI < 0.5')
ax.plot(dB_lat_max[:,1], lat_centres, 'r', label = 'SAI >= 0.5')
ax.set_yticks([-90, -45, 0, 45, 90])
ax.get_yaxis().set_ticklabels([])
ax.set_xlabel(r'$<|\Delta B_R|>$ [nT]')
ax.set_ylim((-90,90))
ax.set_xlim((0,1.5))
ax.text(0.05,0.9,'(e)', fontsize = 14, transform=ax.transAxes, backgroundcolor = 'w')

#ax = fig.add_subplot(gs[0, 2])
#ax.axis('off')
#ax.text(0.2,0.39,'      --- All data', color = 'k', fontsize = 14, transform=ax.transAxes, backgroundcolor = 'w')
#ax.text(0.2,0.27,'   --- SAI < 0.5', color = 'b', fontsize = 14, transform=ax.transAxes, backgroundcolor = 'w')
#ax.text(0.2,0.15,'--- SAI >= 0.5', color = 'r', fontsize = 14, transform=ax.transAxes, backgroundcolor = 'w')

# ax = fig.add_subplot(gs[1, 3])
# ax.plot(r_dV_lat[:,0], lat_centres, 'k', label = r'$r_L$')
# ax.plot(r_dV_lat[:,1], lat_centres, 'r', label = r'$r_S$')
# ax.set_yticks([-90, -45, 0, 45, 90])
# ax.get_yaxis().set_ticklabels([])
# ax.get_xaxis().set_ticklabels([])
# ax.xaxis.tick_top(); ax.xaxis.set_label_position('top') 
# ax.set_xlabel(r'Correlation' +'\n' + r'($<|\Delta V|>$, SSN)')
# ax.set_xlim((-0.5,1))
# ax.plot([0,0],[-90,90],'k--')
# ax.set_ylim((-90,90))
# ax.text(0.05,0.9,'(d)', fontsize = 14, transform=ax.transAxes, backgroundcolor = 'w')
# ax.legend(loc = 'upper right', fontsize = 14, bbox_to_anchor=(1, 1.7), framealpha = 1)

# ax = fig.add_subplot(gs[2, 3])
# ax.plot(r_dB_lat[:,0], lat_centres, 'k', label = 'r_L')
# ax.plot(r_dB_lat[:,1], lat_centres, 'r', label = 'r_S')
# ax.set_yticks([-90, -45, 0, 45, 90])
# ax.get_yaxis().set_ticklabels([])
# ax.set_xlim((-0.5,1))
# ax.plot([0,0],[-90,90],'k--')
# ax.set_xlabel(r'Correlation' +'\n' + r'($<|\Delta B_R|>$, SSN)')
# ax.set_ylim((-90,90))
# ax.text(0.05,0.9,'(g)', fontsize = 14, transform=ax.transAxes, backgroundcolor = 'w')

#ax = fig.add_subplot(gs[0, 3])
#ax.axis('off')
#ax.text(0.6,0.27, r' --- $r_L$', color = 'k', fontsize = 14, transform=ax.transAxes, backgroundcolor = 'w')
#ax.text(0.6,0.15, r' --- $r_S$', color = 'r', fontsize = 14, transform=ax.transAxes, backgroundcolor = 'w')

# <codecell> Compute CR averages of V and Br (not dV and dBr)
df_CR = pd.DataFrame()

lats = np.arange(2.5,180-2.499999,5) * np.pi/180
Nlats = len(lats)-1

for obs in observatories: 
    
    print('Processing ' + obs)
    
    B = np.ones((CRstop-CRstart))*np.nan
    V = np.ones((CRstop-CRstart))*np.nan
    B_eclip = np.ones((CRstop-CRstart))*np.nan
    V_eclip = np.ones((CRstop-CRstart))*np.nan
    B_lat = np.ones((CRstop-CRstart,Nlats))*np.nan
    V_lat = np.ones((CRstop-CRstart,Nlats))*np.nan
    Bsig = np.ones((CRstop-CRstart))*np.nan
    Vsig = np.ones((CRstop-CRstart))*np.nan
    Bsig_eclip = np.ones((CRstop-CRstart))*np.nan
    Vsig_eclip = np.ones((CRstop-CRstart))*np.nan
    Bsig_lat = np.ones((CRstop-CRstart,Nlats))*np.nan
    Vsig_lat = np.ones((CRstop-CRstart,Nlats))*np.nan
    
    nCR = np.ones((CRstop-CRstart))*np.nan
    ssn = np.ones((CRstop-CRstart))*np.nan
    sai = np.ones((CRstop-CRstart))*np.nan
    
    counter = 0 
    for CR in range(CRstart,CRstop):
        brfilepath = HelioMASdir + obs + '\\CR' + str(CR) + '\\br002.hdf'
        vrfilepath = HelioMASdir + obs + '\\CR' + str(CR) + '\\vr002.hdf'
       
        
        
        nCR[counter] = CR + 0.5
        
        smjd = htime.crnum2mjd(CR-1)
        fmjd = htime.crnum2mjd(CR+2)
        
        mask = (ssn_df['mjd'] >= smjd) & (ssn_df['mjd'] < fmjd)
        ssn[counter] = np.nanmean(ssn_df.loc[mask,'ssn'])
        sai[counter] = np.nanmean(ssn_df.loc[mask,'sai'])
        
        
        if (os.path.exists(brfilepath) == True): 
            
            #load the Vr data
            Vr, VXa, VXm, VXr = read_HelioMAS(vrfilepath)
            Vr = Vr * 481.0
            
            #load the Br data
            Br, BXa, BXm, BXr = read_HelioMAS(brfilepath) 
            Br = Br *  2.2e5
            

            
            #take the slice at r = 215 rS
            id_r_v = np.argmin(abs(VXr - 215*u.solRad))
            
            id_r_b = np.argmin(abs(BXr - 215*u.solRad))
            
            #find lats around teh ecliptic
            delta_lat = 7.25 * np.pi/180 *u.rad
            mask_v = ((VXm > np.pi/2*u.rad - delta_lat) & (VXm < np.pi/2*u.rad + delta_lat))
            mask_b = ((BXm > np.pi/2*u.rad - delta_lat) & (BXm < np.pi/2*u.rad + delta_lat))
            
            #compute the difference globally
            V[counter] = np.nanmean(abs(Vr[:,:,id_r_v]))
            B[counter] = np.nanmean(abs(Br[:,:,id_r_b]))
            Vsig[counter] = np.nanstd(abs(Vr[:,:,id_r_v]))
            Bsig[counter] = np.nanstd(abs(Br[:,:,id_r_b]))
            
            #compute the difference at the ecliptic
            V_eclip[counter] = np.nanmean(abs(Vr[:, mask_v, id_r_v]))
            B_eclip[counter] = np.nanmean(abs(Br[:, mask_b, id_r_b]))
            
            Vsig_eclip[counter] = np.nanstd(abs(Vr[:, mask_v, id_r_v]))
            Bsig_eclip[counter] = np.nanstd(abs(Br[:, mask_b, id_r_b]))
            
            
            #now find teh differences at each lat bin
            for ilat in range(0,Nlats):
                mask_v = ((VXm >= lats[ilat]*u.rad) & (VXm < lats[ilat+1]*u.rad))
                mask_b = ((BXm >= lats[ilat]*u.rad) & (BXm < lats[ilat+1]*u.rad))
            
                #compute the difference at the given lat band
                V_lat[counter,ilat] = np.nanmean(abs(Vr[:, mask_v, id_r_v] ))
                B_lat[counter,ilat] = np.nanmean(abs(Br[:, mask_b, id_r_b]))
                Vsig_lat[counter,ilat] = np.nanstd(abs(Vr[:, mask_v, id_r_v] ))
                Bsig_lat[counter,ilat] = np.nanstd(abs(Br[:, mask_b, id_r_b]))
                
            

        else:
            V[counter] = np.nan
            B[counter] = np.nan
            V_eclip[counter] = np.nan
            B_eclip[counter] = np.nan
            V_lat[counter,:] = np.nan
            B_lat[counter,:] = np.nan 
            
            Vsig[counter] = np.nan
            Bsig[counter] = np.nan
            Vsig_eclip[counter] = np.nan
            Bsig_eclip[counter] = np.nan
            Vsig_lat[counter,:] = np.nan
            Bsig_lat[counter,:] = np.nan 
    
        counter = counter + 1
    
    #add the data to the data frame
    df_CR['CR'] = nCR
    df_CR['SSN'] = ssn
    df_CR['SAI'] = sai
    df_CR[obs + '_V_global'] = V
    df_CR[obs + '_V_eclip'] = V_eclip
    df_CR[obs + '_B_global'] = B
    df_CR[obs + '_B_eclip'] = B_eclip
    df_CR[obs + '_Vsig_global'] = Vsig
    df_CR[obs + '_Vsig_eclip'] = Vsig_eclip
    df_CR[obs + '_Bsig_global'] = Bsig
    df_CR[obs + '_Bsig_eclip'] = Bsig_eclip
    for ilat in range(0,Nlats):
        df_CR[obs + '_B_lat_' + str(ilat)] = B_lat[:,ilat]
        df_CR[obs + '_V_lat_' + str(ilat)] = V_lat[:,ilat]
        df_CR[obs + '_Bsig_lat_' + str(ilat)] = Bsig_lat[:,ilat]
        df_CR[obs + '_Vsig_lat_' + str(ilat)] = Vsig_lat[:,ilat]

#convert to datetime
df_CR['datetime'] = htime.mjd2datetime(htime.crnum2mjd(df['CR'].values))

#take averages across all observatories
#observatories = ['hmi', 'mdi', 'solis', 'gong', 'mwo', 'wso', 'kpo']
df_CR['V'] = df_CR[['hmi_V_global','mdi_V_global','solis_V_global',
               'gong_V_global','mwo_V_global','wso_V_global', 
               'kpo_V_global']].mean(axis = 1, skipna=True)
df_CR['V_eclip'] = df_CR[['hmi_V_eclip','mdi_V_eclip','solis_V_eclip',
               'gong_V_eclip','mwo_V_eclip','wso_V_eclip', 
               'kpo_V_eclip']].mean(axis = 1, skipna=True)
df_CR['B'] = df_CR[['hmi_B_global','mdi_B_global','solis_B_global',
               'gong_B_global','mwo_B_global','wso_B_global', 
               'kpo_B_global']].mean(axis = 1, skipna=True)
df_CR['B_eclip'] = df_CR[['hmi_B_eclip','mdi_B_eclip','solis_B_eclip',
               'gong_B_eclip','mwo_B_eclip','wso_B_eclip', 
               'kpo_B_eclip']].mean(axis = 1, skipna=True)
df_CR['Vsig'] = df_CR[['hmi_Vsig_global','mdi_Vsig_global','solis_Vsig_global',
               'gong_Vsig_global','mwo_Vsig_global','wso_Vsig_global', 
               'kpo_Vsig_global']].mean(axis = 1, skipna=True)
df_CR['Vsig_eclip'] = df_CR[['hmi_Vsig_eclip','mdi_Vsig_eclip','solis_Vsig_eclip',
               'gong_Vsig_eclip','mwo_Vsig_eclip','wso_Vsig_eclip', 
               'kpo_Vsig_eclip']].mean(axis = 1, skipna=True)
df_CR['Bsig'] = df_CR[['hmi_Bsig_global','mdi_Bsig_global','solis_Bsig_global',
               'gong_Bsig_global','mwo_Bsig_global','wso_Bsig_global', 
               'kpo_Bsig_global']].mean(axis = 1, skipna=True)
df_CR['Bsig_eclip'] = df_CR[['hmi_Bsig_eclip','mdi_Bsig_eclip','solis_Bsig_eclip',
               'gong_Bsig_eclip','mwo_Bsig_eclip','wso_Bsig_eclip', 
               'kpo_Bsig_eclip']].mean(axis = 1, skipna=True)

for ilat in range(0,Nlats):
    l = str(ilat)
    df_CR['B_lat_' + str(ilat)] = df_CR[['hmi_B_lat_' + l,'mdi_B_lat_' + l,'solis_B_lat_' + l,
                   'gong_B_lat_' + l,'mwo_B_lat_' + l,'wso_B_lat_' + l, 
                   'kpo_B_lat_' + l]].mean(axis = 1, skipna=True)
    df_CR['V_lat_' + str(ilat)] = df_CR[['hmi_V_lat_' + l,'mdi_V_lat_' + l,'solis_V_lat_' + l,
                   'gong_V_lat_' + l,'mwo_V_lat_' + l,'wso_V_lat_' + l, 
                   'kpo_V_lat_' + l]].mean(axis = 1, skipna=True)
    df_CR['Bsig_lat_' + str(ilat)] = df_CR[['hmi_Bsig_lat_' + l,'mdi_Bsig_lat_' + l,'solis_Bsig_lat_' + l,
                   'gong_Bsig_lat_' + l,'mwo_Bsig_lat_' + l,'wso_Bsig_lat_' + l, 
                   'kpo_Bsig_lat_' + l]].mean(axis = 1, skipna=True)
    df_CR['Vsig_lat_' + str(ilat)] = df_CR[['hmi_Vsig_lat_' + l,'mdi_Vsig_lat_' + l,'solis_Vsig_lat_' + l,
                   'gong_Vsig_lat_' + l,'mwo_Vsig_lat_' + l,'wso_Vsig_lat_' + l, 
                   'kpo_Vsig_lat_' + l]].mean(axis = 1, skipna=True)
    


mask_max = ((df_CR['SAI'] >= 0.5))
mask_min = ((df_CR['SAI'] < 0.5)) 

Nall = len(np.isfinite(df_CR['V']))   
Nmin = len(np.isfinite(df_CR.loc[mask_min,'V']))
Nmax = len(np.isfinite(df_CR.loc[mask_max,'V']))

#compute the metrics
print('HelioMAS')
print('<|V|> global: ' + str(round(np.nanmean(df_CR['V']),2)) + 
      ' +/- ' + str(round(np.nanstd(df_CR['V'])/np.sqrt(Nall),2)))
print('<|V|> ecliptic: ' + str(round(np.nanmean(df_CR['V_eclip']),2)) + 
      ' +/- ' + str(round(np.nanstd(df_CR['V_eclip'])/np.sqrt(Nall),2)))

print('<|Br|> global: ' + str(round(np.nanmean(df_CR['B']),2)) + 
      ' +/- ' + str(round(np.nanstd(df_CR['B'])/np.sqrt(Nall),2)))
print('<|Br|> ecliptic: ' + str(round(np.nanmean(df_CR['B_eclip']),2)) + 
      ' +/- ' + str(round(np.nanstd(df_CR['B_eclip'])/np.sqrt(Nall),2)))
print('')

# print('<|dV|> global (low SAI): ' + str(round(np.nanmean(df.loc[mask_min,'dV']),2)) + 
#       ' +/- ' + str(round(np.nanstd(df.loc[mask_min,'dV'])/np.sqrt(Nmin),2)))
# print('<|dV|> global (high SAI): ' + str(round(np.nanmean(df.loc[mask_max,'dV']),2)) + 
#       ' +/- ' + str(round(np.nanstd(df.loc[mask_max,'dV'])/np.sqrt(Nmax),2)))
# print('<|dBr|> global (low SAI): ' + str(round(np.nanmean(df.loc[mask_min,'dB']),2)) + 
#       ' +/- ' + str(round(np.nanstd(df.loc[mask_min,'dB'])/np.sqrt(Nmin),2)))
# print('<|dBr|> global (high SAI): ' + str(round(np.nanmean(df.loc[mask_max,'dB']),2)) + 
#       ' +/- ' + str(round(np.nanstd(df.loc[mask_max,'dB'])/np.sqrt(Nmax),2)))
# print('')

# print('<|dV|> ecliptic (low SAI): ' + str(round(np.nanmean(df.loc[mask_min,'dV_eclip']),2)) + 
#       ' +/- ' + str(round(np.nanstd(df.loc[mask_min,'dV_eclip'])/np.sqrt(Nmin),2)))
# print('<|dV|> ecliptic (high SAI): ' + str(round(np.nanmean(df.loc[mask_max,'dV_eclip']),2)) + 
#       ' +/- ' + str(round(np.nanstd(df.loc[mask_max,'dV_eclip'])/np.sqrt(Nmax),2)))
# print('<|dBr|> ecliptic (low SAI): ' + str(round(np.nanmean(df.loc[mask_min,'dB_eclip']),2)) + 
#       ' +/- ' + str(round(np.nanstd(df.loc[mask_min,'dB_eclip'])/np.sqrt(Nmin),2)))
# print('<|dBr|> ecliptic (high SAI): ' + str(round(np.nanmean(df.loc[mask_max,'dB_eclip']),2)) + 
#       ' +/- ' + str(round(np.nanstd(df.loc[mask_max,'dB_eclip'])/np.sqrt(Nmax),2)))
# print('')

#post 1995 values?

#export the lat composite as an array
B_lat = np.ones((CRstop-CRstart,Nlats))*np.nan
V_lat = np.ones((CRstop-CRstart,Nlats))*np.nan

V_lat_min = np.ones((Nlats,2))*np.nan
V_lat_max = np.ones((Nlats,2))*np.nan
V_lat_all = np.ones((Nlats,2))*np.nan

B_lat_min = np.ones((Nlats,2))*np.nan
B_lat_max = np.ones((Nlats,2))*np.nan
B_lat_all = np.ones((Nlats,2))*np.nan

Bsig_lat = np.ones((CRstop-CRstart,Nlats))*np.nan
Vsig_lat = np.ones((CRstop-CRstart,Nlats))*np.nan

Vsig_lat_min = np.ones((Nlats,2))*np.nan
Vsig_lat_max = np.ones((Nlats,2))*np.nan
Vsig_lat_all = np.ones((Nlats,2))*np.nan

Bsig_lat_min = np.ones((Nlats,2))*np.nan
Bsig_lat_max = np.ones((Nlats,2))*np.nan
Bsig_lat_all = np.ones((Nlats,2))*np.nan

#r_dV_lat = np.ones((Nlats,2))*np.nan
#r_dB_lat = np.ones((Nlats,2))*np.nan

#compute the delat and correlation at each latitude
for ilat in range(0,Nlats):
    B_lat[:,ilat] = df_CR['B_lat_' + str(ilat)].to_numpy()
    V_lat[:,ilat] = df_CR['V_lat_' + str(ilat)].to_numpy()
    
    Bsig_lat[:,ilat] = df_CR['Bsig_lat_' + str(ilat)].to_numpy()
    Vsig_lat[:,ilat] = df_CR['Vsig_lat_' + str(ilat)].to_numpy()
    
    mask_max = ((df_CR['SAI'] >= 0.5))
    mask_min = ((df_CR['SAI'] < 0.5))
    
    V_lat_all[ilat,1] = np.nanmean(V_lat[:,ilat])
    B_lat_all[ilat,1] = np.nanmean(B_lat[:,ilat])
    
    V_lat_max[ilat,1] = np.nanmean(V_lat[mask_max,ilat])
    B_lat_max[ilat,1] = np.nanmean(B_lat[mask_max,ilat])
    
    V_lat_min[ilat,1] = np.nanmean(V_lat[mask_min,ilat])
    B_lat_min[ilat,1] = np.nanmean(B_lat[mask_min,ilat])
    
    Vsig_lat_all[ilat,1] = np.nanmean(Vsig_lat[:,ilat])
    Bsig_lat_all[ilat,1] = np.nanmean(Bsig_lat[:,ilat])
    
    Vsig_lat_max[ilat,1] = np.nanmean(Vsig_lat[mask_max,ilat])
    Bsig_lat_max[ilat,1] = np.nanmean(Bsig_lat[mask_max,ilat])
    
    Vsig_lat_min[ilat,1] = np.nanmean(Vsig_lat[mask_min,ilat])
    Bsig_lat_min[ilat,1] = np.nanmean(Bsig_lat[mask_min,ilat])
    
    # #compute correlations
    # pos = ((np.isfinite(dV_lat[:,ilat]) & (np.isfinite(df['SSN'].values))))
    # r_i = pearsonr(df[pos]['SSN'].values, dV_lat[pos,ilat])
    # r_s = spearmanr(df[pos]['SSN'].values, dV_lat[pos,ilat])
    # r_dV_lat[ilat,0] = r_i[0]
    # r_dV_lat[ilat,1] = r_s[0]
    # pos = ((np.isfinite(dB_lat[:,ilat]) & (np.isfinite(df['SSN'].values))))
    # r_i = pearsonr(df[pos]['SSN'].values, dB_lat[pos,ilat])
    # r_s = spearmanr(df[pos]['SSN'].values, dB_lat[pos,ilat])
    # r_dB_lat[ilat,0] = r_i[0]
    # r_dB_lat[ilat,1] = r_s[0]
    
    
# <codecell> lat plots - V and Vsig
#==============================

lat_centres = 90 - lats[0:len(lats)-1]*180/np.pi -2.5


import matplotlib.gridspec as gridspec
fig = plt.figure()
gs = gridspec.GridSpec(3, 4)

ax = fig.add_subplot(gs[0, 0:3])
plt.plot(df_CR['datetime'],df_CR['SSN']/200, 'k', label = 'SSN/200')
plt.plot(df_CR['datetime'],df_CR['SAI'], 'r', label = 'SAI')
ax.get_xaxis().set_ticklabels([])
plt.ylabel('SSN')
plt.legend(fontsize = 14, bbox_to_anchor=(0.3, .85), framealpha = 1)
ax.text(0.02,.9,'(a)', fontsize = 14, transform=ax.transAxes, backgroundcolor = 'w')
ax.set_xlim((datetime(1975,1,1), datetime(2020,1,1)))
ax.set_ylim((0,1.5))

ax = fig.add_subplot(gs[1, 0:3])
im_v = ax.pcolor(df_CR['datetime'], lat_centres, V_lat.T, norm=plt.Normalize(300,750))
ax.set_yticks([-90, -45, 0, 45, 90])
ax.get_xaxis().set_ticklabels([])
ax.text(0.02,1.05,'(b)' + r'$<|V|>_{CR}$ [km/s]                                                      ',
        fontsize = 14, transform=ax.transAxes, backgroundcolor = 'w')
ax.set_xlim((datetime(1975,1,1), datetime(2020,1,1)))
ax.set_ylim((-90,90))
ax.set_ylabel('Latitude [deg]')
#ax.plot([0, 360],[7.5, 7.5],'w--'); ax.plot([0, 360],[-7.5, -7.5],'w--');
#cb = plt.colorbar(im_v); cb.ax.tick_params(labelsize=12)
#cb.ax.set_title(r'$<|\Delta V|>_{CR}$ [km/s]', fontsize = 14)

axins = inset_axes(ax,
                    width="100%",  # width = 50% of parent_bbox width
                    height="10%",  # height : 5%
                    loc='upper right',
                    bbox_to_anchor=(0.45, 0.65, 0.5, 0.5),
                    bbox_transform=ax.transAxes,
                    borderpad=0,)

#ax.plot([0, 360],[7.5, 7.5],'w--'); ax.plot([0, 360],[-7.5, -7.5],'w--');

cb = fig.colorbar(im_v, cax = axins, orientation = 'horizontal',  pad = -0.1)
cb.ax.tick_params(labelsize=12)

ax =  fig.add_subplot(gs[2, 0:3])
im_b = ax.pcolor(df_CR['datetime'], lat_centres, Vsig_lat.T, norm=plt.Normalize(0,150))
ax.set_yticks([-90, -45, 0, 45, 90])
for tick in ax.get_xticklabels():
            tick.set_rotation(90)
ax.set_xlim((datetime(1975,1,1), datetime(2020,1,1)))
ax.set_ylim((-90,90))
ax.text(0.02,1.05,'(d)' + r'$<|\sigma_V|>_{CR}$ [km/s]                                               ' ,
        fontsize = 14, transform=ax.transAxes, backgroundcolor = 'w')
ax.set_ylabel('Latitude [deg]')

axins = inset_axes(ax,
                    width="100%",  # width = 50% of parent_bbox width
                    height="10%",  # height : 5%
                    loc='upper right',
                    bbox_to_anchor=(0.45, 0.65, 0.5, 0.5),
                    bbox_transform=ax.transAxes,
                    borderpad=0,)

#ax.plot([0, 360],[7.5, 7.5],'w--'); ax.plot([0, 360],[-7.5, -7.5],'w--');

cb = fig.colorbar(im_b, cax = axins, orientation = 'horizontal',  pad = -0.1)
cb.ax.tick_params(labelsize=12)


ax = fig.add_subplot(gs[1, 3])
ax.plot(V_lat_all[:,1], lat_centres, 'k', label = 'All data')
ax.plot(V_lat_min[:,1], lat_centres, 'b', label = 'SAI < 0.5')
ax.plot(V_lat_max[:,1], lat_centres, 'r', label = 'SAI >= 0.5')
ax.set_yticks([-90, -45, 0, 45, 90])
ax.get_yaxis().set_ticklabels([])
ax.set_xlabel(r'$<|V|>$ [km/s]')
ax.xaxis.tick_top(); ax.xaxis.set_label_position('top') 
ax.set_ylim((-90,90))
ax.set_xlim((300,750))
ax.text(0.05,0.9,'(c)', fontsize = 14, transform=ax.transAxes, backgroundcolor = 'w')
ax.legend(loc = 'upper right', fontsize = 14, bbox_to_anchor=(1.15, 1.8), framealpha = 1)

ax = fig.add_subplot(gs[2, 3])
ax.plot(Vsig_lat_all[:,1], lat_centres, 'k', label = 'All data')
ax.plot(Vsig_lat_min[:,1], lat_centres, 'b', label = 'SAI < 0.5')
ax.plot(Vsig_lat_max[:,1], lat_centres, 'r', label = 'SAI >= 0.5')
ax.set_yticks([-90, -45, 0, 45, 90])
ax.get_yaxis().set_ticklabels([])
ax.set_xlabel(r'$<|\sigma_V|>$ [km/s]')
ax.set_ylim((-90,90))
ax.set_xlim((0,150))
ax.text(0.05,0.9,'(e)', fontsize = 14, transform=ax.transAxes, backgroundcolor = 'w')

# <codecell> lat plots - B and Bsig
#==============================

lat_centres = 90 - lats[0:len(lats)-1]*180/np.pi -2.5


import matplotlib.gridspec as gridspec
fig = plt.figure()
gs = gridspec.GridSpec(3, 4)

ax = fig.add_subplot(gs[0, 0:3])
plt.plot(df_CR['datetime'],df_CR['SSN']/200, 'k', label = 'SSN/200')
plt.plot(df_CR['datetime'],df_CR['SAI'], 'r', label = 'SAI')
ax.get_xaxis().set_ticklabels([])
plt.ylabel('SSN')
plt.legend(fontsize = 14, bbox_to_anchor=(0.3, .85), framealpha = 1)
ax.text(0.02,.9,'(a)', fontsize = 14, transform=ax.transAxes, backgroundcolor = 'w')
ax.set_xlim((datetime(1975,1,1), datetime(2020,1,1)))
ax.set_ylim((0,1.5))

ax = fig.add_subplot(gs[1, 0:3])
im_v = ax.pcolor(df_CR['datetime'], lat_centres, B_lat.T, norm=plt.Normalize(0,3))
ax.set_yticks([-90, -45, 0, 45, 90])
ax.get_xaxis().set_ticklabels([])
ax.text(0.02,1.05,'(b)' + r'$<|B_R|>_{CR}$ [nT]                                                            ',
        fontsize = 14, transform=ax.transAxes, backgroundcolor = 'w')
ax.set_xlim((datetime(1975,1,1), datetime(2020,1,1)))
ax.set_ylim((-90,90))
ax.set_ylabel('Latitude [deg]')
#ax.plot([0, 360],[7.5, 7.5],'w--'); ax.plot([0, 360],[-7.5, -7.5],'w--');
#cb = plt.colorbar(im_v); cb.ax.tick_params(labelsize=12)
#cb.ax.set_title(r'$<|\Delta V|>_{CR}$ [km/s]', fontsize = 14)

axins = inset_axes(ax,
                    width="100%",  # width = 50% of parent_bbox width
                    height="10%",  # height : 5%
                    loc='upper right',
                    bbox_to_anchor=(0.45, 0.65, 0.5, 0.5),
                    bbox_transform=ax.transAxes,
                    borderpad=0,)

#ax.plot([0, 360],[7.5, 7.5],'w--'); ax.plot([0, 360],[-7.5, -7.5],'w--');

cb = fig.colorbar(im_v, cax = axins, orientation = 'horizontal',  pad = -0.1)
cb.ax.tick_params(labelsize=12)




ax =  fig.add_subplot(gs[2, 0:3])
im_b = ax.pcolor(df_CR['datetime'], lat_centres, Bsig_lat.T, norm=plt.Normalize(0,1.2))
ax.set_yticks([-90, -45, 0, 45, 90])
for tick in ax.get_xticklabels():
            tick.set_rotation(90)
ax.set_xlim((datetime(1975,1,1), datetime(2020,1,1)))
ax.set_ylim((-90,90))
ax.text(0.02,1.05,'(d)' + r'$<|\sigma_B|>_{CR}$ [nT]                                                     ' , 
        fontsize = 14, transform=ax.transAxes, backgroundcolor = 'w')
ax.set_ylabel('Latitude [deg]')

axins = inset_axes(ax,
                    width="100%",  # width = 50% of parent_bbox width
                    height="10%",  # height : 5%
                    loc='upper right',
                    bbox_to_anchor=(0.45, 0.65, 0.5, 0.5),
                    bbox_transform=ax.transAxes,
                    borderpad=0,)

#ax.plot([0, 360],[7.5, 7.5],'w--'); ax.plot([0, 360],[-7.5, -7.5],'w--');

cb = fig.colorbar(im_b, cax = axins, orientation = 'horizontal',  pad = -0.1)
cb.ax.tick_params(labelsize=12)




ax = fig.add_subplot(gs[1, 3])
ax.plot(B_lat_all[:,1], lat_centres, 'k', label = 'All data')
ax.plot(B_lat_min[:,1], lat_centres, 'b', label = 'SAI < 0.5')
ax.plot(B_lat_max[:,1], lat_centres, 'r', label = 'SAI >= 0.5')
ax.set_yticks([-90, -45, 0, 45, 90])
ax.get_yaxis().set_ticklabels([])
ax.set_xlabel(r'$<|B_R|>$ [km/s]')
ax.xaxis.tick_top(); ax.xaxis.set_label_position('top') 
ax.set_ylim((-90,90))
ax.set_xlim((0,3))
ax.text(0.05,0.9,'(c)', fontsize = 14, transform=ax.transAxes, backgroundcolor = 'w')
ax.legend(loc = 'upper right', fontsize = 14, bbox_to_anchor=(1.15, 1.8), framealpha = 1)

ax = fig.add_subplot(gs[2, 3])
ax.plot(Bsig_lat_all[:,1], lat_centres, 'k', label = 'All data')
ax.plot(Bsig_lat_min[:,1], lat_centres, 'b', label = 'SAI < 0.5')
ax.plot(Bsig_lat_max[:,1], lat_centres, 'r', label = 'SAI >= 0.5')
ax.set_yticks([-90, -45, 0, 45, 90])
ax.get_yaxis().set_ticklabels([])
ax.set_xlabel(r'$<|\sigma_B|>$ [nT]')
ax.set_ylim((-90,90))
ax.set_xlim((0,1.2))
ax.text(0.05,0.9,'(e)', fontsize = 14, transform=ax.transAxes, backgroundcolor = 'w')