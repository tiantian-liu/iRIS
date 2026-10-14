# all the parameters required for running sim_fragmentRISK_3D_MC_seeds_backup.py
# copyright: 10.10.2026 T. Liu @mfn Berlin

import math
import numpy as np
# fragment size min = 1 cm
vc=5e5 #5e13
n = 20
binL = np.zeros(n)
binL[0] = (vc*3/(4*math.pi))**(1/3)*2
binL[1::] = (1.5**(np.arange(n-2,-1,-1)))*1E-2 # in m

# fragment size min = 0.1 cm
vc=5e5 #5e13
n = 25
binLs = np.zeros(n)
binLs[0] = (vc*3/(4*math.pi))**(1/3)*2
binLs[1::] = (1.5**(np.arange(n-2,-1,-1)))*1E-3 # in m

binL2 = np.zeros(28)
binL2[0] = (vc*3/(4*math.pi))**(1/3)*2
binL2[1::] = (1.5**(np.arange(26,-1,-1)))

binV = 4/3*math.pi*(binL/2)**3

params = {"name":'', # name the specific runing case
          "R": 1737.4, # the lunar radius
          "g": 1.625 * 1e-3, # the gravitational acceleration of the Moon, km/s2
          "F_E2M":1.38, # Earth impact flux relative to the Moon
          #"beta0":-80*math.pi/180, #lat
          #"arfa0":90*math.pi/180, #lon
          "n_layer":30, # the crust is divided into n_layers
          "dxLat":10, # to consider the spatial variation of impact flux, the lat is divided with resolution of dxLat

         # "r_lim":[0,10000],#[0,300,1500,10000],
          "rangeLat":[-90*math.pi/180,-80*math.pi/180], # range of high-resolution cells
          "nlatH": 500, # within high-resolution region, Lat is divided into nlatH
          "nlonH": 500,  # within high-resolution region, Lon is divided into nlonH

          #"fixed_phi":0,
          #"catalysm_time":[3.8, 4.2],
          #"catalysm_mag":2,
          #"catalysm_style":'none',
         # "mode": 'constant',
         # "unitS_smallcraters":0.01, #km
          #"dN_small":100e3,
          #"smallDep":1, #m

          # "simulation_mode":'true_time',#'one_impact'
          #"one_impact":0,
          #"only_basins":0,
          #"one_location":0,
          #"sampling_site":0,
          #"events":0,
          #"real_time":1,
          #"flag_add_basin":0,

          "t0":4.31002, # the time period, Ga
          "Dmin":20, # minimum crater diameter
          "Dmax":300, # maximum crater diameter
          "fixed_diameter":0, # assume all craters have same size
          #"flg_minimum_t0":0,

###### the main input for iRIS project #######
          #"from_Vol_to_velocity":0,
          "b_CUM":-1.2, # exponent of scaling laws of size-frequency distribution of ejected fragments
          "b_CUM_steep":-1.8,
          "b_CUM_gentle":-0.7,
          #"b_CUM_vary":0,
          #"b_MAXm":0.8,
          "u_vemax":0.374+0.012*np.exp(-3.175*(0.8-0.884)),#calculate miu Prieur+ 2017
          "C_vemax":0.46, #  0.46,
          "Xmax":1.2, #xmin/a, starting launch position of ejecta
          "C_vecum":0.03,# constant of the scaling law: ejecta with Ve and the cumulative volume, Housen & Holsapple 2011
          #"plateau":20, # 100
          "U": 15, #impact velocity, km/s
          "L": 1, #projectile diameter, km
          "K3":0.4, # Luther+ 2018
          #"ae": 452,
          #"be": 0.735, #constant for calculating Vexc2
          "C_D": 1.54, # constant for calculating transient crater diameter, Liu+2022 jgr
          "D_beta": 0.165,# constant for calculating transient crater diameter, Liu+2022 jgr
          "pp":3320, #density of projectile, kg/m3
          "pt":2940, #density of impact, targetskg/m3
         # "H0":0.1, # km

          'vc':vc, # volume of the bottom layer cells, make it big to prevent impact penetration
          'binL': binL, # bin the size of ejecta fragments
          'fmaxL':1, #constant of the scaling law to calculate the maximum ejecta fragments, Liu+2022 EPSL
          #'fmaxL_picewise':0,
          #'HmaxINSITU':100e3,

          "dxEj":0.1, # resolution of bin ejecta
         # "Rtrack": 10, #track ejecta in 10 radii

          #"noINSITU":1,
          #"mega_e": 0.005,
          #"slopeInsitu":0.4,
          #"ratioDep":0.05,

          #"idx_target_basin":0, # sap:0  Se:12  Humorum:22  Imbrium:27


          #"basin_events": 1,
         # "only_basin":0,
         # "basin_events_number": 0,
          #"basin_melt": 0,
         # "basin_file": ' ',
          "name":' '}

            #0
variants = [{"name":'test',
             "N_res":[1e10,1e5,1e4],#[1e10,1e5,1e4],#[1.5e6,1.6e4,3e3],
             "r_lim":[0,100,1000,10000],#[0,300,1500,10000],
             "dxEj":0.005,
             "L": 0.1,
             "t0":1*1E-7, # in Ga
             "Dmin":1E-3,  #in km
             "Dmax":1E-1,  # in km
             "binL":binLs,
             #"u_vemax":0.5,
             #"C_vecum":0.05,
             #"K3":0.9,
             #"H0":0.01, # km
             #"Rtrack": 20,

             "n_layer":60, #200
             "path_save":"/data/user/tiantians/Python/results/ejectaRISK/backup_output/"
            },
            #1  for plt convenience
            {"name":'',
             "binL":binL2,
            }]

