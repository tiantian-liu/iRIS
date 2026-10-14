#INPUT: change the parameters using parameters.update(var[0]). It update the paramters in sim_batch_fragmentRISK_3D_backup.py
#OUTPUT:  arrayV, the volume of ejecta
#         arrayVe, the corresponding speed of ejecta
#         arrayN, the size-frequency of ejecta

# copyright: 10.10.2026 T. Liu @mfn Berlin

import numpy as np
import os
import pickle
import math

import function_fragmentRISK_3D_backup as f
import sim_batch_fragmentRISK_3D_backup as batch
parameters = batch.params
var = batch.variants
parameters.update(var[0]) # update the parameters according the first input of 'variants' in sim_batch_fragmentRISK_3D_backup.py

path_save = parameters["path_save"]
R = parameters["R"]

# generate cells to record the ejecta
p1 = f.point_no_gradual_res([1e4]) # the global surface is divided by 1E4 uniformly distributed cells
pH =f.point_region_Hresolution(parameters["rangeLat"]) # the region within "rangeLat" having high resolution
# merge p1 and pH. Within the high-resolution region, the low-resolution cells are replaces with high-resolution cells.
# p records the position of cells, sc records the size of cells
[p, sc] = f.point_merge_Hresolution(p1,pH,parameters["rangeLat"])

# once p.txt and sc.txt are created, save them and load them directly next time, which can save modeling time
f.save_variation(p, path_save + "p.txt")
F1 = open(path_save+'/p.txt', 'rb')
p = pickle.load(F1)
F1.close()

f.save_variation(sc, path_save + "sc.txt")
F1 = open(path_save+'/sc.txt', 'rb')
sc = pickle.load(F1)
F1.close()

ResLim = np.min(sc)**0.5 # the size of the high resolution cells
X,Y,Z = p[:,1],p[:,2],p[:,3]
num_location = len(p[:,0])

b_CUM = parameters["b_CUM"]
C_vemax,C_vecum,u_vemax = parameters["C_vemax"], parameters["C_vecum"], parameters["u_vemax"]
Xmax = parameters["Xmax"]
L, g, U = parameters["L"], parameters["g"], parameters["U"]
#ae, be = parameters["ae"], parameters["be"]
D_beta, C_D = parameters["D_beta"], parameters["C_D"]
pp, pt = parameters["pp"],parameters["pt"]
dxEj = parameters["dxEj"]

binL = parameters["binL"]
binV = 4/3*math.pi*(binL/2)**3
binM = pt*binV
binC = 1/binM**b_CUM
nL = len(binL)

# generate the sequence of impact events with random positions and the site-frequency distribution follows Brown et al. 2002
seedStart = 500 # provide the random seeds, any number is acceptable
seeds_beta1 = np.arange(seedStart,600) # here is 600-1-500 = 99 times of MC
seeds_beta2 = seeds_beta1*11
seeds_arfa1 = np.arange(seedStart,600)*3
seeds_arfa2 = seeds_arfa1*7
seeds_ylog = np.arange(seedStart,600)*13

# impact inform contain: [0]: lon (radian), [1]: lat (radian), [2]: Dt (km), [3]:Df (km), [4]: formation time (year)
for i_seed in range(len(seeds_beta1)):
    impact_info0 = f.cal_impactS_info_seeds(seeds_beta1[i_seed],seeds_beta2[i_seed],seeds_arfa1[i_seed],seeds_arfa2[i_seed],seeds_ylog[i_seed])
    f.save_variation(impact_info0 , path_save + "impact_info/" + "impact_info_" + str(seedStart+i_seed) + ".txt")

    idx=np.where(impact_info0[:,1]<-80*math.pi/180)[0] # to save modeling time, setting: only the impact occurs at lat<-80 is considered.
    impact_info = np.array(impact_info0[idx,:])

    n_layer = parameters["n_layer"]
    arrayEmpty = np.ones(num_location,dtype=np.int32)*(n_layer-2) # record which layer is empty to save the upcoming ejecta

    arrayV = np.zeros([n_layer,num_location],dtype=np.float32) # record volume of ejecta fragments
    arrayV[-1,:] = 1.e9 #vc
    arrayN = np.zeros([n_layer,num_location,nL],dtype=np.float32) # record size-frequency distribution of ejecta fragments
    arrayN[-1,:,0] = 1.0
    arrayVe = np.zeros([n_layer,num_location],dtype=np.float32) # record speed/velocity of ejecta fragments


    nimp = len(impact_info[:,0])
    for idx_imp in range(len(impact_info[:,0])): #range(len(impact_info[:,0]))
        print('idx_imp: ', idx_imp,'/',nimp)#D_t = 0.8*C_D/1.6**D_beta * L**(1-D_beta)*U**(2*D_beta)/g**D_beta

        R_t = 1e3*impact_info[idx_imp,3]   # in meter
        R_f = 1e3*impact_info[idx_imp,2]   # in meter
        D_t = R_t*2*1E-3
        Lp = (D_t/(0.8*C_D/1.6**D_beta) * g**D_beta / U**(2*D_beta) )**(1/(1-D_beta))  #the diameter of projectile in km

        betai, arfai = impact_info[idx_imp,1], impact_info[idx_imp,0] #lat lon

###################################### excavation ######################################
        ####### calculate fragment size-frequency distribution of excavated materials  ######
        r_min, r_max = 0, R_t/1e3
        Vexc1 = math.pi*R_t**3/15 # the simplified method to calculate total excavation volume Liu+ 2019,2020,2021
        Vtc = 0.0765*Lp**2.34 *(1.62*0.001)**(-0.66)*parameters['U']**1.3*1E9
        Vproj = 4 / 3 * math.pi * (Lp / 2 * 1E3) ** 3 #in m3
        # different ways to calculate the total excavated volume were test,
        # and the method of Liu* 2019,2020 shows best results if calculating the corresponding thickness distribution of ejecta deposit
        Vexc=Vexc1

        filt_p_excavation = f.filt_points(X, Y, Z, r_min, r_max, beta0=betai, arfa0=arfai) # find the cells for the region of excavation zone
        Vexc_perP = Vexc/len(filt_p_excavation[:, 0])
        nfrag_collect = np.zeros(nL)
        # remove the materials over the excavation zone
        for k in range(len(filt_p_excavation[:,0])):
            index0 = int(filt_p_excavation[k][0])
            nfrag_temp = np.zeros(nL)
            cum_V_index = np.cumsum(arrayV[:,index0])
            if cum_V_index[-1] < Vexc_perP:
                nfrag_temp += sum(arrayN[:,index0],0)
                if k == 0:
                    nfrag_collect = np.array(nfrag_temp)
                else:
                    nfrag_collect = np.vstack([nfrag_collect, nfrag_temp])

                arrayV[:,index0] = 0
                arrayV[-1,index0] = 1.e9
                arrayN[:,index0] = 0
                arrayN[-1,index0,0] = 1
                arrayEmpty[index0] = n_layer-2
                continue
            n_nearest_value = f.locate_depth(Vexc_perP, arrayV[:,index0]) # calculate the index of the deepest layer of excavation zone
            if arrayV[0, index0] == 0 and n_nearest_value < 0:
                print("excavation: fraction, divided by zero")
                os._exit(0)
            fraction = Vexc_perP/arrayV[0,index0] if n_nearest_value<0 else (Vexc_perP-cum_V_index[n_nearest_value])/arrayV[(n_nearest_value+1),index0]
            if fraction>1:
                print("excavation:fraction>1")
                os._exit(0)
            weight_temp = np.ones(n_nearest_value + 2)
            weight_temp[n_nearest_value+1] = fraction

            nfrag_temp = np.sum(np.array(arrayN[0:(n_nearest_value + 1), index0]),0)
            nfrag_temp += np.array(arrayN[(n_nearest_value + 1), index0]) * fraction

            if k == 0:
                nfrag_collect = np.array(nfrag_temp)
            else:
                nfrag_collect = np.vstack([nfrag_collect,nfrag_temp])

            arrayN[0:(n_nearest_value+1),index0] = 0
            arrayN[(n_nearest_value+1),index0] *= (1-fraction)
            arrayV[0:(n_nearest_value+1),index0] = 0
            arrayV[(n_nearest_value+1),index0] *= (1-fraction)
            arrayEmpty[index0] = n_nearest_value

        idx_exc = np.where(arrayV[-1,:] == 0)[0]
        if len(idx_exc)>0:
            print("excavation: bottom layer == 0")
            os._exit(0)

        # get the size-frequency distribution of excavated materials
        nfrag_layer = np.sum(nfrag_collect,axis=0) if len(filt_p_excavation[:,0])>1 else np.array(nfrag_collect)

###################################### in situ fragmentation, such a process is not considered in ejecta-risk model ##########
###################################### fragmentation and ejection of ejecta ################################################
        a = Lp/2
        K2 = (parameters["K3"]*(parameters["g"]*R_t*1e-3)**0.5) /  U/   (R_t*1e-3/a)**(-1/parameters["u_vemax"]) # Luther* 2018
        dxEj = parameters["dxEj"]
        [bin_ve, x] = f.cal_ve(K2,U, R_t*1e-3, a, parameters["u_vemax"], dxEj) # calculate the speed and launch position of ejeata
        Vole_cum = Vproj*parameters["C_vecum"]*(bin_ve/U)**(-3*parameters["u_vemax"]) # ejecta with Ve and the cumulative volume, Housen & Holsapple 2011
        rmax = parameters["fmaxL"] * 0.1 * (R_f * 2) ** (2 / 3) / 2 # maximum fragment size

        if len(np.where(nfrag_layer > 0)[0]) == 1 and np.where(nfrag_layer > 0)[0][0] == 0:
            Lmax = rmax * 2
        else: # if the existing fragment size is samller than the calculated maximum fragment size, rmax is replaced with the existing fragment size
            idx = np.where(nfrag_layer[1::]>0)[0]
            binL0 = binL[idx[0]+1]
            LmaxEXC = np.max(binL0)
            Lmax = rmax * 2 if LmaxEXC > rmax * 2 else LmaxEXC
        i_maxL = np.argmin(abs(binL - Lmax))
        C0 = binC[i_maxL]

        # calculate deposition of ejecta including distance, volume, and fragment size
        [Vexc,L_ejecta, bin_ejectaV, bin_ejectaVn] = f.cal_ejecta_from_excavation(Vole_cum,bin_ve,x, i_maxL, C0)

        sort_arg = np.argsort(L_ejecta)
        sort_arg2 = np.argsort(-bin_ejectaV)
        bin_ve = bin_ve[sort_arg]
        L_ejecta = L_ejecta[sort_arg]
        bin_ejectaV = bin_ejectaV[sort_arg2]
        bin_ejectaVn = bin_ejectaVn[sort_arg2]

        ## the calculated ejecta distribution needs to be further processed
        ## since the bin-annulus/ring of calculated ejecta is smaller than the model resolution
        dL = np.array(L_ejecta[1::]) - np.array(L_ejecta[0:-1])
        idx = np.where(dL > ResLim)[0]

        bin_ve_merge = np.array(bin_ve)
        L_ejecta_merge = np.array(L_ejecta)
        bin_ejectaVn_merge = np.array(bin_ejectaVn)
        bin_ejectaV_merge = np.array(bin_ejectaV)
        if len(idx) > 0:
            imerge = idx[0] - 1
            if imerge >= 0:
                bin_ve_merge = np.array(bin_ve[imerge::])
                bin_ve_merge[0] = np.average(bin_ve[0:imerge])

                L_ejecta_merge = np.array(L_ejecta[imerge::])
                L_ejecta_merge[0] = np.average(L_ejecta[0:imerge])

                bin_ejectaV_merge = np.array(bin_ejectaV[imerge::])
                bin_ejectaV_merge[0] = np.sum(bin_ejectaV[0:imerge])

                bin_ejectaVn_merge = np.array(bin_ejectaVn[imerge::, :])
                bin_ejectaVn_merge[0, :] = np.sum(bin_ejectaVn[0:imerge, :], 0)

        fnum = np.sum(np.sum(bin_ejectaVn_merge, 0) * binV) / Vexc
        bin_ejectaVn_merge = np.array(bin_ejectaVn_merge) / fnum
        bin_ejectaVn_merge2 = np.sum(bin_ejectaVn_merge, 0)

        if np.min(np.array(bin_ejectaVn_merge)) < 0:
            print("bin_ejectaVn<0")
            os._exit(0)
        idx_max = int(parameters["R"] * math.pi // (R_t / 1e3 * dxEj)) - 2
        if idx_max < len(bin_ejectaV_merge):
            bin_ejectaV_merge[idx_max] = np.sum(bin_ejectaV_merge[idx_max::])
            bin_ejectaV_merge[idx_max + 1::] = np.zeros(len(bin_ejectaV_merge[idx_max + 1::]))
            bin_ejectaVn_merge[idx_max] = np.sum(bin_ejectaVn_merge[idx_max::, :], 0)
            bin_ejectaVn_merge[idx_max + 1::] = np.zeros((len(bin_ejectaVn_merge[idx_max + 1::, 0]), nL))
        else:
            idx_max = len(bin_ejectaV_merge) - 1

        for i in range(0,idx_max+1):
            if bin_ve_merge[i] > 2.38: # escape ejecta
                continue
            r_min, r_max = L_ejecta_merge[i], L_ejecta_merge[i + 1]
            r_max = R * math.pi if r_max > R * math.pi else r_max
            filt_p_ejecta = f.filt_points_RISK(p, r_min, r_max, beta0=betai, arfa0=arfai)
            idx_cell = np.array(filt_p_ejecta[:, 0], dtype=np.int32)
            nc = len(idx_cell)
            si = np.sum(sc[idx_cell])
            empty_layer = np.array(arrayEmpty[idx_cell], dtype=np.int32)
            arrayV[empty_layer, idx_cell] = bin_ejectaV_merge[i] / nc
            fnumV = np.sum(arrayV[empty_layer, idx_cell]) / bin_ejectaV_merge[i]
            if fnumV > 0:
                arrayV[empty_layer, idx_cell] = np.array(arrayV[empty_layer, idx_cell]) / fnumV

            arrayN[empty_layer, idx_cell] = np.array(bin_ejectaVn_merge[i]) / nc
            arrayVe[empty_layer, idx_cell] = bin_ve_merge[i]
            fnumN = np.sum(np.sum(arrayN[empty_layer, idx_cell], 0) * binV) / bin_ejectaV_merge[i]
            arrayN[empty_layer, idx_cell] = np.array(arrayN[empty_layer, idx_cell]) / fnumN

            arrayEmpty[idx_cell] -= 1

            idx_empty = np.where(arrayEmpty <= 0)[0]
            if len(idx_empty) > 0:
                print("no empty layer")
                os._exit(0)
        idx_eje = np.where(arrayV[-1, :] == 0)[0]
        if len(idx_eje) > 0:
            print("ejecta: bottom layer == 0")
            os._exit(0)


    f.save_variation(arrayV, path_save+"arrayV/" + "arrayV_" + str(seedStart+i_seed) + ".txt")
    f.save_variation(arrayVe, path_save + "arrayVe/" + "arrayVe_" + str(seedStart+i_seed) + ".txt")

    f.save_variation(arrayN[:,0:(int(num_location/10)+1),:],path_save + "nfrag/" + "nfrag1_" + str(seedStart+i_seed) + ".txt")
    f.save_variation(arrayN[:,(int(num_location/10)+1):(int(num_location/10*2)+1),:], path_save + "nfrag/" + "nfrag2_" + str(seedStart+i_seed) + ".txt")
    f.save_variation(arrayN[:,(int(num_location/10*2)+1):(int(num_location/10*3)+1),:], path_save + "nfrag/" + "nfrag3_" + str(seedStart+i_seed) + ".txt")
    f.save_variation(arrayN[:,(int(num_location/10*3)+1):(int(num_location/10*4)+1),:], path_save + "nfrag/" + "nfrag4_" + str(seedStart+i_seed) + ".txt")
    f.save_variation(arrayN[:,(int(num_location/10*4)+1):(int(num_location/10*5)+1),:], path_save + "nfrag/" + "nfrag5_" + str(seedStart+i_seed) + ".txt")
    f.save_variation(arrayN[:,(int(num_location/10*5)+1):(int(num_location/10*6)+1),:], path_save + "nfrag/" + "nfrag6_" + str(seedStart+i_seed) + ".txt")
    f.save_variation(arrayN[:,(int(num_location/10*6)+1):(int(num_location/10*7)+1),:], path_save + "nfrag/" + "nfrag7_" + str(seedStart+i_seed) + ".txt")
    f.save_variation(arrayN[:,(int(num_location/10*7)+1):(int(num_location/10*8)+1),:], path_save + "nfrag/" + "nfrag8_" + str(seedStart+i_seed) + ".txt")
    f.save_variation(arrayN[:,(int(num_location/10*8)+1):(int(num_location/10*9)+1),:], path_save + "nfrag/" + "nfrag9_" + str(seedStart+i_seed) + ".txt")
    f.save_variation(arrayN[:,(int(num_location/10*9)+1)::,:], path_save + "nfrag/" + "nfrag10_" + str(seedStart+i_seed) + ".txt")

    print(i_seed)

