# all the functions required for running sim_fragmentRISK_3D_MC_seeds_backup.py
# copyright: 10.10.2026 T. Liu @mfn Berlin

import numpy as np
import pickle
import math
from sympy import *
from random import *
import function_fragmentRISK_3D_backup as f
import sim_batch_fragmentRISK_3D_backup as batch
parameters = batch.params
var = batch.variants
parameters.update(var[0])
Dmin, Dmax = parameters["Dmin"], parameters["Dmax"]
fixed_diameter = parameters["fixed_diameter"]
#beta0,arfa0 = parameters["beta0"],parameters["arfa0"]
R = parameters["R"]
vc = parameters["vc"]

b_CUM = parameters["b_CUM"]
pp, pt = parameters["pp"],parameters["pt"]
dxEj = parameters["dxEj"]
binL = parameters["binL"]
binV = 4/3*math.pi*(binL/2)**3
binM = pt*binV


def uniform_points(N):
    phi = (5**0.5 - 1) / 2
    n = np.arange(1,(N+1),1)
    Z = R*(2*n - 1)/N - 1*R
    X = (R**2 - Z**2)**0.5*np.cos(2*math.pi*n*phi)
    Y = (R**2 - Z**2)**0.5*np.sin(2*math.pi*n*phi)
    return X,Y,Z

### P1:  generate uniformly distributed N0 points over the global surface ###
def point_no_gradual_res(N0):
    N = int(N0[0])
    p = np.zeros((N,6))
    p[:,0] = np.arange(N)
    [X_tmp, Y_tmp, Z_tmp] = uniform_points(N)
    p[:,1], p[:,2], p[:,3] = X_tmp, Y_tmp, Z_tmp
    p[:,4] = np.arcsin(Z_tmp/R)
    p[:,5] = np.arctan(Y_tmp/X_tmp)
    X0_idx = np.where(p[:,1] == 0)[0]
    if len(X0_idx):
        p[X0_idx, 5] = 0 if p[X0_idx, 2] > 0 else math.pi
        print('abnormal value modified')
    idx_x_negative = np.where(p[:, 1] < 0)[0]
    for i in idx_x_negative:
        p[i, 5] += math.pi
    return p

### PH:  generate uniformly distributed points (high resolution) over the studying region ###
def point_region_Hresolution(rangeLat):
    nlat,nlon = parameters["nlatH"], parameters["nlonH"]
    lat = np.linspace(rangeLat[0],rangeLat[1],nlat)
    lon = np.linspace(-math.pi/2,1.5*math.pi, nlon)
    N = nlat*nlon
    p = np.zeros((N,6))
    p[:,0] = np.arange(1,N+1)
    count  = 0
    for i in range(nlat):
        lati = lat[i]
        for j in range(nlon):
            loni = lon[j]
            Zi = R*np.sin(lati)
            Xi = R*np.cos(loni)
            Yi = R*sin(loni)
            p[count,1] = np.array(Xi)
            p[count,2] = np.array(Yi)
            p[count,3] = np.array(Zi)
            p[count,4] = lati
            p[count,5] = loni
            count+=1


    return p

### merge P1 & PH ###
def point_merge_Hresolution(p,pH,rangeLat):
    sc1 = 4 * math.pi * R ** 2 / len(p[:, 0])
    s2 =2*math.pi*R**2*np.abs(np.sin(rangeLat[0])-np.sin(rangeLat[1]))
    sc2 = s2/len(pH[:,0])

    idx = np.where((p[:,4]>=rangeLat[0]) & (p[:,4]<=rangeLat[1]))[0]
    p=np.delete(p,idx,axis=0)
    npL = len(p[:,0])
    npH = len(pH[:,0])
    sc = np.ones(npL+npH)
    sc[0:npL] *= sc1
    sc[npL::] *= sc2
    pH[:,0] += np.max(p[:,0])
    pMerge = np.vstack((p,pH))
    pMerge[:,0] = np.arange(len(pMerge[:,0]))
    return pMerge, sc

## consider impacts < 10 m, where Neukum PF cannot be applied. ##
def PF_S(D): # Brown et al. 2002
    c, d = 1.568+0.03, 2.7-0.08 # c +-0.03  d +-0.08
    log_N = c-d*np.log10(D)
    return log_N

def Inverse_D_S(y_log,x_range,tol):
    count = 0
    xm = sum(x_range)/2
    y_temp = PF_S(10**xm)
    while abs(y_temp-y_log) > tol:
        if count >99:
            break

        if y_temp < y_log:
            x_range[1] = xm
        else:
            x_range[0] = xm

        xm = sum(x_range)/2
        y_temp = PF_S(10**xm)

        count = count + 1

    return 10**xm

def cal_R_t(R_f,basin_R_t):
    D_f = 2*R_f
    DQ = 21
    R_t = R_f / 1.25 if R_f < DQ/2 else (D_f*DQ**0.13/1.17)**(1/1.13)/2
    if basin_R_t:
        R_t=basin_R_t
    return R_t

## calculate impact info and consider spatial variation ##
def cal_impactS_info():

    t0 = parameters["t0"]*1E9 #in year
    logN = PF_S(Dmin*1e3)  # D should in meter
    count = np.int32(t0*10**logN/parameters["F_E2M"])

    bin_Lat = np.arange(0,100,parameters["dxLat"])*math.pi/180
    bin_S = 2*math.pi*R**2*np.abs(np.sin(bin_Lat[0:-1])-np.sin(bin_Lat[1::]))
    rflux = np.linspace(0.93,1.05,9)
    Nfrac = bin_S*rflux/np.sum(bin_S*rflux)
    record_beta = []
    record_arfa = []
    rdm_beta1 = Random();    rdm_beta1.seed(1)
    rdm_beta2 = Random();    rdm_beta2.seed(11)
    rdm_arfa1 = Random();    rdm_arfa1.seed(2)
    rdm_arfa2 = Random();    rdm_arfa2.seed(22)
    for i in range(len(Nfrac)):
        aLat, bLat = bin_Lat[i]*180/math.pi, bin_Lat[i+1]*180/math.pi
        counti = int(count/2*Nfrac[i])
        record_beta=record_beta+[rdm_beta1.uniform(aLat, bLat) for i in range(counti)]
        record_beta=record_beta+[rdm_beta2.uniform(-bLat, -aLat) for i in range(counti)]

        record_arfa=record_arfa+[rdm_arfa1.uniform(-180, 180) for i in range(counti)]
        record_arfa=record_arfa+[rdm_arfa2.uniform(-180, 180) for i in range(counti)]
    arfa_arr = np.array(record_arfa) * math.pi / 180
    beta_arr = np.array(record_beta) * math.pi / 180
    count = len(arfa_arr)

    N_Dmin, N_Dmax = 10 ** PF_S(Dmin * 1E3), 10 ** PF_S(Dmax * 1E3)
    if parameters["fixed_diameter"]:
        record_Df = np.full(count, parameters["fixed_diameter"])
        record_diameter = np.full(count, parameters["fixed_diameter"])

    else:
        rdm_ylog = Random();         rdm_ylog.seed(9)
        record_ylog = [rdm_ylog.uniform(0, 1.0) for i in range(count)]
        record_Df, record_diameter = [], []
        impact_count = 0
        record_time = np.linspace(0,t0,count)
        for i in record_time:
            y_log = record_ylog[impact_count] * (N_Dmin - N_Dmax) + N_Dmax
            R_f = f.Inverse_D_S(np.log10(y_log), np.log10([Dmin*1E3, Dmax*1E3]), 0.01) / 2
            R_t = f.cal_R_t(R_f, 0)
            record_Df.append(R_f * 2)
            record_diameter.append(round(2 * R_t, 1))
            impact_count += 1
    impact_info = np.zeros((len(arfa_arr), 5))
    impact_info[:, 0] = np.array(arfa_arr)
    impact_info[:, 1] = np.array(beta_arr)
    impact_info[:, 2] = np.array(record_diameter) * 1e-3 #in km
    impact_info[:, 3] = np.array(record_Df)*1e-3 # in km
    impact_info[:, 4] = np.array(record_time) # in year
    sort_arg = np.argsort(-impact_info[:, 4])
    impact_info = impact_info[sort_arg]


    #f.save_variation(impact_info, parameters["path"] + "impact_info_"+str(t0)+".txt")
    return impact_info

## calculate impact info and consider spatial variation using varying random seeds##
def cal_impactS_info_seeds(seeds_beta1,seeds_beta2,seeds_arfa1,seeds_arfa2,seeds_ylog):

    t0 = parameters["t0"]*1E9 #in year
    logN = PF_S(Dmin*1e3)  # D should in meter
    count = np.int32(t0*10**logN/parameters["F_E2M"])

    bin_Lat = np.arange(0,100,parameters["dxLat"])*math.pi/180
    bin_S = 2*math.pi*R**2*np.abs(np.sin(bin_Lat[0:-1])-np.sin(bin_Lat[1::]))
    rflux = np.linspace(0.93,1.05,9)
    Nfrac = bin_S*rflux/np.sum(bin_S*rflux)
    record_beta = []
    record_arfa = []
    rdm_beta1 = Random();    rdm_beta1.seed(1)
    rdm_beta1 = Random();    rdm_beta1.seed(int(seeds_beta1))
    rdm_beta2 = Random();    rdm_beta2.seed(int(seeds_beta2))
    rdm_arfa1 = Random();    rdm_arfa1.seed(int(seeds_arfa1))
    rdm_arfa2 = Random();    rdm_arfa2.seed(int(seeds_arfa2))
    for i in range(len(Nfrac)):
        aLat, bLat = bin_Lat[i]*180/math.pi, bin_Lat[i+1]*180/math.pi
        counti = int(count/2*Nfrac[i])
        record_beta=record_beta+[rdm_beta1.uniform(aLat, bLat) for i in range(counti)]
        record_beta=record_beta+[rdm_beta2.uniform(-bLat, -aLat) for i in range(counti)]

        record_arfa=record_arfa+[rdm_arfa1.uniform(-180, 180) for i in range(counti)]
        record_arfa=record_arfa+[rdm_arfa2.uniform(-180, 180) for i in range(counti)]
    arfa_arr = np.array(record_arfa) * math.pi / 180
    beta_arr = np.array(record_beta) * math.pi / 180
    count = len(arfa_arr)

    N_Dmin, N_Dmax = 10 ** PF_S(Dmin * 1E3), 10 ** PF_S(Dmax * 1E3)
    if fixed_diameter:
        record_Df = np.full(count, fixed_diameter)
        record_diameter = np.full(count, fixed_diameter)

    else:
        rdm_ylog = Random();         rdm_ylog.seed(int(seeds_ylog))
        record_ylog = [rdm_ylog.uniform(0, 1.0) for i in range(count)]
        record_Df, record_diameter = [], []
        impact_count = 0
        record_time = np.linspace(0,t0,count)
        for i in record_time:
            y_log = record_ylog[impact_count] * (N_Dmin - N_Dmax) + N_Dmax
            R_f = f.Inverse_D_S(np.log10(y_log), np.log10([Dmin*1E3, Dmax*1E3]), 0.01) / 2
            R_t = f.cal_R_t(R_f, 0)
            record_Df.append(R_f * 2)
            record_diameter.append(round(2 * R_t, 1))
            impact_count += 1
    impact_info = np.zeros((len(arfa_arr), 5))
    impact_info[:, 0] = np.array(arfa_arr)
    impact_info[:, 1] = np.array(beta_arr)
    impact_info[:, 2] = np.array(record_diameter) * 1e-3 #in km
    impact_info[:, 3] = np.array(record_Df)*1e-3 # in km
    impact_info[:, 4] = np.array(record_time) # in year
    sort_arg = np.argsort(-impact_info[:, 4])
    impact_info = impact_info[sort_arg]


    #f.save_variation(impact_info, parameters["path"] + "impact_info_"+str(t0)+".txt")
    return impact_info

# search the necessary cells for the region of interest
def filt_points(X,Y,Z,r_min,r_max,beta0,arfa0):
    theta = 2*r_max/R
    if theta> math.pi:
        theta = math.pi
    x0 = R * np.cos(beta0) * np.cos(arfa0)
    y0 = R * np.cos(beta0) * np.sin(arfa0)
    z0 = R * np.sin(beta0)
    r_uplim = 2 * R * np.sin(theta / 2)


    filt_x = np.where((X >= (x0 - r_uplim)) & (X <= (x0 + r_uplim)))[0]
    filt_y = np.where((Y >= (y0 - r_uplim)) & (Y <= (y0 + r_uplim)))[0]
    filt_z = np.where((Z >= (z0 - r_uplim)) & (Z <= (z0 + r_uplim)))[0]
    filt_xyz = np.intersect1d(np.intersect1d(filt_x, filt_y), filt_z)

    if len(filt_xyz) == 0:
        theta = 300*math.pi/180
        r_uplim = 2 * R * np.sin(theta / 2)
        filt_x = np.where((X >= (x0 - r_uplim)) & (X <= (x0 + r_uplim)))[0]
        filt_y = np.where((Y >= (y0 - r_uplim)) & (Y <= (y0 + r_uplim)))[0]
        filt_z = np.where((Z >= (z0 - r_uplim)) & (Z <= (z0 + r_uplim)))[0]
        filt_xyz = np.intersect1d(np.intersect1d(filt_x, filt_y), filt_z)

    len_xyz = len(filt_xyz)
    filt_idx = np.zeros((len_xyz,6))
    filt_idx[:,0] = filt_xyz
    filt_idx[:,1] = X[filt_xyz]
    filt_idx[:,2] = Y[filt_xyz]
    filt_idx[:,3] = Z[filt_xyz]
    filt_idx[:,4] = np.arcsin(filt_idx[:,3]/R)
    filt_idx[:,5] = np.arctan(filt_idx[:,2]/filt_idx[:,1])
    X0_idx = np.where(filt_idx[:,1] == 0)[0]

    if len(X0_idx):
        filt_idx[X0_idx, 5] = 0 if filt_idx[X0_idx, 2]>0 else math.pi
        print('abnormal value modified')
    idx_x_negative = np.where(filt_idx[:,1]<0)[0]

    for i in idx_x_negative:
        filt_idx[i, 5] += math.pi

    beta,arfa = filt_idx[:,4], filt_idx[:,5]
    dist = R*np.arccos(np.cos(beta0)*np.cos(beta)*np.cos(arfa0 - arfa) + np.sin(beta0)*np.sin(beta))
    idx = np.where((dist>=r_min) & (dist<=r_max))[0]
    if len(idx) == 0:
        idx = [np.argmin(dist)]
        p_info = np.zeros((1, 6))
    else:
        p_info = np.zeros((len(idx),6))
    p_info[:,0] = filt_idx[idx,0]
    p_info[:,1] = filt_idx[idx,1]
    p_info[:,2] = filt_idx[idx,2]
    p_info[:,3] = filt_idx[idx,3]
    p_info[:,4] = filt_idx[idx,4]  #beta
    p_info[:,5] = filt_idx[idx,5]  #arfa

    return p_info

def filt_points_RISK(p,r_min,r_max,beta0,arfa0):

    beta,arfa = p[:,4], p[:,5]
    dist = R*np.arccos(np.cos(beta0)*np.cos(beta)*np.cos(arfa0 - arfa) + np.sin(beta0)*np.sin(beta))
    idx = np.where((dist>=r_min) & (dist<=r_max))[0]
    if len(idx) == 0:
        idx = [np.argmin(dist)]
        p_info = np.zeros((1, 3))
    else:
        p_info = np.zeros((len(idx),3))
    p_info[:,0] = p[idx,0]
    p_info[:,1] = p[idx,4]#beta
    p_info[:,2] = p[idx,5]#arfa

    return p_info

def calc_cum_depth(thickness):
    cum_depth = np.cumsum(thickness, dtype=float)
    return cum_depth

# calculate the index of the layer of interest
def locate_depth(depth,thickness):
    cum_depth = calc_cum_depth(thickness)
    n_nearest_value = (cum_depth < depth).argmin() #min(L, key=lambda x: abs(target-x))
    return n_nearest_value - 1

# calculate the speed and launch position of ejeata
def cal_ve(K2,U, R_t, a, miu, dx):

    x = np.arange(1.2*a,R_t,dx*R_t)
    ve = U * K2*(x/a)**(-1/miu)
    return ve, x

# calculate deposition of ejecta including distance (L), volume (bin_ejectaV), and fragment size(bin_ejectaVn)
def cal_ejecta_from_excavation(Vole_cum,ve_bin,x, i_maxL, C0):
    bin_ejectaV = np.zeros(len(x)-1,dtype=np.float32)
    for i in range(len(x) - 1):
        Voli = Vole_cum[i + 1] - Vole_cum[i] if i > 1 else Vole_cum[i]
        bin_ejectaV[i] = Voli
    Vexc = np.sum(bin_ejectaV)


    n_bin = np.zeros(len(binL))
    n_bin[i_maxL] = 1

    for i in np.arange((i_maxL + 1), len(binL)):
        n_bin[i] = C0 * binM[i] ** b_CUM - n_bin[i - 1]
    n_bin *= Vexc / np.sum(n_bin * binV)
    V_bin = n_bin * binV
    V_bin[-1] = Vexc - np.sum(V_bin[0:-1])
    n_bin[-1] = V_bin[-1] / binV[-1]
    V_bin_left = np.array(V_bin)
    n_bin_left = np.array(n_bin)

    t = 2*ve_bin*math.sin(45*math.pi/180)/parameters["g"]
    L = x+ve_bin*math.cos(45*math.pi/180) * t

    #L = np.flipud(L)
    bin_ejectaV =np.flipud(bin_ejectaV)

    bin_ejectaVn = np.zeros(len(binL))
    for i in range(len(bin_ejectaV)):
        Vi = bin_ejectaV[i]
        ni = np.zeros(len(binL))
        cum_Vbin_left = np.cumsum(V_bin_left)

        if i == (len(bin_ejectaV) - 1):
            ni = np.array(n_bin_left)
            bin_ejectaVn = np.vstack([bin_ejectaVn, ni])
            break

        n_near = (cum_Vbin_left < Vi).argmin()
        ni[0:n_near] = np.array(n_bin_left[0:n_near])
        ni[n_near] = (Vi - np.sum(ni * binV)) / binV[n_near]
        n_bin_left[0:n_near + 1] = n_bin_left[0:n_near + 1] - ni[0:n_near + 1]
        V_bin_left = n_bin_left * binV

        bin_ejectaVn = np.array(ni) if i == 0 else np.vstack([bin_ejectaVn, ni])

    return Vexc,L, np.array(bin_ejectaV), bin_ejectaVn

def save_variation(variation,filename):
    F1 = open(filename, 'wb')
    pickle.dump(variation, F1, True)
    F1.close()
    return 1
