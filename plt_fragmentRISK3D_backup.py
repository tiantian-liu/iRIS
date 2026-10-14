# plot the modeling results

# copyright: 10.10.2026 T. Liu @mfn Berlin


import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
from matplotlib.colors import LogNorm
import pickle
import math
from numpy import polyfit, poly1d
from scipy.spatial import Voronoi, voronoi_plot_2d
import function_fragmentRISK_3D_backup as f
import sim_batch_fragmentRISK_3D_backup as batch
parameters = batch.params
var = batch.variants


### don't touch the following two functions ###
def PolygonArea(corners):
    #https://stackoverflow.com/questions/24467972/calculate-area-of-polygon-given-x-y-coordinates
    n = len(corners)
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += corners[i][0] * corners[j][1]
        area -= corners[j][0] * corners[i][1]
    area = abs(area) / 2.0
    return area

def voronoi_finite_polygons_2d(vor, radius=None):
    if vor.points.shape[1] != 2:
        raise ValueError("Requires 2D input")
    pp1 = []
    new_regions = []
    new_vertices = vor.vertices.tolist()

    center = vor.points.mean(axis=0)
    if radius is None:
        radius = np.ptp(vor.points).max()

    # Construct a map containing all ridges for a given point
    all_ridges = {}
    for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
        all_ridges.setdefault(p1, []).append((p2, v1, v2))
        all_ridges.setdefault(p2, []).append((p1, v1, v2))

    # Reconstruct infinite regions
    for p1, region in enumerate(vor.point_region):
        vertices = vor.regions[region]

        if all(v >= 0 for v in vertices):
            # finite region
            new_regions.append(vertices)
            pp1.append(p1)
            continue
        # reconstruct a non-finite region
        ridges = all_ridges[p1]
        new_region = [v for v in vertices if v >= 0]

        for p2, v1, v2 in ridges:
            if v2 < 0:
                v1, v2 = v2, v1
            if v1 >= 0:
                # finite ridge: already in the region
                continue

            # Compute the missing endpoint of an infinite ridge

            t = vor.points[p2] - vor.points[p1]  # tangent
            t /= np.linalg.norm(t)
            n = np.array([-t[1], t[0]])  # normal

            midpoint = vor.points[[p1, p2]].mean(axis=0)
            direction = np.sign(np.dot(midpoint - center, n)) * n
            far_point = vor.vertices[v2] + direction * radius

            new_region.append(len(new_vertices))
            new_vertices.append(far_point.tolist())
        # sort region counterclockwise
        pp1.append(p1)
        vs = np.asarray([new_vertices[v] for v in new_region])
        c = vs.mean(axis=0)
        angles = np.arctan2(vs[:, 1] - c[1], vs[:, 0] - c[0])
        new_region = np.array(new_region)[np.argsort(angles)]
        # finish
        new_regions.append(new_region.tolist())
    return new_regions, np.asarray(new_vertices),pp1


parameters.update(var[0])

R = parameters["R"]
num_location = int(parameters["N_res"][0])
pt = parameters["pt"] # density of the impact target in kg/m3
vc = parameters["vc"]
binL = parameters["binL"]
binV = 4/3*math.pi*(binL/2)**3

nL = len(binL)

path_save = parameters["path_save"]  #

F1 = open(path_save+'/p.txt', 'rb')
p = pickle.load(F1)
F1.close()
num_location = len(p[:,0])
u=np.where(p[:,5]>math.pi)[0]
p[u,5]=p[u,5]-2*math.pi
p[:,4] = p[:,4]*180/math.pi
p[:,5] = p[:,5]*180/math.pi

### choose the plot you want ###
plt_fig = 'poisson_flux'

i_count = 37
while i_count<38:
    ### bestfitting for the u-K parameter from Luther et al. 2018 ###
    if plt_fig == 'u-K':
        f = np.array([0, 0.2, 0.4, 0.6, 0.8, 1.0])
        K3_a = np.array([0.43, 0.53, 0.63, 0.77, 0.81, 0.82])
        K3_b = np.array([0.56, 0.59, 0.60, 0.70, 0.62, 0.55])
        plt.plot(f, K3_a, 'k.')
        plt.plot(f, K3_b, 'r.')
        plt.show()

    ### ejecta thickness distribution for single impact with D=4km ###
    elif plt_fig == 'fit_ejecta_blanket':
        i = 0
        F1 = open(path_save + 'xDist_' + str(i) + '.txt', 'rb')
        xDist = pickle.load(F1)
        F1.close()

        F1 = open(path_save + 'yH_' + str(i) + '.txt', 'rb')
        yH = pickle.load(F1)
        F1.close()

        Rt=2.25
        x = xDist/Rt
        y = yH
        idx = np.where((x<30) & (x>8))[0]
        x2 = np.array(x[idx])
        y2 = np.array(y[idx])

        coeff = polyfit(np.log10(x2),np.log10(y2),1)

        x3 = np.linspace(1,20,20)
        y3 = coeff[0]*np.log10(x3) + coeff[1]
        x4 = 5/Rt
        y4 = coeff[0]*np.log10(x4) + coeff[1]
        x3 = x3
        y3 = 10**y3
        y4 = 10 ** y4
        fig, ax = plt.subplots(1,1, figsize=(5.4, 4.2))
        plt.plot(x,y,"k.")
        plt.plot(x3, y3, "-b")
        plt.plot(x4, y4, "r*")
        plt.xscale('log')
        plt.yscale('log')
        plt.xlim([1,120])
        plt.ylim([1e-5, 200])

        plt.xlabel("x/R")
        plt.ylabel("Ejecta thickness, m")

        plt.savefig(path_save + 'ejectaH_K3S_plateau150.png', transparent=True)


        idx = np.where(xDist<10)
        xDist2 = np.array(xDist[idx])

        plt.show()

    ### quick calculation of K2, K3 and miu ###
    elif plt_fig == "constant_test":
        K3 = np.array([0.43, 0.82])
        miu = np.array([[0.49, 0.82],[0.47, 0.35]])

        K2 = np.zeros([2,2])

        a = 0.5 #km
        R = 4.019 #km
        g = 1.62e-3 #km s-2
        U = 15 #km/s
        for i in range(len(K3)):
            for j in range(len(miu[0,:])):
                K3i = K3[i]
                K2[i][j] = K3i*(g*R)**0.5 / U * (R/a)**(1/miu[i][j])
                print(K3i)
                print(miu[i][j])
                print(K2[i][j])

        print(K2)

    ### estimation of escaped ejecta with varying U, miu and C4 ###
    elif plt_fig == "cal_Vescape": # cal_Vescape(impact_info, U, D_beta, C_D):
        F1 = open(path_save + 'impact_info0.txt', 'rb')
        impact_info = pickle.load(F1)
        F1.close()

        bin_ve = 2.38  # km/s
        U, g, D_beta, C_D = parameters["U"],parameters["g"],parameters["D_beta"], parameters["C_D"]
        #U = 5
        nimp = len(impact_info[:, 0])
        dt = parameters["t0"]*1e9/nimp
        miu = [0.38,0.42,0.5] #parameters["u_vemax"]
        nmiu = len(miu)
        Vescape = np.zeros([nimp,nmiu+1])
        Vescape[:,0] = np.array(1e3 * impact_info[:, 3])
        C4 = [0.06,parameters["C_vecum"]]
        for j in range(nmiu):
            ui = miu[j]
            C_vecum = C4[1]
            for i in range(len(impact_info[:, 0])):
                Rti = 1e3 * impact_info[i, 3]  # D_t/2 # in meter
                Dti = Rti * 2 * 1E-3
                Lpi = (Dti / (0.8 * C_D / 1.6 ** D_beta) * g ** D_beta / U ** (2 * D_beta)) ** (1 / (1 - D_beta))  # in km
                Vproj = 4 / 3 * math.pi * (Lpi / 2 * 1E3) ** 3  # in m3
                Vescape[i,j+1] = Vproj * C_vecum * (bin_ve / U) ** (-3 * ui)  # Vole_cum  # in m3

        fig, axs = plt.subplots(1,1,figsize=(5,5))
        colors = ['k', 'g', 'b','r']
        x = dt*np.arange(nimp)
        for i in range(nmiu):
            yi = np.cumsum(Vescape[:,i+1])
            plt.plot(x,yi,color=colors[i],linestyle='-')
        plt.xlabel('Time (year)')
        plt.ylabel('Accumulated volume of escaped materials (m$^3$)')
        plt.ylim([0,8])
        plt.savefig(path_save + 'figures/'+ 'Vescape_miu_U5.png', transparent=True)
        plt.show()

    ### calculate critical damagement energy for triple-wall infrastructure (Schaefer et al. 2018) ###
    elif plt_fig == "cal_critical_DamageEnergy":
        path_save = "/data/user/tiantians/Python/results/ejectaRISK/"
        tw, tb, tob = 0.3, 0.08, 0.08  #cm
        pp = 2.940 # g/cm3
        K_3S = 1.4
        a = 1/2
        theta = 4/3
        yita = 72 #ksi
        ang = 0 # !! different definition: vertical angle

        v = np.arange(0.05,2.3,0.01)#0.5 # km/s
        dc = np.zeros(len(v))
        Edc = np.zeros(len(v))
        Vdc = np.zeros(len(v))
        Mdc = np.zeros(len(v))
        Edc = np.zeros(len(v))
        for i in range(len(v)):
            vi = v[i]
            dc[i] = ( ((((tw)**a + tb)/K_3S)*(yita/40)**0.5 + tob)/(0.6*pp**0.5*vi**(2/3)) )**(18/19) # cm
            Vdc[i] = 3/4*math.pi*(dc[i]/2)**3 #cm3
            Mdc[i] = pp*Vdc[i]*1e-3 # kg
            Edc[i] = 0.5*Mdc[i]*(vi*1e3)**2 #kg*(m/s)^2
        fig, axs = plt.subplots(1, 1, figsize=(5, 5))
        plt.plot(v, dc, color='k', linestyle='-')
        plt.xlabel('Fragment velocity, km$/s$')
        plt.ylabel('Critical fragment size (cm)') #$^3$
        plt.xlim([0, 2.4])
        plt.savefig(path_save + 'figures/' + 'Critica_lDc.png', transparent=True)

        fig, axs = plt.subplots(1, 1, figsize=(5, 5))
        plt.plot(v, Edc, color='k', linestyle='-')
        plt.xlabel('Fragment velocity, km$/s$')
        plt.ylabel('Critical fragment energy (J)')  # $^3$
        plt.xlim([0, 2.4])
        plt.ylim([95, 150])
        plt.savefig(path_save + 'figures/' + 'Critica_Ek.png', transparent=True)
        plt.show()

    ### calculate ejecta fragment velocity and the reduced Ek at certain height above the surface ###
    elif plt_fig == "cal_ve_atH":
        g = parameters["g"]
        ang = 45*math.pi/180 # verticle = 90
        H0 = [0.02,0.05, 0.1] # km
        v = np.arange(0.05, 2.3, 0.01)  # km/s
        v0x = v * np.cos(ang)
        v0y = v * np.sin(ang)
        Hmax = v0y ** 2 / (2 * g)
        tHmax = v0y / g
        colors = ['k','b','r']
        fig, ax1 = plt.subplots(1, 1, figsize=(6, 6))
        for i in range(len(H0)):
            H = H0[i]
            ci = colors[i]

            tmax2H0 = np.zeros(len(Hmax))
            v_Hy = np.zeros(len(Hmax))
            t1 = np.zeros(len(Hmax))
            lostEk = np.zeros(len(Hmax))

            idx = np.where(Hmax>H)[0]
            tmax2H0[idx] = abs((2 * (Hmax[idx] - H)/g) ** 0.5)
            t1[idx] = tHmax[idx] - tmax2H0[idx]
            v_Hy = v0y[idx]-g*t1[idx]
            vH = (v0x**2+v_Hy**2)**0.5
            lostEk[idx] = (v[idx]**2 - vH[idx]**2)/v[idx]**2 *100
            ax1.plot(v, lostEk, color=ci, linestyle='-')
        ax1.set_xlabel('Fragment velocity, km/s')
        ax1.set_ylabel('The percentage of the reduced kinetic energy, %')  # $^3$
        ax1.set_xlim([0.01,2])
        ax1.set_yscale('log')

        plt.savefig(path_save + 'figures/' + 'LostEk.png', transparent=True)
        plt.show()

    ### plot the ejecta fragment Ek considering varying timeframe ###
    elif plt_fig == "Ek_varyT":
        path1 = "/data/user/tiantians/Python/results/ejectaRISK/MC_1m_100y_dx0005/"
        path2 = "/data/user/tiantians/Python/results/ejectaRISK/MC_1m_200y/"
        F1 = open(path1 + "arrayV/" + 'arrayV_9.txt' , 'rb')
        arrayV1 = pickle.load(F1)
        F1.close()

        F1 = open(path1 + "arrayVe/" + 'arrayVe_9.txt', 'rb')
        arrayVe1 = pickle.load(F1)
        F1.close()

        Ek1 = 0.5 * (pt * arrayV1) * (arrayVe1 * 1e3) ** 2  # kg m2 s-2 -->J
        Vall1 = np.sum(arrayV1[arrayVe1 > 0])

        F1 = open(path2 + "arrayV/" + 'arrayV_22.txt', 'rb')
        arrayV2 = pickle.load(F1)
        F1.close()

        F1 = open(path2 + "arrayVe/" + 'arrayVe_22.txt', 'rb')
        arrayVe2 = pickle.load(F1)
        F1.close()

        Ek2 = 0.5 * (pt * arrayV2) * (arrayVe2 * 1e3) ** 2  # kg m2 s-2 -->J
        Vall2 = np.sum(arrayV2[arrayVe2 > 0])



        a = np.linspace(-1, 6, 50)
        x0 = 10 ** a
        x, y1, y2 = np.zeros(len(x0) - 1), np.zeros(len(x0) - 1), np.zeros(len(x0) - 1)
        for i in range(len(x)):
            a, b = x0[i], x0[i + 1]
            idx = np.where((Ek1 > a) & (Ek1 < b))
            x[i] = (a + b) / 2
            y1[i] = np.sum(arrayV1[idx])
            idx = np.where((Ek2 > a) & (Ek2 < b))
            y2[i] = np.sum(arrayV2[idx])
        fig, ax1 = plt.subplots(1, 1, figsize=(5, 5))
        ax1.set_xlabel('Fragment kinetic energy, J')
        ax1.set_ylabel('Volume, m$^3$')  # $^3$
        ax1.set_xscale('log')
        ax1.set_yscale('log')
        ax1.set_xlim([1e-1, 1e6])
        ax1.plot(x, np.cumsum(y1) , '-k.')
        ax1.plot(x, np.cumsum(y2) , '-r.')
        plt.savefig('/data/user/tiantians/Python/results/ejectaRISK/ProbEk/Prob_Ek_varyT_V.png',transparent=True)

        fig, ax2 = plt.subplots(1, 1, figsize=(5, 5))
        ax2.set_xlabel('Fragment kinetic energy, J')
        ax2.set_ylabel('Probability')  # $^3$
        ax2.set_xscale('log')
        ax2.set_yscale('log')
        ax2.set_xlim([1e-1, 1e6])
        p1 = y1 / Vall1
        p2 = y2 / Vall2
        idx = np.where((x >= 10) & (x <= 200))
        ax2.plot(x, y1 / Vall1, '-k.')
        ax2.plot(x, y2 / Vall2, '-r.')
        plt.savefig('/data/user/tiantians/Python/results/ejectaRISK/ProbEk/Prob_Ek_varyT_Prob.png', transparent=True)

        plt.show()

    ### map the spatial distribution of the total fragments Ek over the polar region ###
    elif plt_fig == "polar_map_energy":
        nsim = 2
        for ii in range(1,nsim):
            ii = 11
            F1 = open(path_save + "arrayV/" + 'arrayV_' + str(ii) + '.txt', 'rb')
            arrayV = pickle.load(F1)
            F1.close()

            F1 = open(path_save + "arrayVe/" + 'arrayVe_' + str(ii) + '.txt', 'rb')
            arrayVe = pickle.load(F1)
            F1.close()

            Vejecta = np.sum(arrayV[0:-1], 0)
            Mejecta = pt * Vejecta
            Ek0 = 0.5 * (pt * np.array(arrayV)) * np.array(arrayVe) ** 2  # if Ve = 0, it means not ejecta
            Ek = np.sum(Ek0, 0)

            idx2 = np.where(p[:, -1] < 0)[0]
            p[idx2, -1] = 360 + p[idx2, -1]
            idx = np.where(p[:,4]<-88)[0]# the plot range lat<-88
            p2 = np.array(p[idx,:])

            lat0 = p[idx, 4]
            lon0 = p[idx, 5]

            lat = np.unique(lat0)
            lon = np.unique(lon0)
            nlat = np.unique(lat).size
            nlon = np.unique(lon).size

            z0 = Ek[idx] * 1e6 # 1J = 1 kg. m2. s-2

            X, Y, dist, Z = np.zeros((nlat,nlon)), np.zeros((nlat,nlon)), np.zeros((nlat,nlon)), np.zeros((nlat,nlon))
            beta0, arfa0 = -90*math.pi/180, 0
            for  i in range(len(idx)):
                lati,loni = p2[i,4], p2[i,5]
                zi = z0[i]
                ila = np.argmin(np.abs(lati-lat))
                ilo = np.argmin(np.abs(loni-lon))
                X[ila,ilo] = loni
                Y[ila,ilo] = lati
                beta1,arfa1 = lati*math.pi/180,loni*math.pi/180
                dist[ila,ilo] = R * np.arccos(np.cos(beta0) * np.cos(beta1) * np.cos(arfa0 - arfa1) + np.sin(beta0) * np.sin(beta1))
                Z[ila,ilo] = zi
            cmap = plt.get_cmap('coolwarm')  # RdBu  coolwarm
            cmap.set_under('k')
            cmap.set_over('r')

            fig, axs = plt.subplots(1,1,figsize=(10,5),subplot_kw=dict(projection='polar'))
            levels = np.array([0.001,1,3, 5, 10,30,50,100,300,500,1000])*1e3
            p1=axs.contourf(X*math.pi/180,dist,Z,norm='log',levels=levels,cmap=cmap,extend='both')#levels=levels, ,norm=LogNorm(vmin=1e-10,vmax=0.01)
            cbar =fig.colorbar(p1,ax=axs,extend='both')
            plt.savefig(path_save + 'figures/'+ 'MC_POLAR_1m_allsim_'+str(ii)+'part.png', transparent=True)
            plt.show()

    ### plot the ejecta fragment Ek from the multiple MC simulations ###
    elif plt_fig == "data4_polar_map_energy_allsim":
        path = "/data/user/tiantians/Python/results/ejectaRISK/MC_more_1m_100y/"
        first_time = 0
        first_statistic = 0
        if first_time:
            pf_V = path+"/arrayV/"
            f_folders_V = os.listdir(pf_V)
            f_folders_V.sort()

            pf_Ve = path + "/arrayVe/"
            f_folders_Ve = os.listdir(pf_Ve)
            f_folders_Ve.sort()

            Ek_allsim = np.zeros((len(p[:,0]),len(f_folders_V)))
            V_allsim = np.zeros((len(p[:, 0]), len(f_folders_V)))
            for i in range(len(f_folders_V)):
                print(i)
                fVi = f_folders_V[i]
                fVei = f_folders_Ve[i]

                F1 = open(path + "arrayV/" + fVi, 'rb')
                arrayVi = pickle.load(F1)
                F1.close()

                F1 = open(path + "arrayVe/" + fVei, 'rb')
                arrayVei = pickle.load(F1)
                F1.close()

                Ek0 = 0.5 * (pt * np.array(arrayVi)) * np.array(arrayVei) ** 2  # if Ve = 0, it means not ejecta
                Eki = np.sum(Ek0, 0)
                Vi = np.sum(arrayVi[0:-1,:], 0)

                Ek_allsim[:,i] = np.array(Eki)
                V_allsim[:,i] = np.array(Vi)
                f.save_variation(Ek_allsim, path + "Ek_allSim.txt")
                f.save_variation(V_allsim, path + "V_allSim.txt")
        else:
            F1 = open(path + "Ek_allSim.txt", 'rb')
            Ek_allsim = pickle.load(F1)
            F1.close()

            F1 = open(path + "V_allSim.txt", 'rb')
            V_allsim = pickle.load(F1)
            F1.close()

            if first_statistic:
                a = np.linspace(-1, 6, 50)
                x0 = 10 ** a
                nEkbin = len(x0) - 1
                nSim = len(Ek_allsim[0, :])
                x_Ekbin, y_sumbinV = np.zeros(nEkbin), np.zeros((nEkbin, nSim))
                for i in range(nEkbin):
                    print(i)
                    a, b = x0[i], x0[i + 1]
                    x_Ekbin[i] = (a + b) / 2
                    for j in range(nSim):
                        Ek0j = np.array(Ek_allsim[:, j]) * 1e6
                        V0j = np.array(V_allsim[:, j])
                        idx = np.where((Ek0j > a) & (Ek0j < b))
                        #print(np.sum(V0j[idx]))
                        y_sumbinV[i, j] = np.sum(V0j[idx])
                f.save_variation(x_Ekbin, path + "x_Ekbin.txt")
                f.save_variation(y_sumbinV, path + "y_sumbinV.txt")

            else:
                F1 = open(path + "x_Ekbin.txt", 'rb')
                x_Ekbin = pickle.load(F1)
                F1.close()

                F1 = open(path + "y_sumbinV.txt", 'rb')
                y_sumbinV = pickle.load(F1)
                F1.close()

            fig, ax1 = plt.subplots(1, 1, figsize=(5, 5))
            ax1.set_xlabel('Kinetic energy, J')
            ax1.set_ylabel('Probability')
            ax1.set_xscale("log")
            ax1.set_yscale("log")
            ax1.set_ylim([1e-5,1])
            a = [10,50,100,300,500,590]
            colors = ["y","r","c","b","g","k"]
            lw = [0.5,0.5,0.5,0.5,0.5,3,0.5,0.5,0.5]
            for i in range(len(a)):
                ai=a[i]
                y_avgV = np.median(y_sumbinV[:,0:ai],axis = 1)
                y_quartile1 = np.percentile(y_sumbinV[:,0:ai],25,axis = 1)
                y_quartile3 = np.percentile(y_sumbinV[:, 0:ai], 75, axis=1)

                # plt.fill_between(x_Ekbin,y1,y2,color="grey")
                plt.plot(x_Ekbin,y_avgV/np.sum(y_avgV),color=colors[i],marker=".",linestyle="-",linewidth=lw[i])
            y1, y2 = y_quartile1 / np.sum(y_avgV), y_quartile3/np.sum(y_avgV)
            plt.fill_between(x_Ekbin, y1, y2, color="grey", alpha =0.3)

            plt.savefig(path + 'figures/' + 'MC_more_1m_100y_allSim_statistic_med.png', transparent=True)

            y_avgV = np.average(y_sumbinV, axis=1)
            y_stdV = np.std(y_sumbinV, axis=1)
            fig, ax2 = plt.subplots(1, 1, figsize=(5, 5))
            ax2.set_xlabel('Kinetic energy, J')
            ax2.set_ylabel('Probability')
            ax2.set_xscale("log")
            ax2.set_yscale("log")
            ax2.plot(x_Ekbin,y_stdV/np.sum(y_avgV),color="k",marker=".",linestyle="-",linewidth=1)
            plt.savefig(path + 'figures/' + 'MC_more_1m_100y_allSim_statistic_std.png', transparent=True)

            print(1)
            ### area percentage/probability  ###
            idx = np.where(p[:, 4] < -80)[0]
            nump = len(idx)
            med = np.median(Ek_allsim,axis = 1)[idx] * 1e6
            quantile1 = np.percentile(Ek_allsim, 25, axis=1)[idx] * 1e6
            quantile3 = np.percentile(Ek_allsim, 75, axis=1)[idx] * 1e6
            criticalEk = np.arange(10,200,10)
            alldata = [med,quantile1,quantile3]
            n = len(criticalEk)
            areaPerc = np.zeros((n,3))
            for j in range(3):
                datai = alldata[j]
                for i in range(n):
                    Eki = criticalEk[i]
                    idxi = np.where(datai>=Eki)[0]
                    areaPerc[i,j] = len(idxi)/nump
            fig, ax3 = plt.subplots(1, 1, figsize=(5, 5))
            ax3.set_xlabel('Kinetic energy, J')
            ax3.set_ylabel('Probability')
            ax3.set_ylim([-0.01,1.01])
            # ax3.set_xscale("log")
            # ax3.set_yscale("log")
            ax3.plot(criticalEk, areaPerc[:,0], color="k", marker=".", linestyle="-", linewidth=2)
            ax3.plot(criticalEk, areaPerc[:, 1], color="k", marker=".", linestyle="--", linewidth=2)
            ax3.plot(criticalEk, areaPerc[:, 2], color="k", marker=".", linestyle="-.", linewidth=2)
            plt.savefig(path + 'figures/' + 'MC_more_1m_100y_allSim_areaPerc.png', transparent=True)

            ### map average Ek ###
            idx2 = np.where(p[:, -1] < 0)[0]
            p[idx2, -1] = 360 + p[idx2, -1]
            idx = np.where(p[:, 4] < -88)[0]  # range(15000,len(p[:,0]))
            p2 = np.array(p[idx, :])

            lat0 = p[idx, 4]
            lon0 = p[idx, 5]

            lat = np.unique(lat0)
            lon = np.unique(lon0)
            nlat = np.unique(lat).size
            nlon = np.unique(lon).size

            z0 = np.median(Ek_allsim,axis = 1)[idx] * 1e6
            z0 = np.percentile(Ek_allsim, 75, axis=1)[idx]* 1e6
            z0 = np.percentile(Ek_allsim, 25, axis=1)[idx] * 1e6
            # plt.figure()
            # plt.plot(range(len(z0)),z0,"k.")
            # plt.show()
            X, Y, dist, Z = np.zeros((nlat, nlon)), np.zeros((nlat, nlon)), np.zeros((nlat, nlon)), np.zeros((nlat, nlon))
            beta0, arfa0 = -90 * math.pi / 180, 0
            for i in range(len(idx)):
                lati, loni = p2[i, 4], p2[i, 5]
                zi = z0[i]
                ila = np.argmin(np.abs(lati - lat))
                ilo = np.argmin(np.abs(loni - lon))
                X[ila, ilo] = loni
                Y[ila, ilo] = lati
                beta1, arfa1 = lati * math.pi / 180, loni * math.pi / 180
                dist[ila, ilo] = R * np.arccos(np.cos(beta0) * np.cos(beta1) * np.cos(arfa0 - arfa1) + np.sin(beta0) * np.sin(beta1))
                Z[ila, ilo] = zi
            cmap = plt.get_cmap('coolwarm')  # RdBu  coolwarm
            cmap.set_under('grey')
            cmap.set_over('y')

            fig, axs = plt.subplots(1, 1, figsize=(10, 5), subplot_kw=dict(projection='polar'))
            levels = np.array([1, 3, 5, 10, 30, 50, 100, 300, 500, 1000]) * 1e0
            levels = range(18,24) #(35,46) (80,120)
            p1 = axs.contourf(X * math.pi / 180, dist, Z, levels=levels, cmap=cmap, extend='both')  # norm='log', levels=levels, ,norm=LogNorm(vmin=1e-10,vmax=0.01)
            cbar = plt.colorbar(p1, ax=axs, extend='both')
            plt.savefig(path + 'figures/' + 'MC_more_POLAR_1m_100y_allSim_quartile1.png', transparent=True)
            plt.show()

    ### plot distribution of differently-sized fragments considering varying b ###
    elif plt_fig == "cumulative_grainV":
       ### the data come from "polar_map_grainH"
        F1 = open("/data/user/tiantians/Python/results/ejectaRISK/grainV/grainV_b1.2.txt", 'rb')
        grainV_bgentle = pickle.load(F1)
        F1.close()
        F1 = open("/data/user/tiantians/Python/results/ejectaRISK/grainV/grainV_b1.8.txt", 'rb')
        grainV_bsteep = pickle.load(F1)
        F1.close()

        x = np.array(grainV_bgentle[:, 0])
        y1 = np.array(grainV_bgentle[:, 1])
        y2 = np.array(grainV_bsteep[:, 1])
        fig, ax1 = plt.subplots(1, 1, figsize=(5, 5))
        ax1.plot(x, y1, '-k.')
        ax1.plot(x, y2, '-r.')
        ax1.set_xlabel('Fragment size, m')
        ax1.set_ylabel('Volume, m$^3$')  # $^3$
        ax1.set_yscale('log')
        ax1.set_xlim([0.001, 0.1])
        plt.savefig('/data/user/tiantians/Python/results/ejectaRISK/GrainSmall.png', transparent=True)

        ### cumulative plot ###
        x = np.array(grainV_bgentle[:,0])
        y1 = np.cumsum(grainV_bgentle[:,1])
        y2 = np.cumsum(grainV_bsteep[:,1])

        fig, ax1 = plt.subplots(1, 1, figsize=(5, 5))

        ax1.plot(x, y1, color='k', marker='.',linestyle='-')
        ax1.plot(x, y2, color='r', marker='.',linestyle='-')
        ax1.set_xlabel('Fragment size, m')
        ax1.set_ylabel('Cumulative volume, m$^3$')  # $^3$
        #plt.ylabel('Cumulative volume, m$^3$')
        #ax1.set_xscale('log')
        ax1.set_yscale('log')
        ax1.set_xlim([0.0001,0.11])
        plt.savefig('/data/user/tiantians/Python/results/ejectaRISK/CumGrain.png', transparent=True)
        plt.show()

    ### map the spatial distribution of the volume of small fragments ###
    elif plt_fig == "polar_map_grainH":

        i = 9
        F1 = open(path_save + "arrayV/" + 'arrayV_' + str(i) + '.txt', 'rb')
        arrayV = pickle.load(F1)
        F1.close()

        F1 = open(path_save + "arrayVe/" + 'arrayVe_' + str(i) + '.txt', 'rb')
        arrayVe = pickle.load(F1)
        F1.close()

        F1 = open(path_save + "nfrag/" + 'nfrag1_' + str(i) + '.txt', 'rb')
        nfrag1 = pickle.load(F1)
        F1.close()
        F1 = open(path_save + "nfrag/" + 'nfrag2_' + str(i) + '.txt', 'rb')
        nfrag2 = pickle.load(F1)
        F1.close()
        F1 = open(path_save + "nfrag/" + 'nfrag3_' + str(i) + '.txt', 'rb')
        nfrag3 = pickle.load(F1)
        F1.close()
        F1 = open(path_save + "nfrag/" + 'nfrag4_' + str(i) + '.txt', 'rb')
        nfrag4 = pickle.load(F1)
        F1.close()
        F1 = open(path_save + "nfrag/" + 'nfrag5_' + str(i) + '.txt', 'rb')
        nfrag5 = pickle.load(F1)
        F1.close()
        F1 = open(path_save + "nfrag/" + 'nfrag6_' + str(i) + '.txt', 'rb')
        nfrag6 = pickle.load(F1)
        F1.close()
        F1 = open(path_save + "nfrag/" + 'nfrag7_' + str(i) + '.txt', 'rb')
        nfrag7 = pickle.load(F1)
        F1.close()
        F1 = open(path_save + "nfrag/" + 'nfrag8_' + str(i) + '.txt', 'rb')
        nfrag8 = pickle.load(F1)
        F1.close()
        F1 = open(path_save + "nfrag/" + 'nfrag9_' + str(i) + '.txt', 'rb')
        nfrag9 = pickle.load(F1)
        F1.close()
        F1 = open(path_save + "nfrag/" + 'nfrag10_' + str(i) + '.txt', 'rb')
        nfrag10 = pickle.load(F1)
        F1.close()

        arrayN = np.column_stack((nfrag1, nfrag2, nfrag3, nfrag4, nfrag5, nfrag6, nfrag7, nfrag8, nfrag9, nfrag10))

    ## prepare data to plot cumulative grain V ##
        V0 = arrayN* binV
        V = np.sum(np.sum(V0[0:-1],0),0)
        Vgrain = np.zeros((len(V),2))
        Vgrain[:,0] = np.array(binL)
        Vgrain[:,1] = np.array(V)
        f.save_variation(Vgrain, "/data/user/tiantians/Python/results/ejectaRISK/grainV/grainV_b1.2.txt")
    ## prepare data to plot cumulative grain V ##

        N0 =  np.sum(arrayN[:, :, 18::],0)
        idx = np.where(N0 > 0)[0]
        V0 = np.sum(N0 * binV[18::],1)

        idx2 = np.where(p[:, -1] < 0)[0]
        p[idx2, -1] = 360 + p[idx2, -1]
        idx = np.where(p[:,4]<-88)[0]
        p2 = np.array(p[idx,:])

        lat0 = p[idx, 4]
        lon0 = p[idx, 5]


        lat = np.unique(lat0)
        lon = np.unique(lon0)
        nlat = np.unique(lat).size
        nlon = np.unique(lon).size

        z0 = V0[idx]

        X, Y, dist, Z = np.zeros((nlat,nlon)), np.zeros((nlat,nlon)), np.zeros((nlat,nlon)), np.zeros((nlat,nlon))
        beta0, arfa0 = -90*math.pi/180, 0
        for  i in range(len(idx)):
            lati,loni = p2[i,4], p2[i,5]
            zi = V0[i]
            ila = np.argmin(np.abs(lati-lat))
            ilo = np.argmin(np.abs(loni-lon))
            X[ila,ilo] = loni
            Y[ila,ilo] = lati
            beta1,arfa1 = lati*math.pi/180,loni*math.pi/180
            dist[ila,ilo] = R * np.arccos(np.cos(beta0) * np.cos(beta1) * np.cos(arfa0 - arfa1) + np.sin(beta0) * np.sin(beta1))
            Z[ila,ilo] = zi
        cmap = plt.get_cmap('coolwarm')  # RdBu  coolwarm
        cmap.set_under('grey')
        cmap.set_over('y')

        fig, axs = plt.subplots(1,1,figsize=(10,5),subplot_kw=dict(projection='polar'))
        levels = np.array([0.1,1,3, 5, 10,30,50,100,1000,1e4])*1e-7
        p1=axs.contourf(X*math.pi/180,dist,Z,norm='log',levels=levels,cmap=cmap,extend='both')#levels=levels, ,norm=LogNorm(vmin=1e-10,vmax=0.01)
        cbar =plt.colorbar(p1,ax=axs,extend='both')
        plt.savefig(path_save + 'figures/'+ 'MC_POLAR_1m_d1.8.png', transparent=True)
        plt.show()

        print(1)

    ### histogram of the Ve and  Ek ###
    elif plt_fig == "global_statisticse":

        i = 9
        F1 = open(path_save + "arrayV/" + 'arrayV_' + str(i) + '.txt', 'rb')
        arrayV = pickle.load(F1)
        F1.close()

        F1 = open(path_save + "arrayVe/" + 'arrayVe_' + str(i) + '.txt', 'rb')
        arrayVe = pickle.load(F1)
        F1.close()

        Vall = np.sum(arrayV[arrayVe>0])

        ve_min = 1e-3  #np.min(arrayVe[arrayVe > 0])
        ve_max = 2.2
        # idx = np.where(arrayVe>=ve_min)[0] #np.where((arrayVe<0.01) & (arrayVe>=ve_min))[0]
        # perc = np.sum(arrayV[idx])/Vall

        x0 = np.linspace(ve_min,ve_max,50)
        x, y = np.zeros(len(x0)-1),np.zeros(len(x0)-1)
        for i in range(len(x)):
            a, b = x0[i],x0[i+1]
            idx = np.where((arrayVe>a) & (arrayVe<b))
            x[i] = (a+b)/2
            y[i] =np.sum(arrayV[idx])
        plt.bar(x,y/Vall,width=(x0[1]-x0[0])*0.9,color='grey')
        plt.yscale('log')
        plt.xlim([0, 2])
        plt.ylim([1e-6, 1.1])

        ### create data for sensitivity study ####
        prob_Ve = np.zeros((len(x),2))
        prob_Ve[:,0] = np.array(x)
        prob_Ve[:,1] = np.array(y/Vall)
        f.save_variation(prob_Ve, "/data/user/tiantians/Python/results/ejectaRISK/ProbVe/Prob_Ve_ref.txt")

        Ek0 = 0.5 * (pt * arrayV) * (arrayVe * 1e3) ** 2  # kg m2 s-2 -->J
        a = np.linspace(-1, 6, 50)
        x0 = 10**a
        x, y = np.zeros(len(x0) - 1), np.zeros(len(x0) - 1)
        for i in range(len(x)):
            a, b = x0[i], x0[i + 1]
            idx = np.where((Ek0 > a) & (Ek0 < b))
            x[i] = (a + b) / 2
            y[i] = np.sum(arrayV[idx])
        plt.figure()
        #plt.bar(x, y/Vall, width=(x0[1] - x0[0]) * 0.9,color='grey')
        plt.plot(x, y / Vall, '-k.')
        plt.xscale('log')
        plt.yscale('log')
        #plt.xlim([1,1e5])
        #plt.ylim([1e-7,1.1])
        prob_Ek = np.zeros((len(x), 2))
        prob_Ek[:, 0] = np.array(x)
        prob_Ek[:, 1] = np.array(y / Vall)
        f.save_variation(prob_Ek, "/data/user/tiantians/Python/results/ejectaRISK/ProbEk/Prob_Ek_ref.txt")

        # plt.savefig(path_save + 'figures/' + 'hist_1m_dx0.005Ek_100y.png', transparent=True)
        plt.show()

    ### plot the sensitivity of MC ###
    elif plt_fig == 'global_sensitivity_Prob_Ve_Ek':
        ## data come from "global_statisticse" ##
        pf = "/data/user/tiantians/Python/results/ejectaRISK/ProbVe/"
        f_folders = os.listdir(pf)
        f_folders.sort()
        colors = ['k','r','g','b','y','m','c']
        lw = [4,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5]
        fig, ax1 = plt.subplots(1, 1, figsize=(5, 5))
        ax1.set_xlabel('Fragment velocity, km/s')
        ax1.set_ylabel('Probability')  # $^3$
        # ax1.set_xscale('log')
        ax1.set_yscale('log')
        ax1.set_xlim([0.0, 2])
        for i in range(len(f_folders)):
            fi = pf + f_folders[i]
            F1 = open(fi, 'rb')
            proVe0 = pickle.load(F1)
            F1.close()
            idx = np.where(proVe0[:,1]==0)[0]
            proVe=np.delete(proVe0,idx,axis=0)
            ax1.plot(proVe[:,0], proVe[:,1], color=colors[i], marker='.', linestyle='-',linewidth=lw[i])
        plt.savefig('/data/user/tiantians/Python/results/ejectaRISK/fig_sensitivity/sensitivity_Ve.png', transparent=True)

        pf = "/data/user/tiantians/Python/results/ejectaRISK/ProbEk/"
        f_folders = os.listdir(pf)
        f_folders.sort()
        colors = ['k', 'r', 'g', 'b', 'y', 'm', 'c']
        fig, ax1 = plt.subplots(1, 1, figsize=(5, 5))
        ax1.set_xlabel('Fragment kinetic energy, J')
        ax1.set_ylabel('Probability')  # $^3$
        # ax1.set_xscale('log')
        ax1.set_xscale('log')
        ax1.set_yscale('log')
        ax1.set_xlim([1e-1, 1e6])
        for i in range(len(f_folders)):
            fi = pf + f_folders[i]
            F1 = open(fi, 'rb')
            proEk0 = pickle.load(F1)
            F1.close()
            idx = np.where(proEk0[:, 1] == 0)[0]
            proEk = np.delete(proEk0, idx, axis=0)
            ax1.plot(proEk[:, 0], proEk[:, 1], color=colors[i], marker='.', linestyle='-',linewidth=lw[i])
        plt.savefig('/data/user/tiantians/Python/results/ejectaRISK/fig_sensitivity/sensitivity_Ek2.png', transparent=True)

        plt.show()
        print(1)

    ### for single impact: map Ve, mass, and Ek ###
    elif plt_fig == 'map_energy':
        i = 9
        F1 = open(path_save + "arrayV/" + 'arrayV_' + str(i) + '.txt', 'rb')
        arrayV = pickle.load(F1)
        F1.close()

        F1 = open(path_save + "arrayVe/" + 'arrayVe_' + str(i) + '.txt', 'rb')
        arrayVe = pickle.load(F1)
        F1.close()

        Vejecta = np.sum(arrayV[0:-1],0)
        Mejecta = pt * Vejecta
        Ek0 = 0.5 * (pt * np.array(arrayV)) * np.array(arrayVe)**2 # if Ve = 0, it means not ejecta
        Ek = np.sum(Ek0,0)

        fig, ax = plt.subplots(3, 1, figsize=(3, 9))
        cu, co, c_zero = [0.5, 0.5, 0.5, 1], [0.1, 0, 0, 1], [1, 0, 1, 1]
        cmap = plt.get_cmap('coolwarm')#RdBu  coolwarm
        cmap.set_over(co)
        cmap.set_under(cu)

        points = np.zeros((len(p[:, 0]), 2))
        points[:, 0], points[:, 1] = p[:, 5], p[:, 4]
        vor = Voronoi(points)
        id_p = vor.point_region
        regions, vertices, pp1 = voronoi_finite_polygons_2d(vor)

        i = 0
        parea = []
        for region in regions:
            polygon = vertices[region]
            parea.append(PolygonArea(polygon))
        patches = []
        ####velocity
        Ve = np.sum(arrayVe, 0)
        cv = np.zeros(len(p[:, 0]))
        cM = np.zeros(len(p[:, 0]))
        cEk = np.zeros(len(p[:, 0]))
        for region in regions:
            polygon0 = vertices[region]
            polygon0[:, 0] = (polygon0[:, 0] + 180) / 360
            polygon0[:, 1] = (polygon0[:, 1] + 90) / 180
            polygon = Polygon(polygon0)
            patches.append(polygon)
            cv[i] = Ve[pp1[i]]
            cM[i] = Mejecta[pp1[i]]
            cEk[i] = Ek[pp1[i]]
            i += 1
        cmap.set_over('gold')
        cmap.set_under("grey")
        plg = PatchCollection(patches, cmap=cmap, norm=LogNorm(vmin=0.1, vmax=2.38*0.8), linewidth=0, edgecolors='k', alpha=1)  # norm=LogNorm(vmin=0.05, vmax=0.5),  Normalize
        plg.set_array(np.array(cv))
        ax[0].add_collection(plg, autolim=True)
        cbar = plt.colorbar(plg, extend='both') #'neither', 'both', 'min', 'max'
        ax[0].set_title('velocity, km s-1')
        ax[0].set_xlim([0.5,1]) #[0.746, 0.754]  [0.55,0.95]
        ax[0].set_ylim([0.1,0.9]) #[0.492, 0.508]  [0.1,0.9]
        ax[0].set_xticks([])
        ax[0].set_yticks([])


        # mass
        plg = PatchCollection(patches, cmap=cmap, norm=LogNorm(vmin=1e7, vmax=1e11), linewidth=0, edgecolors='k',alpha=1)  # norm=LogNorm(vmin=0.05, vmax=0.5),  Normalize
        plg.set_array(np.array(cM))
        ax[1].add_collection(plg, autolim=True)
        cbar = plt.colorbar(plg, extend='both')  # 'neither', 'both', 'min', 'max'
        ax[1].set_title('Mass, kg')
        ax[1].set_xlim([0.5,1])  # [0.746, 0.754]  [0.55,0.95]
        ax[1].set_ylim([0.1,0.9])  # [0.492, 0.508]  [0.1,0.9]
        ax[1].set_xticks([])
        ax[1].set_yticks([])


        ## energy
        plg = PatchCollection(patches, cmap=cmap, norm=LogNorm(vmin=1e7, vmax=1e11), linewidth=0, edgecolors='k', alpha=1)  # norm=LogNorm(vmin=0.05, vmax=0.5),  Normalize
        plg.set_array(np.array(cEk))
        ax[2].add_collection(plg, autolim=True)
        cbar = plt.colorbar(plg, extend='both') #'neither', 'both', 'min', 'max'
        ax[2].set_title('kinetic energy, J')
        ax[2].set_xlim([0.5,1])  # [0.746, 0.754]  [0.55,0.95]
        ax[2].set_ylim([0,1])  # [0.492, 0.508]  [0.1,0.9]
        ax[2].set_xticks([])
        ax[2].set_yticks([])
        #plt.plot((p[:, 5] + 180) / 360, (p[:, 4] + 90) / 180,'k.')

        plt.savefig(path_save + 'L_1km.png', transparent=True)
        plt.show()

    ### poisson probability for impact flux ###
    elif plt_fig == "poisson_flux":
        k = np.arange(1,30,1)
        lamd_all = [6,10]
        colors = ["k","b","y"]
        fig, ax1 = plt.subplots(1, 1, figsize=(5, 5))
        ax1.set_xlabel('k')
        ax1.set_ylabel('P(X=k)')
        ax1.set_xlim([0,30])
        ax1.set_ylim([0,0.18])
        # ax1.set_xscale("log")
        # ax1.set_yscale("log")
        for j in range(len(lamd_all)):
            lamd=lamd_all[j]
            p = np.zeros(len(k))
            for i in range(len(k)):
                p[i] = (lamd**k[i] * np.exp(-lamd)) / math.factorial(k[i])
            plt.plot(k,p,marker=".",markersize=8,linestyle="-",linewidth =2,color=colors[j])
        plt.savefig('/data/user/tiantians/Python/results/ejectaRISK/poisson_flux.png', transparent=True)
        plt.show()
        print(1)


    i_count+=1





