
#import package

import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from mpl_toolkits.basemap import Basemap


####################
#Overall functions in the notebook
def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap

def obtain_paths():
    savepath='../Chp1/v5/figs/'
    loadpath='../../Year 3/Setonix_results/nc_file/'
    return savepath, loadpath



####################
#generate netCDF
def obtain_cross_area_idx(loadpath,run_number,dz,cross=np.arange(300)):
    ds_KE=xr.open_dataset(loadpath+'run{:02d}_KE_output.nc'.format(run_number))
    rho_mean=ds_KE['rho_mean']
    idx_2d=np.nan_to_num(rho_mean[0,:,:])
    cross_area_idx=np.zeros(len(cross))
    for i in range(len(cross)):
        cross_area_idx[i]=np.count_nonzero(idx_2d[:,i])
    cross_area=cross_area_idx*dz*250e3
    return cross_area_idx,cross_area

def obtain_MLD_for_the_run(loadpath,k):
    ds_MLD=xr.open_dataset(loadpath+'sum_MLD_frontal.nc')
    MLD_obs_sum=ds_MLD['MLD']
    MLD_run_k=MLD_obs_sum[k,...]
    return MLD_run_k


####################
# obtain the parameter set up in each experiment
def obtain_run_case_summary(loadpath):
    '''
    obtain the parameter set up in each experiment
    '''
    # ds=xr.open_dataset(loadpath+'run_case_parameters_summary.nc')
    ds=xr.open_dataset(loadpath+'run_case_parameters_summary_updated.nc')
    run_numbers=ds['run_number']
    run_number_names=ds['run_number_name']
    t0s=ds['t0']
    tis=ds['ti']
    Bs=ds['Bs']
    Slope_burgers=ds['Slope_Burgers']
    alphas=ds['alpha']
    N0s=ds['N0']
    fs=ds['f']

    return run_numbers,run_number_names,t0s,tis,Bs,Slope_burgers,alphas,N0s,fs


####################
# SST satellite plot
def get_SST_n_bath(loadpath):
    ds=xr.open_dataset(loadpath+'H8_May_Oct_2022.nc')
    # Load some bathy data
    bath=xr.open_dataset('Z:/Shared_data/Bathymetry/GA_WEL_NWS_250m_DEM.nc')
    lat=ds['lat']
    lon=ds['lon']
    time=ds['time']
    sst=ds['sea_surface_temperature']-273.15

    return lat,lon,time,sst,bath

def plot_sst_satellite(fig,lat,lon,sst,bath,time,dx=2e3,scale=1e4,i=75,vmin=20,vmax=26):
    #default plotting date is i= 75 in the selected dataset

    ax=fig.add_subplot(111,facecolor='grey')
    cax=ax.pcolormesh(lon,lat,sst[i,...],cmap='inferno',vmin=vmin,vmax=vmax)
    cbar = plt.colorbar(cax, )
    cbar.ax.set_xlabel('SST $[\degree C]$',fontsize=8)
    cbar.ax.xaxis.set_label_position('top')
    title=ax.set_title('Sea Surface Temperature {:}'.format(time[i].values.astype('datetime64[D]').astype('str')),fontsize=10)
    ct= plt.contour(bath['X'],bath['Y'],-bath['topo'],[50,200,],colors='k',linewidths=0.5)
    ctf = plt.contourf(bath['X'],bath['Y'],bath['topo'],[0,1e5],colors='grey')    
    ct_cont= plt.contour(bath['X'],bath['Y'],-bath['topo'],[0],colors='k',linewidths=1) 
    ax.set_xlabel('Longitude [$\degree$E]',fontsize=8)
    ax.set_ylabel('Latitude [$\degree$N]',fontsize=8)
    ax.set_aspect('equal')
    ax.set_xlim(114,119)
    ax.set_ylim(-21.5,-18)

    Tx,Ty = np.gradient(sst[i,...], dx)
    p2 = ax.pcolormesh(lon,lat, np.abs(Tx+1j*Ty)*scale, cmap='bone_r',vmin=0,vmax=1,alpha=0.1)
    #inset an Australia map
    ax2 = fig.add_axes([0.6,0.175,0.15,0.15])
    # Define the basemap
    m = Basemap(projection='merc', llcrnrlat=-45, urcrnrlat=-5, llcrnrlon=105, urcrnrlon=165, ax=ax2)
    # m = Basemap(projection='ortho', lat_0=-25, lon_0=135,  width=8e6, height=8e6, ax=ax2)
    # Draw the coastlines, countries, and states
    m.drawcoastlines()
    m.drawcountries()
    # Draw the ocean and land with custom colors
    m.drawmapboundary(fill_color='lightblue')
    m.fillcontinents(color='grey', lake_color='lightblue')
    # Draw a box for the region between lat (-21.5,-18) and lon (115,119)
    lon1, lat1, lon2, lat2 = 115, -21.5, 119, -18
    x1, y1 = m(lon1, lat1)
    x2, y2 = m(lon2, lat2)
    m.plot([x1, x2], [y1, y1], 'r-')
    m.plot([x1, x2], [y2, y2], 'r-')
    m.plot([x1, x1], [y1, y2], 'r-')
    m.plot([x2, x2], [y1, y2], 'r-')            

    return cax, p2

def obtain_M2_n_PEZ(loadpath):
    ds_L_sum=xr.open_dataset(loadpath+'Summary_L_mix_MLD_M2')
    L_M2_sum=ds_L_sum['L_M2']
    M2_surf_sum=ds_L_sum['M2_bar']
    return L_M2_sum,M2_surf_sum

def plot_baes_PEZ(fig,loadpath,k=1,run_date=np.arange(120),cross=np.arange(300)):
    L_M2_sum,M2_surf_sum=obtain_M2_n_PEZ(loadpath)

    ax1=fig.add_subplot(111)
    PEZ=ax1.plot(L_M2_sum[k,...],run_date,color='k',label='PEZ',linewidth=2)

    cs1=ax1.pcolormesh(cross,run_date,np.abs(M2_surf_sum[k,...]),cmap='inferno',
                    norm=colors.LogNorm(vmin=1e-9,vmax=1e-7))

    ax1.fill_betweenx(np.arange(120),0, L_M2_sum[k,...], color='grey',zorder=10)
    ax1.set_ylim(0,120)
    ax1.set_xlim(150,300)
    plt.legend(fontsize=10,loc='lower left')
    ax1.set_ylabel('Time [day]',fontsize=8)
    ax1.set_xlabel('Cross shelf distance [km]',fontsize=8)


    cbar=plt.colorbar(cs1)
    cbar.set_label('$|M^{2}|$',fontsize=8,labelpad=-30,y=1.05,rotation=0)
    cbar.ax.tick_params(axis='both',which='major',labelsize=8)

    return ax1

def obtain_cs_path(loadpath,sel_lev=[-1e-8]):
    L_M2_sum,M2_surf_sum=obtain_M2_n_PEZ(loadpath)
    cs=plt.contour(np.arange(300),np.arange(120),M2_surf_sum[1,...],levels=sel_lev,colors='k')
    cs_path=cs.collections[0].get_paths()
    cs_path_vertices=np.array(cs_path[0].vertices)
    plt.close()
    return cs_path_vertices

def plot_PEZ_of_new_M2(fig,loadpath,cs_path_vertices,k=1,xlims=(150,300),ylims=(0,120),run_date=np.arange(120),cross=np.arange(300)):
    L_M2_sum,M2_surf_sum=obtain_M2_n_PEZ(loadpath)

    # cmap = plt.get_cmap('inferno')
    # cutted_inferno = truncate_colormap(cmap, 0.2,1)

    ax1=fig.add_subplot(111)
    cs1=ax1.pcolormesh(cross,run_date,np.abs(M2_surf_sum[k,...]),cmap='inferno',
                    norm=colors.LogNorm(vmin=1e-9,vmax=1e-7))
    
    ax1.plot(cs_path_vertices[7::,0],cs_path_vertices[7::,1],color='k')
    ax1.fill_betweenx(y=cs_path_vertices[7::,1], 
                      x1=np.zeros(cs_path_vertices[7::,0].size),
                      x2=cs_path_vertices[7::,0],
                      color='grey',zorder=10)
 
    ax1.set_ylim(ylims)
    ax1.set_xlim(xlims)
    ax1.set_ylabel('Time [day]',fontsize=8)
    ax1.set_xlabel('Cross shelf distance [km]',fontsize=8)
    cbar=plt.colorbar(cs1)
    cbar.set_label('$|M^{2}|$',fontsize=8,labelpad=-30,y=1.05,rotation=0)
    cbar.ax.tick_params(axis='both',which='major',labelsize=8)

    return fig


####################
# Base case characterization plots
def ax_frameless(ax):
    f1=ax.tick_params(axis='x', which='both', bottom=False,top=False, labelbottom=False)
    f2=ax.tick_params(axis='y', which='both', right=False,left=False, labelleft=False)
    ax.yaxis.set_label_coords(-.3, .5)
    return f1, f2

def plot_cross_KE(ax,cross,EKE,MKE,xlims,ylims):
    '''
    plot per unit mass EKE and MKE as a function of cross-shelf location 
    '''
    EKE=ax.plot(cross,EKE,color='r',label='EKE',linewidth=0.75)
    MKE=ax.plot(cross,MKE,color='k',label='MKE',linewidth=0.75)
    ax.set_xlim(xlims)
    ax.set_ylim(ylims)
    ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    # ax.set_xticklabels([])
    # ax.set_yticklabels([])
    # ax.set_ylabel('KE [$m^2/s^2$]',fontsize=8)
    ax.legend(loc='upper left',fontsize=6,frameon=False)
    return EKE, MKE

def plot_rho_w_quiver(ax,cross,along,rho,uu,vv,xlims,ylims,vmin=999.8,vmax=1000.4):
    '''
    plot surface density contourf with depth averaged velocity perturbation quiver (u' v')
    default density range is in [999.8,1000.8]
    noted that the quiver plot for the velocity use every 5km in space
    '''
    cmap = plt.get_cmap('inferno_r')
    cutted_inferno_r = truncate_colormap(cmap, 0, 0.8)
    P=ax.pcolormesh(cross,along, rho, cmap=cutted_inferno_r,
                vmin=vmin,vmax=vmax)
    # Cbar=plt.colorbar(P,cax=cax)
    Q=ax.quiver(cross[::5],along[::5],uu[::5,::5],vv[::5,::5],
                angles='xy',scale_units='xy', scale=0.01,minshaft=2)
    ax.set_xlim(xlims)
    ax.set_ylim(ylims)
    # ax.set_ylabel('Along-shelf distance [km]',fontsize=8)
    ax.patch.set_facecolor('grey')
    return P, Q, 

def plot_meanflow_w_rho(ax,cross,depth,meanflow,rho,
                        xlims,ylims,rho_levels=np.arange(999.8,1000.8,0.05),vmin=-0.05,vmax=0.05):
    '''
    plot surface density contourf with depth averaged velocity perturbation quiver (u' v')
    default density range is in [999.8,1000.8]
    noted that the quiver plot for the velocity use every 5km in space
    '''
    P=ax.pcolormesh(cross,depth,meanflow, cmap='RdBu_r',vmin=vmin,vmax=vmax)
    # Cbar=plt.colorbar(P,cax=cax)
    C=ax.contour(cross,depth,rho,levels=rho_levels,colors='k',linewidths=0.5,)
    C2=ax.contour(cross,depth,rho,levels=[1000.2],colors='k',linewidths=2,)
    C3=ax.contour(cross,depth,rho,levels=[1000.5],colors='gold',linewidths=2,)
    ax.set_xlim(xlims)
    ax.set_ylim(ylims)
    # ax.set_ylabel('Depth [m]',fontsize=8)
    ax.patch.set_facecolor('grey')
    return P, C, 

def obtain_base_case_data(loadpath,run_number=32):

    # subds1=xr.open_dataset(loadpath+'run{:02d}_EnergyBudget_VA_Domain_n_PEZ.nc'.format(run_number))
    # EKE_pu=subds1['EKE_domain_vol_ave']
    # MKE_pu=subds1['MKE_domain_vol_ave']

    ds_baseKE=xr.open_dataset(loadpath+'run{:02d}_KE_output.nc'.format(run_number))
    EKE_pu=ds_baseKE['unit_area_EKE']
    MKE_pu=ds_baseKE['unit_area_MKE']

    subds3=xr.open_dataset(loadpath+'run{:02d}_KE_output.nc'.format(run_number))
    rho_bar=subds3['rho_mean']
    u_bar=subds3['u_mean']
    layer_depth=subds3['layer_depth']
    depth=-layer_depth

    subds4=xr.open_dataset(loadpath+'run{:02d}_vel_V.nc'.format(run_number))
    v_bar=subds4['V_mean']

    ds_new_coord=xr.open_dataset(loadpath+'run{:02d}_new_coord.nc'.format(run_number))
    up_bbl=ds_new_coord['up_bbl_coord']
    vp_bbl=ds_new_coord['vp_bbl_coord']  
    rho_bbl=ds_new_coord['rho_bbl_coord']


    return EKE_pu,MKE_pu,rho_bbl,up_bbl,vp_bbl,v_bar,u_bar,rho_bar,depth

def plot_snapshots_basecase_w_PEZ_in_time(fig,L_M2_sum,sel_dates,loadpath,
                                          xlims,ylims_topview,ylims_sideview,ylims_KE,
                                          
                                          cross=np.arange(300),along=np.arange(250)):
    subplot_index=['a','b','c','d','e','f','g','h','i','j','k','l',]
    EKE_ua,MKE_ua,rho_bbl,up_bbl,vp_bbl,v_bar,u_bar,rho_bar,depth=obtain_base_case_data(loadpath)
    gs = fig.add_gridspec(nrows=4, ncols=4, height_ratios=[0.25,1.0,0.5,0.5],
                width_ratios=[1,1,1,0.05],hspace=0.25,wspace=0.4)
    ax_1=fig.add_subplot(gs[0, 0])
    ax_1.set_ylabel('EKE $[m^{2}/s^{2}]$',fontsize=8)
    f1,f2=ax_frameless(ax_1)
    ax_1.yaxis.set_label_coords(-.15, .5)

    ax_2=fig.add_subplot(gs[1, 0])
    f1,f2=ax_frameless(ax_2)
    ax_2.set_ylabel('Along-shelf distance [km]',fontsize=8)
    ax_2.yaxis.set_label_coords(-.25, .5)
    # ax2.set_aspect('equal')

    ax_3=fig.add_subplot(gs[2, 0])
    f1,f2=ax_frameless(ax_3)
    ax_3.set_ylabel('Depth [m]',fontsize=8)
    ax_3.yaxis.set_label_coords(-.3, .5)

    ax_4=fig.add_subplot(gs[3, 0])
    f1,f2=ax_frameless(ax_4)
    ax_4.set_ylabel('Depth [m]',fontsize=8)
    ax_4.yaxis.set_label_coords(-.3, .5)

    for col in range(3):
        i=sel_dates[col]
        frontal_reg=L_M2_sum[1,i]

        ax1 = fig.add_subplot(gs[0, col])
        subplot_int1=int(3*0+col)
        EKE,MKE=plot_cross_KE(ax1,cross,EKE_ua[i,...],MKE_ua[i,...],xlims,ylims_KE)
        ax1.set_title('Day {:01d}'.format(i),fontsize=8)
        ax1.axvspan(frontal_reg,300,color='lightblue')
        ax1.text(0.02, 0.85, subplot_index[subplot_int1], 
                transform=ax1.transAxes,zorder=10,fontsize=8)
        
        ax2=fig.add_subplot(gs[1,col])
        subplot_int2=int(3*1+col)

        up_sel=up_bbl[i,2,...]
        vp_sel=vp_bbl[i,2,...]

        smooth_up=up_sel.rolling(along_shelf=5,cross_shelf=5,center=True).mean(
            dim={"along_shelf","cross_shelf"},skipna=True)
        smooth_vp=vp_sel.rolling(along_shelf=5,cross_shelf=5,center=True).mean(
            dim={"along_shelf","cross_shelf"},skipna=True)


        p2,q2=plot_rho_w_quiver(ax2,cross,along,rho_bbl[i,2,...],
                smooth_up,smooth_vp,xlims,ylims_topview,vmin=1000.1,vmax=1000.5)
        ax2.text(0.02, 0.95, subplot_index[subplot_int2], 
                transform=ax2.transAxes,zorder=10,fontsize=8)
    #     ax2.set_aspect('equal')

        ax3=fig.add_subplot(gs[2,col])
        subplot_int3=int(3*2+col)
        p3,c3,=plot_meanflow_w_rho(ax3,cross,depth,v_bar[i,...],rho_bar[i,...],
                xlims,ylims_sideview,rho_levels=np.arange(999.0,1002.7,0.05),vmax=0.05,vmin=-0.05)
        ax3.text(0.02, 0.9, subplot_index[subplot_int3], 
                transform=ax3.transAxes,zorder=10,fontsize=8)
        

        ax4=fig.add_subplot(gs[3,col])
        subplot_int4=int(3*3+col)
        p4,c4,=plot_meanflow_w_rho(ax4,cross,depth,u_bar[i,...],rho_bar[i,...],
                xlims,ylims_sideview,rho_levels=np.arange(999.0,1002.7,0.05),vmax=0.02,vmin=-0.02)
        ax4.set_xlabel('Cross-shelf distance [km]',fontsize=8)
        ax4.text(0.02, 0.9, subplot_index[subplot_int4], transform=ax4.transAxes,zorder=10,fontsize=8)
        

    cax2=fig.add_subplot(gs[1,3])
    cbar2=plt.colorbar(p2,cax=cax2)
    cbar2.ax.set_title(r'$\rho$',fontsize=8,rotation=0,)
    cax2.quiverkey(q2, 1.25, 1.2, 0.2, r'u=0.2 m/s', fontproperties={'size': 6})
    cax3=fig.add_subplot(gs[2,3])
    cbar3=plt.colorbar(p3,cax=cax3)
    cbar3.ax.set_title('$\overline{V}$',fontsize=8,rotation=0)
    cax4=fig.add_subplot(gs[3,3])
    cbar4=plt.colorbar(p4,cax=cax4)
    cbar4.ax.set_title('$\overline{U}$',fontsize=8,rotation=0)

    plt.tight_layout()

    return fig

def plot_snapshots_basecase_in_time(fig,sel_dates,loadpath,
                                        xlims,ylims_topview,ylims_sideview,ylims_KE,
                                        cross=np.arange(300),along=np.arange(250)):
    
    EKE_ua,MKE_ua,rho_bbl,up_bbl,vp_bbl,v_bar,u_bar,rho_bar,depth=obtain_base_case_data(loadpath)
    subplot_index=['a','b','c','d','e','f','g','h','i','j','k','l',]
    gs = fig.add_gridspec(nrows=4, ncols=4, height_ratios=[0.25,1.0,0.5,0.5],
                width_ratios=[1,1,1,0.05],hspace=0.25,wspace=0.4)
    ax_1=fig.add_subplot(gs[0, 0])
    ax_1.set_ylabel('EKE $[m^{2}/s^{2}]$',fontsize=8)
    f1,f2=ax_frameless(ax_1)
    ax_1.yaxis.set_label_coords(-.15, .5)

    ax_2=fig.add_subplot(gs[1, 0])
    f1,f2=ax_frameless(ax_2)
    ax_2.set_ylabel('Along-shelf distance [km]',fontsize=8)
    ax_2.yaxis.set_label_coords(-.25, .5)
    # ax2.set_aspect('equal')

    ax_3=fig.add_subplot(gs[2, 0])
    f1,f2=ax_frameless(ax_3)
    ax_3.set_ylabel('Depth [m]',fontsize=8)
    ax_3.yaxis.set_label_coords(-.3, .5)

    ax_4=fig.add_subplot(gs[3, 0])
    f1,f2=ax_frameless(ax_4)
    ax_4.set_ylabel('Depth [m]',fontsize=8)
    ax_4.yaxis.set_label_coords(-.3, .5)

    for col in range(3):
        i=sel_dates[col]
        ax1 = fig.add_subplot(gs[0, col])
        subplot_int1=int(3*0+col)
        EKE,MKE=plot_cross_KE(ax1,cross,EKE_ua[i,...],MKE_ua[i,...],xlims,ylims_KE)
        ax1.set_title('Day {:01d}'.format(i),fontsize=8)
        ax1.text(0.02, 0.85, subplot_index[subplot_int1], 
                transform=ax1.transAxes,zorder=10,fontsize=8)
        
        ax2=fig.add_subplot(gs[1,col])
        subplot_int2=int(3*1+col)

        up_sel=up_bbl[i,2,...]
        vp_sel=vp_bbl[i,2,...]
        smooth_up=up_sel.rolling(along_shelf=5,cross_shelf=5,center=True).mean(
            dim={"along_shelf","cross_shelf"},skipna=True)
        smooth_vp=vp_sel.rolling(along_shelf=5,cross_shelf=5,center=True).mean(
            dim={"along_shelf","cross_shelf"},skipna=True)

        p2,q2=plot_rho_w_quiver(ax2,cross,along,rho_bbl[i,2,...],
                smooth_up,smooth_vp,xlims,ylims_topview,vmin=1000.1,vmax=1000.5)

        ax2.text(0.02, 0.95, subplot_index[subplot_int2], 
                transform=ax2.transAxes,zorder=10,fontsize=8)

        ax3=fig.add_subplot(gs[2,col])
        subplot_int3=int(3*2+col)
        p3,c3,=plot_meanflow_w_rho(ax3,cross,depth,v_bar[i,...],rho_bar[i,...],
                xlims,ylims_sideview,rho_levels=np.arange(999.0,1002.7,0.05),vmax=0.05,vmin=-0.05)
        ax3.text(0.02, 0.9, subplot_index[subplot_int3], 
                transform=ax3.transAxes,zorder=10,fontsize=8)
        
        ax4=fig.add_subplot(gs[3,col])
        subplot_int4=int(3*3+col)
        p4,c4,=plot_meanflow_w_rho(ax4,cross,depth,u_bar[i,...],rho_bar[i,...],
                xlims,ylims_sideview,rho_levels=np.arange(999.0,1002.7,0.05),vmax=0.02,vmin=-0.02)
        ax4.set_xlabel('Cross-shelf distance [km]',fontsize=8)
        ax4.text(0.02, 0.9, subplot_index[subplot_int4], transform=ax4.transAxes,zorder=10,fontsize=8)
        

    cax2=fig.add_subplot(gs[1,3])
    cbar2=plt.colorbar(p2,cax=cax2)
    cbar2.ax.set_title(r'$\rho$',fontsize=8,rotation=0,)
    cax2.quiverkey(q2, 1.16, 0.03, 0.2, r'u=0.2 m/s', fontproperties={'size': 6})
    cax3=fig.add_subplot(gs[2,3])
    cbar3=plt.colorbar(p3,cax=cax3)
    cbar3.ax.set_title('$\overline{V}$',fontsize=8,rotation=0)
    cax4=fig.add_subplot(gs[3,3])
    cbar4=plt.colorbar(p4,cax=cax4)
    cbar4.ax.set_title('$\overline{U}$',fontsize=8,rotation=0)

    plt.tight_layout()

    return fig

def load_data_for_plane_comparsion(loadpath,run_number=32):
    # ds_abs=xr.open_dataset(loadpath+'run{:02d}_abs_plane_rho_vel.nc'.format(run_number))
    ds_abs=xr.open_dataset(loadpath+'run{:02d}_abs_plane_rho_vel_update.nc'.format(run_number))
    yl=ds_abs['along_shelf']
    xl=ds_abs['cross_shelf']
    rho_abs_2yr_t0_base=ds_abs['rho_2lr_abs']
    Ro_abs_2yr_t0_base=ds_abs['Ro_2lr_abs']
    rho_surf_t0_base=ds_abs['rho_surf']
    Ro_surf_t0_base=ds_abs['Ro_surf']

    return xl,yl,rho_abs_2yr_t0_base,Ro_abs_2yr_t0_base,rho_surf_t0_base,Ro_surf_t0_base

def plot_two_plane_comparsion(fig,loadpath,rho_level=np.arange(1000.1,1002.7,0.05)):
    xl,yl,rho_abs_2yr_t0_base,Ro_abs_2yr_t0_base,rho_surf_t0_base,Ro_surf_t0_base=load_data_for_plane_comparsion(loadpath)
    gs = fig.add_gridspec(nrows=1, ncols=3, width_ratios=[1,1,0.05],wspace=0)
    ax=fig.add_subplot(gs[0,0])
    cax=ax.pcolormesh(xl/1000,yl/1000,Ro_surf_t0_base,cmap='RdBu',vmax=1,vmin=-1)
    # plt.colorbar(cax)
    ax.contour(xl/1000,yl/1000,rho_surf_t0_base,
                levels=rho_level,colors='k',linewidths=1,zorder=1)
    ax.contour(xl/1000,yl/1000,rho_surf_t0_base,
                levels=[1000.2],colors='k',linewidths=2,)
    ax.set_xlim(150,300)
    ax.set_ylim(0,250)
    ax.set_aspect('equal')
    ax.set_xlabel('Cross-shelf distance [km]',fontsize=8)
    ax.set_ylabel('Along-shelf distance [km]',fontsize=8)

    ax2=fig.add_subplot(gs[0,1])
    cax2=ax2.pcolormesh(xl/1000,yl/1000,Ro_abs_2yr_t0_base,cmap='RdBu',vmax=1,vmin=-1)
    ax2.contour(xl/1000,yl/1000,rho_abs_2yr_t0_base,
                levels=rho_level,colors='k',linewidths=1,zorder=1)
    ax2.contour(xl/1000,yl/1000,rho_abs_2yr_t0_base,
                levels=[1000.2],colors='k',linewidths=2,)
    ax2.set_xlim(150,300)
    ax2.set_ylim(0,250)
    ax2.set_aspect('equal')
    ax2.set_xlabel('Cross-shelf distance [km]',fontsize=8)
    ax.text(0.021, 0.95, 'a', transform=ax.transAxes,zorder=10,fontsize=8)
    ax2.text(0.021, 0.95, 'b', transform=ax2.transAxes,zorder=10,fontsize=8)
    cax3=fig.add_subplot(gs[0,2])
    cbar=plt.colorbar(cax2,cax=cax3)
    cbar.ax.set_title(r'$Ro$',fontsize=8,rotation=0,)

    return fig


####################
# Energetics plots

def obtain_vol_ave_data_for_energy_analysis(loadpath,run_number=32):
    ds_KE=xr.open_dataset(loadpath+'run{:02d}_KE_output.nc'.format(run_number))
    reference_array=ds_KE['rho_mean'][0,...]
    idx_2d=np.nan_to_num(reference_array)
    layer_depth=ds_KE['layer_depth']
    dz=layer_depth.values[1]-layer_depth.values[0]
    dep_ave_EKE=ds_KE['unit_area_EKE']
    dep_ave_TKE=ds_KE['unit_area_TKE']
    dep_ave_MKE=ds_KE['unit_area_MKE']

    ds_budget=xr.open_dataset(loadpath+'run{:02d}_multi_calc_output.nc'.format(run_number))
    depth_int_bc=ds_budget['depth_int_baroclinic_instability']
    depth_int_hs=ds_budget['depth_int_horizontal_shear_instability']
    depth_int_vs=ds_budget['depth_int_vertical_shear_instability']

    xl=np.arange(300)
    cross_area_idx=np.zeros(len(xl))
    for i in range(len(xl)):
        cross_area_idx[i]=np.count_nonzero(idx_2d[:,i])
        
    cross_area=cross_area_idx*dz*250e3
    EKE_pu=dep_ave_EKE.mean(skipna='True',axis=1)
    TKE_pu=dep_ave_TKE.mean(skipna='True',axis=1)
    MKE_pu=dep_ave_MKE.mean(skipna='True',axis=1)
    bc_dep_ave=depth_int_bc/cross_area
    hs_dep_ave=depth_int_hs/cross_area
    vs_dep_ave=depth_int_vs/cross_area
    bc_pu=bc_dep_ave.mean(skipna='True',axis=1)
    hs_pu=hs_dep_ave.mean(skipna='True',axis=1)
    vs_pu=vs_dep_ave.mean(skipna='True',axis=1)


    ds_eng=xr.open_dataset(loadpath+'run{:02d}_buoyancy_trans_vol_ave.nc'.format(run_number))
    # eddy_trans_pu=ds_eng['eddy_trans_vol_ave']
    # mean_trans_pu=ds_eng['mean_trans_vol_ave']
    eddy_trans_pu=ds_eng['eddy_trans_vol_ave']
    mean_trans_pu=ds_eng['mean_trans_vol_ave'] 

    return hs_pu,vs_pu,bc_pu,TKE_pu,EKE_pu,MKE_pu,eddy_trans_pu,mean_trans_pu

def plot_KEs(ax,TKE,MKE,EKE,ti,t0,xlims,ylims=(1e-8,1e-2),run_date=np.arange(120)):
    p1=ax.plot(run_date,TKE,label=r"$TKE$",color='k',)#linestyle='dashed')
    p2=ax.plot(run_date,MKE,label=r"$MKE$",color='b',)#linestyle='dashed')
    p3=ax.plot(run_date,EKE,label=r"$EKE$",color='r')
    ax.scatter(ti,EKE[ti],marker='o',color='k',s=15,zorder=2)
    # ax.scatter(t0,EKE[t0],marker='x',color='k',s=15,zorder=2)
    ax.set_xlim(xlims)
    ax.set_ylim(ylims)
    ax.set_yscale('log')
    ax.set_ylabel('Kinetic Energy \n $[m^{2}/s^{2}]$',fontsize=8)
    ax.tick_params(axis='both',which='major',labelsize=8)
    ax.text(0.1,0.80,'TKE',fontsize=6,color='k',transform=ax.transAxes,)
    ax.text(0.45,0.40,'MKE',fontsize=6,color='b',transform=ax.transAxes,)
    ax.text(0.3,0.75,'EKE',fontsize=6,color='r',transform=ax.transAxes,)

    return p1,p2,p3

def plot_energy_budget(ax,hs,vs,bc,ti,t0,xlims,ylims=(-0.5e-8,3e-8),run_date=np.arange(120)):
    p1=ax.plot(run_date,hs,label=r"$\langle v'u' \rangle V_x $",color='b',linestyle='dashed')
    p2=ax.plot(run_date,vs,label=r"$\langle v'w' \rangle V_z $",color='r',linestyle='dashed')
    p3=ax.plot(run_date,bc,label=r"$\langle w'b' \rangle$",color='k')
    ax.text(0.8,0.15,r"$\langle v'u' \rangle V_x $",color='b',fontsize=6,transform=ax.transAxes,)
    ax.text(0.6,0.15,r"$\langle v'w' \rangle V_z $",color='r',fontsize=6,transform=ax.transAxes,)
    ax.text(0.65,0.55,r"$\langle w'b' \rangle$",color='k',fontsize=6,transform=ax.transAxes,)
    ax.scatter(ti,bc[ti],marker='o',color='k',s=15,zorder=2)
    ax.scatter(t0,bc[t0],marker='x',color='k',s=15,zorder=2)
    ax.set_xlim(xlims)
    ax.set_ylim(ylims)
    ax.set_ylabel('EKE conversion \n $[m^{2}/s^{3}]$',fontsize=8)
    ax.tick_params(axis='both',which='major',labelsize=8)
    ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))

    return p1,p2,p3

def plot_trans(ax,eddy_trans,mean_trans,ti,t0,xlims,ylims=(-2e-3,5e-3),run_date=np.arange(120)):
    p2=ax.plot(run_date,mean_trans,color='b')
    p1=ax.plot(run_date,eddy_trans,color='r')
    # ax.scatter(ti,eddy_trans[ti],marker='o',color='k',s=15,zorder=2)
    ax.scatter(t0,eddy_trans[t0],marker='x',color='k',s=15,zorder=2)
    ax.set_xlim(xlims)
    # ax.axhline(y=0,linestyle='dashed',color='k')
    ax.set_ylim(ylims)
    ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    ax.set_ylabel(r'Magnitude of $u\rho$'+ '\n $[kg/m^2s]$',fontsize=8)
    ax.text(0.15,0.5,'by mean flow',fontsize=6,color='b',transform=ax.transAxes,)
    ax.text(0.7,0.65,'by eddy ',fontsize=6,color='r',transform=ax.transAxes,)
    ax.tick_params(axis='both',which='major',labelsize=8)

    return p1,p2

def plot_energetic_analysis(fig,loadpath,xlims,ylims_KEs,ylims_budget,ylims_trans,k=1,run_date=np.arange(120)):
    hs_pu,vs_pu,bc_pu,TKE_pu,EKE_pu,MKE_pu,eddy_trans_pu,mean_trans_pu= obtain_vol_ave_data_for_energy_analysis(loadpath)
    run_numbers,run_number_names,t0s,tis,Bs,Slope_burgers,alphas,N0s,fs=obtain_run_case_summary(loadpath)
    ti=int(tis[k])
    t0=int(t0s[k])
    gs = fig.add_gridspec(nrows=3, ncols=1,wspace=0.3)
    ax1=fig.add_subplot(gs[0,0])
    p4,p5,p6=plot_KEs(ax1,TKE_pu,MKE_pu,EKE_pu,ti,t0,xlims,ylims_KEs,)
    ax1.set_xticklabels([])
    ax1.text(0.02, 0.9, 'a', transform=ax1.transAxes,zorder=10,fontsize=8)

    ax2=fig.add_subplot(gs[1,0])
    p1,p2,p3=plot_energy_budget(ax2,hs_pu,vs_pu,bc_pu,ti,t0,xlims,ylims_budget,)
    ax2.set_xticklabels([])
    ax2.text(0.02, 0.9, 'b', transform=ax2.transAxes,zorder=10,fontsize=8)

    ax3=fig.add_subplot(gs[2,0])
    p7,p8=plot_trans(ax3,np.abs(eddy_trans_pu),np.abs(mean_trans_pu),ti,t0,xlims,ylims_trans,)
    ax3.set_xlabel('Time [day]',fontsize=8)
    ax3.text(0.02, 0.9, 'c', transform=ax3.transAxes,zorder=10,fontsize=8)
    plt.tight_layout()

    return fig

######################
# Case comparsion
def obtain_scenario(scenario):
    case_numbers=np.array([[39,32,38],[35,32,34],[33,32,16]]) 
    rename_case_numbers1=np.array([[6,1,5],[4,1,3],[2,1,7]]) 
    rename_case_numbers2=np.array([['6, \u03B1 =0.05%','1, \u03B1 =0.1%','5, \u03B1 =0.2%'],
                                  ['4, N0=5e-3 s^-1','1, N0=7e-3 s^-1','3, N0=1e-2 s^-1'],
                                  ['2, Tide and H=-100 Wm^-2','1, H=-100 Wm^-2','7, H=-50 Wm^-2']]) 
    
    case_k=np.array([[11,1,9],[7,1,5],[3,1,0]])
    if str(scenario)=='alpha':
        case_group_idx=0
    elif str(scenario)=='N0':
        case_group_idx=1
    elif str(scenario)=='force':
        case_group_idx=2
    
    run_number_sum=case_numbers[case_group_idx]
    run_number_name_sum1=rename_case_numbers1[case_group_idx]
    run_number_name_sum2=rename_case_numbers2[case_group_idx]
    run_idx_sum=case_k[case_group_idx]


    return run_number_sum,run_number_name_sum1,run_number_name_sum2,run_idx_sum

def plot_cross_EKE(ax,cross,EKE,xlims=(150,300),ylims=(0,5e-3)):
    p1=ax.plot(cross,EKE,color='k')
    ax.set_xlim(xlims)
    ax.set_ylim(ylims)
    ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    ax.tick_params(axis='both',which='major',labelsize=8)
    ax.grid(linewidth=0.5,axis='both')

    return p1

def plot_case_compare_snapshots(fig,scenario,t0s,loadpath,
                                xlims=(150,300),
                                ylims_KEs=(0,3e-3),
                                ylims_topview=(0,250),
                                ylims_sideview=(-220,0),
                                cross=np.arange(300),along=np.arange(250),
                                rho_max=1000.7,rho_min=1000.1,
                                Vbar_max=0.05,Vbar_min=-0.05,
                                Ubar_max=0.01,Ubar_min=-0.01):
    subplot_index=['a','b','c','d','e','f','g','h','i','j','k','l',]
    run_number_sum,run_number_name_sum,run_number_name_sum2,run_idx_sum=obtain_scenario(str(scenario))
    gs = fig.add_gridspec(nrows=4, ncols=4, height_ratios=[0.25,0.9,0.5,0.5],
                        width_ratios=[1,1,1,0.05],hspace=0.25,wspace=0.4)
    ax_1=fig.add_subplot(gs[0, 0])
    ax_1.set_ylabel('EKE $[m^{2}/s^{2}]$',fontsize=8)
    f1,f2=ax_frameless(ax_1)
    ax_2=fig.add_subplot(gs[1, 0])
    f1,f2=ax_frameless(ax_2)
    ax_2.set_ylabel('Along-shelf distance [km]',fontsize=8)
    ax_3=fig.add_subplot(gs[2, 0])
    f1,f2=ax_frameless(ax_3)
    ax_3.set_ylabel('Depth [m]',fontsize=8)
    ax_4=fig.add_subplot(gs[3, 0])
    f1,f2=ax_frameless(ax_4)
    ax_4.set_ylabel('Depth [m]',fontsize=8)
    ax_4.yaxis.set_label_coords(-.3, .5)
    
    for col in range(3):
        
        run_number=run_number_sum[col]
        run_number_name=run_number_name_sum[col]
        run_number_idx=run_idx_sum[col]
        t0=int(t0s[run_number_idx])

        subds1=xr.open_dataset(loadpath+'run{:02d}_abs_plane_rho_vel_update.nc'.format(run_number))
        rho_2lr_abs=subds1['rho_2lr_abs']
        up_2lr_abs=subds1['up_2lr_abs']
        vp_2lr_abs=subds1['vp_2lr_abs']

        subds3=xr.open_dataset(loadpath+'run{:02d}_KE_output.nc'.format(run_number))
        EKE_ua=subds3['unit_area_EKE']
        rho_mean=subds3['rho_mean']
        u_bar=subds3['u_mean']
        layer_depth=subds3['layer_depth']
        depth=-layer_depth
        
        subds4=xr.open_dataset(loadpath+'run{:02d}_vel_V.nc'.format(run_number))
        v_bar=subds4['V_mean']

        ax1 = fig.add_subplot(gs[0, col])
        p1=plot_cross_EKE(ax1,cross,EKE_ua[t0,...],xlims,ylims_KEs)
        ax1.set_title(r'Run {:01d}'.format(run_number_name),fontsize=8)

        ax2=fig.add_subplot(gs[1,col])
        p2,q2=plot_rho_w_quiver(ax2,cross,along,rho_2lr_abs,
                up_2lr_abs,vp_2lr_abs,xlims,ylims_topview,vmin=rho_min,vmax=rho_max)
        ax2.tick_params(axis='both',which='major',labelsize=8)
        ax2.set_aspect('equal')
        
        ax3=fig.add_subplot(gs[2,col])
        p3,c3,=plot_meanflow_w_rho(ax3,cross,depth,v_bar[t0,...],rho_mean[t0,...],
                xlims,ylims_sideview,rho_levels=np.arange(1000.1,1002.7,0.05),vmax=Vbar_max,vmin=Vbar_min)
        ax3.tick_params(axis='both',which='major',labelsize=8)

        ax4=fig.add_subplot(gs[3,col])
        p4,c4,=plot_meanflow_w_rho(ax4,cross,depth,u_bar[t0,...],rho_mean[t0,...],
                xlims,ylims_sideview,rho_levels=np.arange(1000.1,1002.7,0.05),vmax=Ubar_max,vmin=Ubar_min)
        ax4.tick_params(axis='both',which='major',labelsize=8)
        ax4.set_xlabel('Cross-shelf distance [km]',fontsize=8)

        subplot_int1=int(3*0+col)
        subplot_int2=int(3*1+col)
        subplot_int3=int(3*2+col)
        subplot_int4=int(3*3+col)
        ax1.text(0.05, 0.85, subplot_index[subplot_int1], 
                transform=ax1.transAxes,zorder=10,fontsize=8)
        ax2.text(0.05, 0.95, subplot_index[subplot_int2], 
                transform=ax2.transAxes,zorder=10,fontsize=8)
        ax3.text(0.05, 0.9, subplot_index[subplot_int3], 
                transform=ax3.transAxes,zorder=10,fontsize=8)
        ax4.text(0.05, 0.9, subplot_index[subplot_int4], 
                transform=ax4.transAxes,zorder=10,fontsize=8)
        
    cax2=fig.add_subplot(gs[1,3])
    cbar2=plt.colorbar(p2,cax=cax2)
    cbar2.ax.set_title(r'$\rho$',fontsize=8,rotation=0,)
    cax2.quiverkey(q2, 1.25, 1.2, 0.2, r'u=0.2 m/s', fontproperties={'size': 6})
    cax3=fig.add_subplot(gs[2,3])
    cbar3=plt.colorbar(p3,cax=cax3)
    cbar3.ax.set_title('$\overline{V}$',fontsize=8,rotation=0)
    cax4=fig.add_subplot(gs[3,3])
    cbar4=plt.colorbar(p4,cax=cax4)
    cbar4.ax.set_title('$\overline{U}$',fontsize=8,rotation=0)
    plt.tight_layout()

    return fig

def obtain_domain_ave_EKE_n_bc(loadpath,run_number):
    """
    the way of calculating the domain volume averaged in this function has been deprecated 
    as it will leads to calculation error (weighting)
    """
    ds_KE=xr.open_dataset(loadpath+'run{:02d}_KE_output.nc'.format(run_number))
    dep_int_EKE=ds_KE['depth_int_EKE']
    rho_bar=ds_KE['rho_mean']
    layer_depth=ds_KE['layer_depth']
    dz=layer_depth[1]-layer_depth[0]

    ds_budget=xr.open_dataset(loadpath+'run{:02d}_multi_calc_output.nc'.format(run_number))
    depth_int_bc=ds_budget['depth_int_baroclinic_instability']
    total_int_bc=np.nansum(depth_int_bc,axis=1)*1e3
    total_int_EKE=np.nansum(dep_int_EKE,axis=1)*1e3
    idx_2d=np.nan_to_num(rho_bar[0,:,:])
    total_grids=np.count_nonzero(idx_2d)
    total_vol=total_grids*1e3*1e3*dz*250
    domain_vol_ave_EKE=total_int_EKE/(total_vol.values)
    domain_vol_ave_bc=total_int_bc/(total_vol.values)

    return domain_vol_ave_EKE,domain_vol_ave_bc

def obtain_domain_ave_EKE_n_bc_v2(loadpath,run_number):
    ds_KE=xr.open_dataset(loadpath+'run{:02d}_KE_output.nc'.format(run_number))
    EKE_ua=ds_KE['unit_area_EKE']
    domain_vol_ave_EKE=EKE_ua.mean(skipna='True',axis=1)
    rho_bar=ds_KE['rho_mean']
    layer_depth=ds_KE['layer_depth']
    dz=(layer_depth[1]-layer_depth[0]).values

    ds_budget=xr.open_dataset(loadpath+'run{:02d}_multi_calc_output.nc'.format(run_number))
    depth_int_bc=ds_budget['depth_int_baroclinic_instability']
    xl=np.arange(300)
    idx_2d=np.nan_to_num(rho_bar[0,:,:])
    local_area=np.trapz(np.trapz(idx_2d,axis=1,dx=1e3),dx=dz)
    cross_area_idx=np.zeros(len(xl))
    for i in range(len(xl)):
        cross_area_idx[i]=np.count_nonzero(idx_2d[:,i])
    cross_area=cross_area_idx*dz*250e3

    bc_ua=depth_int_bc/cross_area
    domain_vol_ave_bc=bc_ua.mean(skipna='True',axis=1)

    return domain_vol_ave_EKE,domain_vol_ave_bc

def obtain_domain_ave_EKE_n_bc_n_EKEdt(loadpath,run_number):
    ds_KE=xr.open_dataset(loadpath+'run{:02d}_KE_output.nc'.format(run_number))
    EKE_ua=ds_KE['unit_area_EKE']
    domain_vol_ave_EKE=EKE_ua.mean(skipna='True',axis=1)
    EKE_dt=np.gradient(domain_vol_ave_EKE,86400)
    rho_bar=ds_KE['rho_mean']
    layer_depth=ds_KE['layer_depth']
    dz=(layer_depth[1]-layer_depth[0]).values

    ds_budget=xr.open_dataset(loadpath+'run{:02d}_multi_calc_output.nc'.format(run_number))
    depth_int_bc=ds_budget['depth_int_baroclinic_instability']
    xl=np.arange(300)
    idx_2d=np.nan_to_num(rho_bar[0,:,:])
    local_area=np.trapz(np.trapz(idx_2d,axis=1,dx=1e3),dx=dz)
    cross_area_idx=np.zeros(len(xl))
    for i in range(len(xl)):
        cross_area_idx[i]=np.count_nonzero(idx_2d[:,i])
    cross_area=cross_area_idx*dz*250e3

    bc_ua=depth_int_bc/cross_area
    domain_vol_ave_bc=bc_ua.mean(skipna='True',axis=1)

    return domain_vol_ave_EKE,domain_vol_ave_bc,EKE_dt

def plot_case_compare_wb_EKE(fig,scenario,tis,t0s,loadpath,
                         xlims=(0,60),
                         ylims_EKEs=(0,0.5e-3),
                         ylims_wbs=(0,2e-8),
                         run_date=np.arange(120)):
    run_number_sum,run_number_name_sum,run_number_name_sum2,run_idx_sum=obtain_scenario(str(scenario))
    color_group=['b','k','r']
    ax1=fig.add_subplot(121)
    ax2=fig.add_subplot(122)
    for k in range(len(run_number_sum)):
        run_number=run_number_sum[k]
        run_idx=int(run_idx_sum[k])
        ti=int(tis[run_idx])
        t0=int(t0s[run_idx])

        # EKE_VA,bc_VA=obtain_domain_ave_EKE_n_bc(loadpath,run_number)
        EKE_VA,bc_VA=obtain_domain_ave_EKE_n_bc_v2(loadpath,run_number)
        # ax1.plot(run_date,EKE_VA,label='run {:02d}'.format(run_number_name_sum[k]),color=color_group[k])
        ax1.plot(run_date,EKE_VA,label='run '+ '{}'.format(run_number_name_sum2[k]),color=color_group[k])
        ax1.scatter(ti,EKE_VA[ti],marker='o',color=color_group[k],s=20,zorder=2)
        ax1.scatter(t0,EKE_VA[t0],marker='x',color=color_group[k],s=20,zorder=2)

        # ax2.plot(run_date,bc_VA,label='run {:02d}'.format(run_number_name_sum[k]),color=color_group[k])
        ax2.plot(run_date,bc_VA,label='run '+ '{}'.format(run_number_name_sum2[k]),color=color_group[k])
        ax2.scatter(ti,bc_VA[ti],marker='o',color=color_group[k],s=20,zorder=2)
        ax2.scatter(t0,bc_VA[t0],marker='x',color=color_group[k],s=20,zorder=2)

    ax1.set_xlim(xlims)
    ax1.set_ylim(ylims_EKEs)
    ax1.legend(fontsize=7)
    ax1.tick_params(axis='both',which='major',labelsize=8)
    ax1.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    ax1.set_ylabel("Volume averaged EKE $[m^2/s^2]$",fontsize=8)
    ax1.set_xlabel('Time [day]',fontsize=8)

    ax2.set_xlim(xlims)
    ax2.set_ylim(ylims_wbs)
    ax2.legend(fontsize=7)
    ax2.tick_params(axis='both',which='major',labelsize=8)
    ax2.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    ax2.set_ylabel("Volume averaged $w'b' \;[m^2/s^3]$",fontsize=8)
    ax2.set_xlabel('Time [day]',fontsize=8)

    # add subplot label
    ax1.text(0.02, 0.95, '{}'.format(chr(97)), transform=ax1.transAxes,zorder=20,fontsize=9)
    ax2.text(0.02, 0.95, '{}'.format(chr(98)), transform=ax2.transAxes,zorder=20,fontsize=9)

    ax1.grid(alpha=0.5)
    ax2.grid(alpha=0.5)
    plt.tight_layout()

    return fig

def plot_case_compare_wb_EKE_addon(fig,scenario,tis,t0s,loadpath,
                         xlims=(0,60),
                         ylims_EKEs=(0,0.5e-3),
                         ylims_wbs=(0,2e-8),
                         ylims_EKE_dt=(0,1e-9),
                         run_date=np.arange(120)):
    run_number_sum,run_number_name_sum,run_number_name_sum2,run_idx_sum=obtain_scenario(str(scenario))
    color_group=['b','k','r']
    ax1=fig.add_subplot(131)
    ax2=fig.add_subplot(132)
    ax3=fig.add_subplot(133)
    for k in range(len(run_number_sum)):
        run_number=run_number_sum[k]
        run_idx=int(run_idx_sum[k])
        ti=int(tis[run_idx])
        t0=int(t0s[run_idx])

        # EKE_VA,bc_VA=obtain_domain_ave_EKE_n_bc(loadpath,run_number)
        EKE_VA,bc_VA,EKE_dt=obtain_domain_ave_EKE_n_bc_n_EKEdt(loadpath,run_number)
        # ax1.plot(run_date,EKE_VA,label='run {:02d}'.format(run_number_name_sum[k]),color=color_group[k])
        ax1.plot(run_date,EKE_VA,label='run '+ '{}'.format(run_number_name_sum2[k]),color=color_group[k])
        ax1.scatter(ti,EKE_VA[ti],marker='o',color=color_group[k],s=20,zorder=2)
        ax1.scatter(t0,EKE_VA[t0],marker='x',color=color_group[k],s=20,zorder=2)

        # ax2.plot(run_date,bc_VA,label='run {:02d}'.format(run_number_name_sum[k]),color=color_group[k])
        ax2.plot(run_date,bc_VA,label='run '+ '{}'.format(run_number_name_sum2[k]),color=color_group[k])
        ax2.scatter(ti,bc_VA[ti],marker='o',color=color_group[k],s=20,zorder=2)
        ax2.scatter(t0,bc_VA[t0],marker='x',color=color_group[k],s=20,zorder=2)

        ax3.plot(run_date,EKE_dt,label='run '+ '{}'.format(run_number_name_sum2[k]),color=color_group[k])
        ax3.scatter(ti,EKE_dt[ti],marker='o',color=color_group[k],s=20,zorder=2)
        ax3.scatter(t0,EKE_dt[t0],marker='x',color=color_group[k],s=20,zorder=2)
    ax1.set_xlim(xlims)
    ax1.set_ylim(ylims_EKEs)
    ax1.legend(fontsize=8)
    ax1.tick_params(axis='both',which='major',labelsize=8)
    ax1.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    ax1.set_ylabel("Volume averaged EKE $[m^2/s^2]$",fontsize=8)
    ax1.set_xlabel('Time [day]',fontsize=8)

    ax2.set_xlim(xlims)
    ax2.set_ylim(ylims_wbs)
    ax2.legend(fontsize=8)
    ax2.tick_params(axis='both',which='major',labelsize=8)
    ax2.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    ax2.set_ylabel("Volume averaged $w'b' \;[m^2/s^3]$",fontsize=8)
    ax2.set_xlabel('Time [day]',fontsize=8)

    ax3.set_xlim(xlims)
    ax3.set_ylim(ylims_EKE_dt)
    ax3.legend(fontsize=8)
    ax3.tick_params(axis='both',which='major',labelsize=8)
    ax3.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    ax3.set_ylabel("dEKE/dt (Volume averaged) $\;[m^2/s^3]$",fontsize=8)
    ax3.set_xlabel('Time [day]',fontsize=8)

    # add subplot label
    ax1.text(0.02, 0.95, '{}'.format(chr(97)), transform=ax1.transAxes,zorder=20,fontsize=9)
    ax2.text(0.02, 0.95, '{}'.format(chr(98)), transform=ax2.transAxes,zorder=20,fontsize=9)
    ax3.text(0.02, 0.95, '{}'.format(chr(99)), transform=ax3.transAxes,zorder=20,fontsize=9)

    ax1.grid(alpha=0.5)
    ax2.grid(alpha=0.5)
    ax3.grid(alpha=0.5)
    plt.tight_layout()

    return fig

def plt_case_comparsion_of_urho(fig,loadpath,scenario,tis,t0s,xlims=(0,80),ylims=(0,2e-4)):
    run_number_sum,run_number_name_sum,run_idx_sum=obtain_scenario(str(scenario))
    gs = fig.add_gridspec(nrows=3, ncols=1,hspace=0.3)
    for i in range (3):
        ax=fig.add_subplot(gs[i,0])
        run_number=run_number_sum[i]
        run_name=run_number_name_sum[i]
        run_number_idx=run_idx_sum[i]
        t0=int(t0s[run_number_idx])
        ti=int(tis[run_number_idx])
        ds_eng=xr.open_dataset(loadpath+'run{:02d}_EnergyBudget_VA_Domain_n_PEZ.nc'.format(run_number))

        eddy_trans_pu=ds_eng['eddy_trans_domain_vol_ave']
        mean_trans_pu=ds_eng['mean_trans_domain_vol_ave']  

        p1,p2=plot_trans(ax,np.abs(eddy_trans_pu),np.abs(mean_trans_pu),ti,t0,xlims,ylims,)
        ax.set_title('Run {}'.format(run_name),fontsize=8)
        # add label to each subplot
        ax.text(0.02, 0.85, '{}'.format(chr(97+i)), transform=ax.transAxes,zorder=10,fontsize=9)

    plt.tight_layout()

    return p1,p2

############################################
# quantify the fastest eddy growth
def obtain_peak_EKE_loc(loadpath):
    ori_ds=xr.open_dataset(loadpath+'fitted_loc_w_wavelength.nc')
    fitted_loc=ori_ds['fitted_loc']
    L_obs_old=ori_ds['L_obs_w_new_loc']
    return fitted_loc

def obtain_L_obs_w_peak_loc(loadpath):
    ori_ds=xr.open_dataset(loadpath+'fitted_loc_w_wavelength.nc')
    fitted_loc=ori_ds['fitted_loc']
    L_obs_old=ori_ds['L_obs_w_new_loc']
    return fitted_loc,L_obs_old

def obtain_Ueddy_Bt_fit(loadpath,run_date=np.arange(120)):
    run_numbers,run_number_names,t0s,tis,Bs,Slope_burgers,alphas,N0s,fs=obtain_run_case_summary(loadpath)

    u_eddy_sum=[]
    Bt_sqrt_sum=[]

    for k in range(12):
        run_number=int(run_numbers[k])
        start=int(tis[k])
        end=int(t0s[k])
        ds_to_open2=xr.open_dataset(loadpath+'run{:02d}_eddy_flux_components.nc'.format(run_number))
        u_eddy=ds_to_open2['u_eddy_sel']
        B=np.abs(Bs[k].values)
        Bt_sqrt=np.sqrt(B*run_date*86400)

        u_eddy_sum.append(u_eddy[start:end])
        Bt_sqrt_sum.append(Bt_sqrt[start:end])

    u_eddy_develop=np.concatenate(u_eddy_sum)
    Bt_sqrt_develop=np.concatenate(Bt_sqrt_sum)

    fit_Ueddy_Bt_sqrt=np.polyfit(Bt_sqrt_develop,u_eddy_develop,1)
    return fit_Ueddy_Bt_sqrt

def plot_Ueddy_Bt(fig,loadpath,
                  xlims=(0,0.5),ylims=(0,0.1),
                  run_date=np.arange(120)):
    
    color_group=color_group=['k','k','grey','grey','orange','orange','r','r','b','b','g','g']
    run_numbers,run_number_names,t0s,tis,Bs,Slope_burgers,alphas,N0s,fs=obtain_run_case_summary(loadpath)
    fit_Ueddy_Bt_sqrt=obtain_Ueddy_Bt_fit(loadpath)

    ax1=fig.add_subplot(111)
    for k in range(1,12,2):
        run_number=int(run_numbers[k])
        run_number_name=run_number_names[k]
        start=int(tis[k])
        end=int(t0s[k])

        ds_to_open2=xr.open_dataset(loadpath+'run{:02d}_eddy_flux_components.nc'.format(run_number))
        u_eddy=ds_to_open2['u_eddy_sel']
        B=np.abs(Bs[k].values)
        Bt_power=np.sqrt(B*run_date*86400)
        ax1.scatter(Bt_power[start:end],u_eddy[start:end],
                    color=color_group[k],marker='x',s=10,label='run {:02d}'.format(run_number_name))
        
    for k in range(0,12,2):
        run_number=int(run_numbers[k])
        run_number_name=run_number_names[k]
        start=int(tis[k])
        end=int(t0s[k])

        ds_to_open2=xr.open_dataset(loadpath+'run{:02d}_eddy_flux_components.nc'.format(run_number))
        u_eddy=ds_to_open2['u_eddy_sel']
        B=np.abs(Bs[k].values)
        Bt_power=np.sqrt(B*run_date*86400)
        ax1.scatter(Bt_power[start:end],u_eddy[start:end],
                    color=color_group[k],marker=',',s=10,label='run{:02d}  S={:.3f}'.format(run_number_name, Slope_burgers[k]))
        # ax1.scatter(Bt_power[start:end],u_eddy[start:end],
        #             color=color_group[k],marker=',',s=10,label='run {:02d}'.format(run_number_name))


    ax1.set_ylabel('$U_{eddy}$  $[ms^{-1}]$',fontsize=8)
    ax1.set_xlabel('$(Bt)^{1/2} \; [ms^{-1}]$',fontsize=8)
    ax1.set_xlim(xlims)
    ax1.set_ylim(ylims)

    ax1.tick_params(which='minor', labelleft=False)
    ax1.legend(ncol=2,fontsize=7,loc='upper left')
    ax1.grid()

    # add the u_eddy vs Bt^{1/2} fit
    x=np.linspace(0,0.5,100)
    y=fit_Ueddy_Bt_sqrt[0]*x+fit_Ueddy_Bt_sqrt[1]
    ax1.plot(x,y,'k--',alpha=0.75)
    # write the equation
    ax1.text(0.6,0.8,'$ U_{eddy}$'+'$= {:.2f}(Bt)^{{1/2}}$'.format(fit_Ueddy_Bt_sqrt[0]),
            fontsize=8,transform=ax1.transAxes)
    # set equal ax
    # ax1.set_aspect('equal')

    return fig

def obtain_Lr_theory_n_obs(loadpath):
    ds_length=xr.open_dataset(loadpath+'Summary_Lr_obs_from_ave_up_qc_v4.nc')

    Lr_obs=ds_length['Lr_obs_qc'][0:12] #lambda 
    Lr_theory=ds_length['Lr_theory'][0:12] # Rossby radius

    return Lr_theory,Lr_obs,

def obtain_lambda_Rossby_radius_fit(loadpath):
    run_numbers,run_number_names,t0s,tis,Bs,Slope_burgers,alphas,N0s,fs=obtain_run_case_summary(loadpath)
    Lr_theory,Lr_obs=obtain_Lr_theory_n_obs(loadpath)

    Lr_theory_dev_sum=[]
    Lr_obs_dev_sum=[]
    for k in range(12):
        start_time=int(tis[k])
        end_time=int(t0s[k])
        Lr_theory_dev_sum.append(Lr_theory[k,start_time:end_time].values-Lr_theory[k,start_time].values)
        Lr_obs_dev_sum.append(Lr_obs[k,start_time:end_time].values-Lr_obs[k,start_time].values)


    Lr_obs_develop=np.concatenate(Lr_obs_dev_sum)
    Lr_theory_develop=np.concatenate(Lr_theory_dev_sum)

    mask = ~np.isnan(Lr_obs_develop)
    fit_lambda_Rossby_radius = np.polyfit(Lr_theory_develop[mask], Lr_obs_develop[mask], 1)
    return fit_lambda_Rossby_radius

def plot_lambda_vs_Rossby_radius(fig,loadpath,xlims=(0,8),ylims=(-5,30),):
    color_group=color_group=['k','k','grey','grey','orange','orange','r','r','b','b','g','g']
    run_numbers,run_number_names,t0s,tis,Bs,Slope_burgers,alphas,N0s,fs=obtain_run_case_summary(loadpath)
    fit_lambda_Rossby_radius=obtain_lambda_Rossby_radius_fit(loadpath)
    Lr_theory,Lr_obs=obtain_Lr_theory_n_obs(loadpath)

    ax=fig.add_subplot(111)

    for k in [1,3,5,7,9,11]:
        start_time=int(tis[k])
        end_time=int(t0s[k])
        ax.scatter(Lr_theory[k,start_time:end_time]-Lr_theory[k,start_time],
                Lr_obs[k,start_time:end_time]-Lr_obs[k,start_time],
                    color=color_group[k],label='run {:02d}'.format(run_number_names[k]),
                    marker='x',s=15,zorder=2)

    for k in [0,2,4,6,8,10,]:
        start_time=int(tis[k])
        end_time=int(t0s[k])
        ax.scatter(Lr_theory[k,start_time:end_time]-Lr_theory[k,start_time],
                Lr_obs[k,start_time:end_time]-Lr_obs[k,start_time],
                    color=color_group[k],label='run {:02d}, S={:.3f}'.format(run_number_names[k], Slope_burgers[k]),
                    marker=',',s=10)
        # ax.scatter(Lr_theory[k,start_time:end_time]-Lr_theory[k,start_time],
        #         Lr_obs[k,start_time:end_time]-Lr_obs[k,start_time],
        #             color=color_group[k],label='run {:02d}'.format(run_number_names[k],),
        #             marker=',',s=10)

    ax.legend(ncol=2,fontsize=7,loc='lower right')
    ax.grid()
    ax.set_xlabel(r'$ \Delta \;L_r= L_r-L_{r,t_f}$ [km]',fontsize=8)
    ax.set_ylabel(r'$\Delta \;\lambda =\lambda_{obs}-\lambda_{t_f}$ [km]',fontsize=8)
    ax.set_xlim(xlims)
    ax.set_ylim(ylims)
    ax.tick_params(axis='both',which='major',labelsize=8)
    
    ax.plot(np.arange(0,10),fit_lambda_Rossby_radius[0]*np.arange(0,10)+int(fit_lambda_Rossby_radius[1]),color='grey',linewidth=1)
    ax.text(0.8, 0.85, '$\Delta \;\lambda \sim {:.1f} \;\Delta\; L_r$'.format(fit_lambda_Rossby_radius[0]), 
            transform=ax.transAxes, fontsize=8)
    
    return fig

def obtain_dLr_dt(loadpath,coef=3.6):
    run_numbers,run_number_names,t0s,tis,Bs,Slope_burgers,alphas,N0s,fs=obtain_run_case_summary(loadpath)
    Lr_theory,Lr_obs=obtain_Lr_theory_n_obs(loadpath)
    dlambda_dt_obs=np.zeros(12)
    dlambda_dt_theory=np.zeros(12)
    for k in range (12):
        ti=int(tis[k])
        t0=int(t0s[k])
        dlambda_dt_obs[k]=((Lr_obs[k,t0]-Lr_obs[k,ti])/(t0-ti)/86400)
        B_value=np.abs(Bs[k].values)
        f_value=np.abs(fs[k])
        dlambda_dt_theory[k]=coef*(B_value)**0.5/(2*(t0-ti)*86400)**0.5/f_value
    
    return dlambda_dt_obs,dlambda_dt_theory

def plot_dLr_dt_compare(fig,loadpath,xlims=(0,0.01),ylims=(0,0.01)):
    dlambda_dt_obs,dlambda_dt_theory=obtain_dLr_dt(loadpath)
    ax=fig.add_subplot(111)
    ax.scatter(dlambda_dt_theory,dlambda_dt_obs*1e3,color='k',)
    # equal aspect ratio
    ax.set_aspect('equal', adjustable='box')
    ax.grid(alpha=0.5)
    # set xy notation as sci
    ax.ticklabel_format(style='sci', axis='both', scilimits=(0,0))
    ax.set_xlabel(r'${3.6B}^{1/2}/f(2t)^{1/2} \; [ms^{-1}]$', fontsize=8)
    ax.set_ylabel(r'$d\Delta\lambda/dt \;[ms^{-1}]$', fontsize=8)

    ax.set_xlim(xlims)
    ax.set_ylim(ylims)
    # set ticks label fontsize
    ax.tick_params(axis='both',which='major',labelsize=8)
    ax.xaxis.get_offset_text().set_fontsize(8)
    ax.yaxis.get_offset_text().set_fontsize(8)

    return fig

def obtain_EKE_Bt_fit(loadpath,run_date=np.arange(120)):
    run_numbers,run_number_names,t0s,tis,Bs,Slope_burgers,alphas,N0s,fs=obtain_run_case_summary(loadpath)

    EKE_sum=[]
    Bt_sum=[]

    for k in range(12):
        run_number=int(run_numbers[k])
        start=int(tis[k])
        end=int(t0s[k])
        ds_to_open2=xr.open_dataset(loadpath+'run{:02d}_eddy_flux_components.nc'.format(run_number))
        u_eddy=ds_to_open2['u_eddy_sel']
        EKE=u_eddy**2/2
        B=np.abs(Bs[k].values)
        Bt=B*run_date*86400

        EKE_sum.append(EKE[start:end])
        Bt_sum.append(Bt[start:end])

    EKE_develop=np.concatenate(EKE_sum)
    Bt_develop=np.concatenate(Bt_sum)

    fit_EKE_Bt=np.polyfit(Bt_develop,EKE_develop,1)
    return fit_EKE_Bt

def plot_EKE_Bt(fig,loadpath,xlims=(1e-2,2e-1),ylims=(1e-5,1e-2),run_date=np.arange(120)):
    color_group=color_group=['k','k','grey','grey','orange','orange','r','r','b','b','g','g']
    run_numbers,run_number_names,t0s,tis,Bs,Slope_burgers,alphas,N0s,fs=obtain_run_case_summary(loadpath)
    fit_EKE_Bt=obtain_EKE_Bt_fit(loadpath)
    ax1=fig.add_subplot(111)
    for k in range(1,12,2):
        run_number=int(run_numbers[k])
        run_number_name=run_number_names[k]
        start=int(tis[k])
        end=int(t0s[k])
        ds_to_open2=xr.open_dataset(loadpath+'run{:02d}_eddy_flux_components.nc'.format(run_number))
        u_eddy=ds_to_open2['u_eddy_sel']
        EKE=u_eddy**2/2
        B=np.abs(Bs[k].values)
        Bt=B*run_date*86400
        ax1.scatter(Bt[start:end],EKE[start:end],
                    color=color_group[k],marker='x',s=10,label='run {:02d}'.format(run_number_name))

    for k in range(0,12,2):
        run_number=int(run_numbers[k])
        run_number_name=run_number_names[k]
        start=int(tis[k])
        end=int(t0s[k])
        ds_to_open2=xr.open_dataset(loadpath+'run{:02d}_eddy_flux_components.nc'.format(run_number))
        u_eddy=ds_to_open2['u_eddy_sel']
        EKE=u_eddy**2/2
        B=np.abs(Bs[k].values)
        Bt=B*run_date*86400
        ax1.scatter(Bt[start:end],EKE[start:end],
                    color=color_group[k],marker=',',s=10,label='run {:02d}, S={:.3f}'.format(run_number_name, Slope_burgers[k]))

    ax1.set_ylabel('$EKE$  $[m^2s^{-2}]$',fontsize=8)
    ax1.set_xlabel('Bt $[m^2s^{-2}]$',fontsize=8)
    ax1.set_xlim(xlims)
    ax1.set_ylim(ylims)

    ax1.tick_params(which='minor', labelleft=False,labelbottom=False)
    ax1.legend(ncol=2,fontsize=7,loc='upper left')
    ax1.set_yscale('log')
    ax1.set_xscale('log')
    ax1.grid()

    x=np.linspace(0,0.5,100)
    y=fit_EKE_Bt[0]*x+fit_EKE_Bt[1]
    # ax1.plot(x,y,'k--',alpha=0.75)
    # write the equation
    ax1.text(0.6,0.75,'$ EKE = {:.2f}(Bt)$'.format(fit_EKE_Bt[0]),
            fontsize=8,transform=ax1.transAxes)
    # set equal ax
    # ax1.set_aspect('equal')

    return fig

def obtain_Ueddy_Vbar_fit(loadpath,run_date=np.arange(120)):
    run_numbers,run_number_names,t0s,tis,Bs,Slope_burgers,alphas,N0s,fs=obtain_run_case_summary(loadpath)

    Vbar_sum=[]
    Ueddy_sum=[]
    for k in range(0,12):

        start_time=int(tis[k])
        end_time=int(t0s[k])
        run_number=run_numbers[k]

        ds_V=xr.open_dataset(loadpath+'run{:02d}_v_mean_da_at_EKE_max.nc'.format(run_number))
        v_mean_at_EKE_max=ds_V['v_mean_da_at_EKE_max']
        ds_eddy=xr.open_dataset(loadpath+'run{:02d}_u_eddy_at_EKE_max.nc'.format(run_number))
        u_eddy_mean=ds_eddy['u_eddy_mean_at_EKE_max']
        Vbar_sum.append(v_mean_at_EKE_max[start_time:end_time])
        Ueddy_sum.append(u_eddy_mean[start_time:end_time])
        
    Vbar_develop=np.concatenate(Vbar_sum)
    Ueddy_develop=np.concatenate(Ueddy_sum)

    mask = ~np.isnan(Ueddy_develop)
    fit_Ueddy_Vbar=np.polyfit(Vbar_develop[mask],Ueddy_develop[mask],1)
    return fit_Ueddy_Vbar

def plot_Ueddy_Vbar(fig,loadpath,xlims=(0,0.05),ylims=(0,0.05)):
    color_group=color_group=['k','k','grey','grey','orange','orange','r','r','b','b','g','g']
    run_numbers,run_number_names,t0s,tis,Bs,Slope_burgers,alphas,N0s,fs=obtain_run_case_summary(loadpath)
    fit_Ueddy_Vbar=obtain_Ueddy_Vbar_fit(loadpath)
    ax=fig.add_subplot(111)

    for k in range(1,12,2):
        
        run_number=int(run_numbers[k])
        run_number_name=run_number_names[k]
        start_time=int(tis[k])
        end_time=int(t0s[k])

        ds_V=xr.open_dataset(loadpath+'run{:02d}_v_mean_da_at_EKE_max.nc'.format(run_number))
        v_mean_at_EKE_max=ds_V['v_mean_da_at_EKE_max']

        ds_eddy=xr.open_dataset(loadpath+'run{:02d}_u_eddy_at_EKE_max.nc'.format(run_number))
        u_eddy_mean=ds_eddy['u_eddy_mean_at_EKE_max']

        ax.scatter(v_mean_at_EKE_max[start_time:end_time],
            u_eddy_mean[start_time:end_time],color=color_group[k],marker='x',s=10,
            label='run {:02d}'.format(run_number_name))


    for k in range(0,12,2):
        
        run_number=int(run_numbers[k])
        run_number_name=run_number_names[k]
        start_time=int(tis[k])
        end_time=int(t0s[k])

        ds_V=xr.open_dataset(loadpath+'run{:02d}_v_mean_da_at_EKE_max.nc'.format(run_number))
        v_mean_at_EKE_max=ds_V['v_mean_da_at_EKE_max']

        ds_eddy=xr.open_dataset(loadpath+'run{:02d}_u_eddy_at_EKE_max.nc'.format(run_number))
        u_eddy_mean=ds_eddy['u_eddy_mean_at_EKE_max']

  
        ax.scatter(v_mean_at_EKE_max[start_time:end_time],
            u_eddy_mean[start_time:end_time],color=color_group[k],marker=',',s=10,
            label='run {:02d}, S={:.3f}'.format(run_number_name, Slope_burgers[k]))
            # label='run {:02d}'.format(run_number_name),marker=',',s=10)


    ax.plot(np.arange(2),fit_Ueddy_Vbar[0]*np.arange(2)+fit_Ueddy_Vbar[1],color='grey',linewidth=1)
    ax.text(0.82, 0.65, r'$U_{eddy}$'+'$\sim {:.1f}$'.format(fit_Ueddy_Vbar[0])+'$\overline{v}$', 
        transform=ax.transAxes, fontsize=8)
    ax.legend(ncol=2,fontsize=7,loc='lower right')
    ax.grid()
    ax.set_xlabel(r'$ \overline{v}$  [m/s]',fontsize=8)
    ax.set_ylabel(r'$|U_{eddy}|$ [m/s]',fontsize=8)
    ax.set_xlim(xlims)
    ax.set_ylim(ylims)
    ax.tick_params(axis='both',which='major',labelsize=8)

    return fig

def obtain_theory_n_obs_MLD(loadpath,run_date=np.arange(120)):
    run_numbers,run_number_names,t0s,tis,Bs,Slope_burgers,alphas,N0s,fs=obtain_run_case_summary(loadpath)
    ds_MLD=xr.open_dataset(loadpath+'sum_MLD_frontal.nc')
    MLD_obs_sum=ds_MLD['MLD'][0:12,...]

    MLD_theory_sum=np.zeros([12,120])
    run_date_sec=run_date*24*3600

    for k in range(12):
        B=np.abs(Bs[k].values)
        N0=N0s[k].values
        MLD_theory_sum[k,...]=np.sqrt(2*B*run_date_sec)/N0

    return MLD_theory_sum,MLD_obs_sum

def plot_MLD_obs_n_theory(loadpath,fig,):
    MLD_theory_sum,MLD_obs_sum=obtain_theory_n_obs_MLD(loadpath)
    run_numbers,run_number_names,t0s,tis,Bs,Slope_burgers,alphas,N0s,fs=obtain_run_case_summary(loadpath)
    color_group=color_group=['k','k','grey','grey','orange','orange','r','r','b','b','g','g']
    
    ax=fig.add_subplot(111)

    for k in [1,3,5,7,9,11]:
        run_number_name=run_number_names[k]
        ax.scatter(MLD_theory_sum[k,0:60],MLD_obs_sum[k,0:60],
                    color=color_group[k],label='run {:02d}'.format(run_number_name),
                    marker='x',s=15,zorder=2)
        
    for k in [0,2,4,6,8,10,]:
        run_number_name=run_number_names[k]
        ax.scatter(MLD_theory_sum[k,0:60],MLD_obs_sum[k,0:60],
                    color=color_group[k],label='run {:02d}'.format(run_number_name),
                    marker=',',s=15,zorder=2)
    ax.set_xlabel(r"$h= \frac{(2Bt)^{1/2}}{N_0}$", fontsize=10)
    ax.set_ylabel('$MLD_{obs} $', fontsize=10)
    ax.tick_params(axis='both',which='major',labelsize=8)
    ax.set_xlim(0,200)
    ax.set_ylim(0,200)
    ax.legend(ncol=2,fontsize=7,loc='lower right')
    plt.grid(alpha=0.5)
    plt.tight_layout()

    return fig
#########################
#TBD

def plot_side_rho_isotherm(ax,cross,depth,rho_side,temp_side,xlims,ylims,
                           temp_level=np.arange(19,24,0.5),vmax=999.8,vmin=1000.8):
    cmap0 = plt.get_cmap('inferno_r')
    cutted_inferno_r = truncate_colormap(cmap0, 0, 0.8)
    cs=ax.pcolormesh(cross,depth,rho_side,cmap=cutted_inferno_r,vmin=vmin,vmax=vmax)
    ct=ax.contour(cross,depth,temp_side,levels=temp_level,colors='k',linewidths=0.5,)
    ax.set_xlim(xlims)
    ax.set_ylim(ylims)
    ax.tick_params(axis='both',which='major',labelsize=8)
    ax.patch.set_facecolor('grey')
    return cs, ct

def plot_MKE_EKEs(ax,MKE,EKE,time,ti,t0,xlims,ylims=(1e-8,1e-2)):
    p1=ax.plot(time,MKE,label=r"$MKE$",color='b',)#linestyle='dashed')
    p2=ax.plot(time,EKE,label=r"$EKE$",color='r')
    ax.scatter(ti,EKE[ti],marker='o',color='k',s=15,zorder=2)
    ax.scatter(t0,EKE[t0],marker='x',color='k',s=15,zorder=2)
    ax.set_xlim(xlims)
    ax.set_ylim(ylims)
    ax.set_yscale('log')
    ax.set_ylabel('Kinetic Energy \n $[m^{2}/s^{2}]$',fontsize=8)
    ax.tick_params(axis='both',which='major',labelsize=8)
    ax.text(0.45,0.45,'MKE',fontsize=6,color='b',transform=ax.transAxes,)
    ax.text(0.3,0.85,'EKE',fontsize=6,color='r',transform=ax.transAxes,)

    return p1,p2
