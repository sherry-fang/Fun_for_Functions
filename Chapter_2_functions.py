import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib import colors

def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap

def obtain_paths():
    savepath='../Chp2/v5/figs/'
    loadpath='../../Year 3/Setonix_results/nc_file/'
    return savepath, loadpath

def load_2d_ds(file_path=r'Z:/Base_2d/gen_ds'):
    file_names=os.listdir(file_path)
    file_ob_list=[]
    for file_name in file_names:
        if 'base_2d_processed_ds_0' in file_name:
            fileob=file_path+'/'+file_name
            file_ob_list.append(fileob)

    ds=xr.open_mfdataset(file_ob_list,concat_dim='time',combine='nested')
    myds=ds.sortby('time')
    return myds

def load_mean_vel_n_rho_from_2d_ds():
    myds=load_2d_ds()
    temp=myds['temp_instant'][:,:,1,:]
    rho=myds['rho_instant'][:,:,1,:]
    u=myds['u_instant'][:,:,1,:]
    v=myds['v_instant'][:,:,1,:]
    
    layer_depth=myds['layer_depth']
    xl=myds['cross_shelf']
    yl=myds['along_shelf']
    return temp,rho,u,v,layer_depth,xl,yl



def load_2d_vars():
    myds=load_2d_ds()
    temp=myds['temp_instant'][:,:,1,:]
    rho=myds['rho_instant'][:,:,1,:]
    layer_depth=myds['layer_depth']
    xl=myds['cross_shelf']
    yl=myds['along_shelf']
    return temp,rho,layer_depth,xl,yl

def plot_vel_2d(fig,rho_levels=np.arange(998,1005,0.05),
                 days_sel=[0,24,35,],
                 run_date=np.arange(120),
                 xlims=(150,300),
                 ylims=(-220,0),
                 cross=np.arange(300),
                 vmax_V=0.05,vmin_V=-0.05,
                 vmax_U=0.02,vmin_U=-0.02):
    temp,rho,u,v,layer_depth,xl,yl=load_mean_vel_n_rho_from_2d_ds()

    gs=fig.add_gridspec(nrows=2,ncols=4,width_ratios=[1,1,1,0.05],hspace=0.2,wspace=0.35)

    ax_1=fig.add_subplot(gs[0, 0])
    f1_1,f2_1=ax_frameless(ax_1)
    ax_1.set_ylabel('z [m]',fontsize=8)
    ax_1.yaxis.set_label_coords(-.3, .5)

    ax_2=fig.add_subplot(gs[1, 0])
    f1_2,f2_2=ax_frameless(ax_2)
    ax_2.set_ylabel('z [m]',fontsize=8)
    ax_2.yaxis.set_label_coords(-.3, .5)

    for i in range(3):

        ax1=fig.add_subplot(gs[0,i])
        ax2=fig.add_subplot(gs[1,i])
 

        p1,c1,=plot_meanflow_w_rho(ax1,cross,-layer_depth,v[days_sel[i],...],rho[days_sel[i],...],
                xlims,ylims,rho_levels=np.arange(999.0,1002.7,0.05),vmax=vmax_V,vmin=vmin_V)

        ax1.set_title('Day '+str(days_sel[i]),fontsize=8)
        ax1.set_xlim(xlims)
        ax1.set_ylim(ylims)
        #add label to each subplot
        ax1.text(0.02, 0.9, '{}'.format(chr(97+i)), transform=ax1.transAxes,zorder=10,fontsize=8)   

        p2,c2,=plot_meanflow_w_rho(ax2,cross,-layer_depth,u[days_sel[i],...],rho[days_sel[i],...],
                xlims,ylims,rho_levels=np.arange(999.0,1002.7,0.05),vmax=vmax_U,vmin=vmin_U)
        ax2.set_xlabel('Cross-shelf distance [km]',fontsize=8)
        ax2.text(0.02, 0.9, '{}'.format(chr(100+i)), transform=ax2.transAxes,zorder=10,fontsize=8)

    cax1=fig.add_subplot(gs[0,3])
    cbar1=plt.colorbar(p1,cax=cax1)
    cbar1.ax.set_title('$\overline{V} \;[ms^{-1}]$',fontsize=8,rotation=0)
    cax2=fig.add_subplot(gs[1,3])
    cbar2=plt.colorbar(p2,cax=cax2)
    cbar2.ax.set_title('$\overline{U} \;[ms^{-1}]$',fontsize=8,rotation=0)

    

    plt.tight_layout()

    return fig

def load_2d_flux_MKE(loadpath):
    ds=xr.open_dataset(loadpath+'2D_case_mean_temp_flux_MKE.nc')
    mean_flux_dep_ave=ds['uT_mean_rm_initial_dep_ave']
    MKE_dep_ave=ds['MKE_dep_ave']

    return mean_flux_dep_ave,MKE_dep_ave

def plot_vel_2d_snapshots(loadpath,fig,rho_levels=np.arange(998,1005,0.05),
                 days_sel=[0,24,35,],
                 run_date=np.arange(120),
                 xlims=(150,300),
                 ylims=(-220,0),
                 ylims_KE=(0,1e-3),
                 ylims_flux=(0,0.1),
                 cross=np.arange(300),
                 vmax_V=0.05,vmin_V=-0.05,
                 vmax_U=0.02,vmin_U=-0.02):
    temp,rho,u,v,layer_depth,xl,yl=load_mean_vel_n_rho_from_2d_ds()
    mean_flux_dep_ave,MKE_dep_ave=load_2d_flux_MKE(loadpath)
    subplot_index=['a','b','c','d','e','f','g','h','i','j','k','l',]
    gs = fig.add_gridspec(nrows=4, ncols=4, height_ratios=[0.5,0.5,0.75,0.75],
                width_ratios=[1,1,1,0.05],hspace=0.35,wspace=0.35)
    
    ax_1=fig.add_subplot(gs[0, 0])
    ax_1.set_ylabel('MKE $[m^{2}s^{-2}]$',fontsize=8)
    f1,f2=ax_frameless(ax_1)
    ax_1.yaxis.set_label_coords(-.25, .5)

    ax_2=fig.add_subplot(gs[1, 0])
    f1,f2=ax_frameless(ax_2)
    ax_2.set_ylabel('Temp flux [$^\circ Cms^{-1}$]',fontsize=8)
    ax_2.yaxis.set_label_coords(-.25, .5)
    
    ax_3=fig.add_subplot(gs[2, 0])
    f1,f2=ax_frameless(ax_3)
    ax_3.set_ylabel('z [m]',fontsize=8)
    ax_3.yaxis.set_label_coords(-.3, .5)

    ax_4=fig.add_subplot(gs[3, 0])
    f1,f2=ax_frameless(ax_4)
    ax_4.set_ylabel('z [m]',fontsize=8)
    ax_4.yaxis.set_label_coords(-.3, .5)


    for i in range(3):

        date=int(days_sel[i])
        ax1 = fig.add_subplot(gs[0, i])
        subplot_int1=int(3*0+i)

        MKE=ax1.plot(cross,MKE_dep_ave[date,...],color='navy',linewidth=0.75,)
        ax1.set_xlim(xlims)
        ax1.set_ylim(ylims_KE)
        ax1.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
        ax1.set_title('Day {:01d}'.format(date),fontsize=8)
        ax1.text(0.02, 0.88, subplot_index[subplot_int1], 
                transform=ax1.transAxes,zorder=10,fontsize=8)
        
        ax2=fig.add_subplot(gs[1,i])
        subplot_int2=int(3*1+i)
        mean_flux=ax2.plot(cross,np.abs(mean_flux_dep_ave[date,...]),color='navy',linewidth=0.75,)#clip_on=False)
        ax2.text(0.02, 0.88, subplot_index[subplot_int2], 
                transform=ax2.transAxes,zorder=10,fontsize=8)
        
        ax2.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
        ax2.set_xlim(xlims)
        ax2.set_ylim(ylims_flux)       

        ax3=fig.add_subplot(gs[2,i])
        subplot_int3=int(3*2+i)
        p1,c1,=plot_meanflow_w_rho(ax3,cross,-layer_depth,v[date,...],rho[date,...],
                xlims,ylims,rho_levels=np.arange(999.0,1002.7,0.05),vmax=vmax_V,vmin=vmin_V)

        ax3.set_title('Day '+str(date),fontsize=8)
        ax3.set_xlim(xlims)
        ax3.set_ylim(ylims)
        #add label to each subplot
        ax3.text(0.02, 0.9, subplot_index[subplot_int3], 
                transform=ax3.transAxes,zorder=10,fontsize=8)
        

        ax4=fig.add_subplot(gs[3,i])
        subplot_int4=int(3*3+i)
        p2,c2,=plot_meanflow_w_rho(ax4,cross,-layer_depth,u[date,...],rho[date,...],
                xlims,ylims,rho_levels=np.arange(999.0,1002.7,0.05),vmax=vmax_U,vmin=vmin_U)
        ax4.set_xlabel('Cross-shelf distance [km]',fontsize=8)
        ax4.text(0.02, 0.9, subplot_index[subplot_int4], 
                transform=ax4.transAxes,zorder=10,fontsize=8)
        
    cax1=fig.add_subplot(gs[2,3])
    cbar1=plt.colorbar(p1,cax=cax1)
    cbar1.ax.set_title('$\overline{V} \;[ms^{-1}]$',fontsize=8,rotation=0)
    cax2=fig.add_subplot(gs[3,3])
    cbar2=plt.colorbar(p2,cax=cax2)
    cbar2.ax.set_title('$\overline{U} \;[ms^{-1}]$',fontsize=8,rotation=0)

    plt.tight_layout()

    return fig


def plot_temp_2d(fig,rho_levels=np.arange(998,1005,0.05),
                 days_sel=[0,15,30,45,60,75,],
                 run_date=np.arange(120),
                 xlims=(0,300),
                 ylims=(-220,0)):
    temp,rho,layer_depth,xl,yl=load_2d_vars()
    cmap = plt.get_cmap('inferno')
    cutted_inferno = truncate_colormap(cmap, 0.2,1.0)

    gs=fig.add_gridspec(nrows=2,ncols=4,width_ratios=[1,1,1,0.05],hspace=0,wspace=0.3)

    for i in range(len(days_sel)):
        row=i//3
        col=i%3
        ax=fig.add_subplot(gs[row,col])
        temp_sel=temp[days_sel[i],:,:]
        rho_sel=rho[days_sel[i],:,:]
        
        cax=ax.pcolormesh(xl/1000,-layer_depth,temp_sel,cmap=cutted_inferno,vmax=23.5,vmin=19)
        ax.contour(xl/1000,-layer_depth,rho_sel,levels=rho_levels,colors='k',linewidths=0.5)
        ax.tick_params(axis='both',labelsize=8)
        ax.set_title('Day '+str(days_sel[i]),fontsize=8)
        ax.set_aspect('equal')
        ax.set_xlim(xlims)
        ax.set_ylim(ylims)
        #add label to each subplot
        ax.text(0.02, 0.9, '{}'.format(chr(97+i)), transform=ax.transAxes,zorder=10,fontsize=8)

    cax2=fig.add_subplot(gs[0:2,3])
    cbar2=plt.colorbar(cax,cax=cax2,fraction=0.5)
    cbar2.ax.set_title(r'Temp [$^\circ$ C]',fontsize=8,rotation=0,)
    cbar2.ax.tick_params(labelsize=8)

    fig.text(0.05, 0.5, 'z [m]', va='center', rotation='vertical',fontsize=8)
    fig.text(0.5, 0.1, 'Cross-shelf distance [km]', ha='center',fontsize=8)

    return fig

def load_MLD_for_2D(loadpath):
    ds_to_open=xr.open_dataset(loadpath+'MLD_obs_n_theory_2d.nc')
    MLD_obs=ds_to_open['MLD_obs_2d']
    MLD_theory=ds_to_open['MLD_theory_2d']

    return MLD_obs,MLD_theory

def plot_MLD_for_2D(fig,loadpath,run_date=np.arange(120),xlims=(0,120),ylims=(0,160)):
    MLD_obs,MLD_theory=load_MLD_for_2D(loadpath)
    ax=fig.add_subplot(111)
    ax.scatter(run_date,MLD_obs,label='Obs',color='k')
    ax.plot(run_date,MLD_theory,label='Theory',color='k',alpha=0.75)
    ax.set_xlabel('Time [day]',fontsize=8)
    ax.set_ylabel('Mixed layer depth [m]',fontsize=8)
    ax.grid(alpha=0.5)
    ax.legend(fontsize=8)
    ax.set_xlim(xlims)
    ax.set_ylim(ylims)
    ax.tick_params(axis='both',which='major',labelsize=8)
    return fig

def load_MLD_for_3Dbase(loadpath,k=int(1)):
    ds_MLD=xr.open_dataset(loadpath+'sum_MLD_frontal.nc')
    MLD_obs_3D=ds_MLD['MLD'][k,...].values
    return MLD_obs_3D



def plot_MLD_obs_n_theory_for_both_Bases(loadpath,fig,run_date=np.arange(120),xlims=(0,120),ylims=(0,160)):
    MLD_obs_3D=load_MLD_for_3Dbase(loadpath)
    MLD_obs_2D,MLD_theory=load_MLD_for_2D(loadpath)

    ax=fig.add_subplot(111)
    ax.scatter(run_date,MLD_obs_2D,label='Obs_2D',color='k')
    ax.scatter(run_date,MLD_obs_3D,label='Obs_3D',color='r',marker='x')
    ax.plot(run_date,MLD_theory,label='Theory',color='k',alpha=0.75)
    ax.set_xlabel('Time [day]',fontsize=8)
    ax.set_ylabel('Mixed layer depth [m]',fontsize=8)
    ax.grid(alpha=0.5)
    ax.legend(fontsize=8)
    ax.set_xlim(xlims)
    ax.set_ylim(ylims)
    ax.tick_params(axis='both',which='major',labelsize=8)

    return fig

def load_2d_3d_compare_ds(loadpath):
    '''
    This data set shows the time series variables of the 25 km costal box for testing the temperature budget
    '''
    ds_base_case_compare = xr.open_dataset(loadpath+'temp_balance_test_for_base_cases.nc')
    dTdt_3d=ds_base_case_compare['temp_rate_3d_pu']
    dTdt_2d=ds_base_case_compare['temp_rate_2d_pu']
    TF_crossshore_3d=ds_base_case_compare['TF_total_3d_pu']
    TF_crossshore_2d=ds_base_case_compare['TF_total_2d_pu']
    TF_surf_3d=ds_base_case_compare['surf_temp_flux_3d_pu']
    TF_surf_2d=ds_base_case_compare['surf_temp_flux_2d_pu']

    return dTdt_3d, dTdt_2d, TF_surf_3d, TF_surf_2d, TF_crossshore_3d, TF_crossshore_2d

def plot_temp_budget_2d(fig,loadpath,
                               run_date=np.arange(120),
                               xlims=(0,120),
                               ylims=(-8e-7,2e-7)):
    dTdt_3d, dTdt_2d, TF_surf_3d, TF_surf_2d, TF_crossshore_3d, TF_crossshore_2d=load_2d_3d_compare_ds(loadpath)
    
    ax=fig.add_subplot(111)
    ax.plot(run_date,dTdt_2d,label='dT/dt',color='k')
    ax.plot(run_date,-TF_crossshore_2d,label='$TF_o$',color='navy')
    ax.axhline(y=TF_surf_2d,label='$TF_s$',color='orange')
    ax.plot(run_date,-TF_crossshore_2d+TF_surf_2d-dTdt_2d,label='Residual',
            color='grey')

    ax.set_xlabel('Time [day]',fontsize=8)
    ax.set_ylabel(r'Temperature change rate $[^\circ Cs^{-1}]$',fontsize=8)
    # ax.legend(bbox_to_anchor=(0.8, -0.25),ncol=4,fontsize=8)
    ax.legend(loc='upper left',ncol=4,fontsize=8)
    ax.grid(alpha=0.5)
    ax.set_xlim(xlims)
    ax.set_ylim(ylims)
    ax.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
    ax.yaxis.get_offset_text().set_fontsize(8)
    ax.tick_params(axis='both', which='major', labelsize=8)

    return fig

def plot_temp_budget_w_both_2d_n_3d(fig,loadpath,
                                    run_date=np.arange(120),
                                    xlims=(0,120),
                                    ylims=(-8e-7,6e-7)):

    dTdt_3d, dTdt_2d, TF_surf_3d, TF_surf_2d, TF_crossshore_3d, TF_crossshore_2d=load_2d_3d_compare_ds(loadpath)
    ax=fig.add_subplot(111)
    ax.plot(run_date,dTdt_3d,label='$dT/dt_{3d}$',color='k')
    ax.plot(run_date,-TF_crossshore_3d,label='$TF_{o, 3d}$',color='r')
    ax.axhline(y=TF_surf_3d,label='$TF_{s, 3d}$',color='b')
    ax.plot(run_date,dTdt_2d,label='$dT/dt_{2d}$',color='k',linestyle='dashed')
    ax.plot(run_date,-TF_crossshore_2d,label='$TF_{o, 2d}$',color='r',linestyle='dashed')
    ax.axhline(y=TF_surf_2d,label='$TF_{s, 2d}$',color='b',linestyle='dashed')

    ax.set_xlabel('Time [day]',fontsize=8)
    ax.set_ylabel(r'Temperature change rate $[^\circ Cs^{-1}]$',fontsize=8)
    ax.legend(ncol=2,loc='upper left',fontsize=8)
    ax.grid(alpha=0.5)
    ax.set_xlim(xlims)
    ax.set_ylim(ylims)
    ax.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
    ax.yaxis.get_offset_text().set_fontsize(8)
    ax.tick_params(axis='both', which='major', labelsize=8)

    return fig

def load_2d_3d_compare_ds_w_filename(loadpath,file_name):
    '''
    This data set shows the time series variables of the 25 km costal box for testing the temperature budget
    '''
    ds_base_case_compare = xr.open_dataset(loadpath+str(file_name))
    dTdt_3d=ds_base_case_compare['temp_rate_3d_pu']
    dTdt_2d=ds_base_case_compare['temp_rate_2d_pu']
    TF_crossshore_3d=ds_base_case_compare['TF_total_3d_pu']
    TF_crossshore_2d=ds_base_case_compare['TF_total_2d_pu']
    TF_surf_3d=ds_base_case_compare['surf_temp_flux_3d_pu']
    TF_surf_2d=ds_base_case_compare['surf_temp_flux_2d_pu']

    return dTdt_3d, dTdt_2d, TF_surf_3d, TF_surf_2d, TF_crossshore_3d, TF_crossshore_2d

def plot_temp_budget_compare_CV(fig,loadpath,file_names,cross_shelf_loc,
                                    run_date=np.arange(120),
                                    xlims=(0,120),
                                    ylims=(-8e-7,8e-7)):
    gs=fig.add_gridspec(nrows=3,ncols=1,hspace=0.3)
    for i in range(len(file_names)):
        file_name=file_names[i]
        cross_loc=cross_shelf_loc[i]
        dTdt_3d, dTdt_2d, TF_surf_3d, TF_surf_2d, TF_crossshore_3d, TF_crossshore_2d=load_2d_3d_compare_ds_w_filename(loadpath,file_name)
        ax=fig.add_subplot(gs[i,0])
        ax.plot(run_date,dTdt_3d,label='$dT/dt_{3d}$',color='k')
        ax.plot(run_date,-TF_crossshore_3d,label='$TF_{o, 3d}$',color='r')
        ax.axhline(y=TF_surf_3d,label='$TF_{s, 3d}$',color='b')
        ax.plot(run_date,dTdt_2d,label='$dT/dt_{2d}$',color='k',linestyle='dashed')
        ax.plot(run_date,-TF_crossshore_2d,label='$TF_{o, 2d}$',color='r',linestyle='dashed')
        ax.axhline(y=TF_surf_2d,label='$TF_{s, 2d}$',color='b',linestyle='dashed')
        ax.set_title('At {:02d} km from the coast'.format(cross_loc),fontsize=8)
        ax.grid(alpha=0.5)
        ax.set_xlim(xlims)
        ax.set_ylim(ylims)
        ax.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
        ax.yaxis.get_offset_text().set_fontsize(8)
        ax.tick_params(axis='both', which='major', labelsize=8)
        ax.text(0.02, 0.9, '{}'.format(chr(97+i)), transform=ax.transAxes,zorder=10,fontsize=8)

    plt.tight_layout()    
    ax.legend(bbox_to_anchor=(0.9, -0.35),ncol=6,fontsize=6)
    # plt.subplots_adjust(left=0.07, right=0.93, wspace=0.25, hspace=0.35)
    fig.text(0.05, 0.5, 'Temperature change rate $[^\circ Cs^{-1}]$', va='center', rotation='vertical',fontsize=8)
    fig.text(0.5, 0.05, 'Time [day]', ha='center',fontsize=8)

    return fig


def load_calc_eddy_flux(loadpath,filename):
    ds=xr.open_dataset(loadpath+filename)
    calc_eddy_flux=ds['cross_shelf_temp_flux_calc_eddy'].values
    
    return calc_eddy_flux

def plot_temp_budget_compare_CV_w_details(fig,loadpath,file_names,file_names_calc_eddy,
                                cross_shelf_loc,
                                    run_date=np.arange(120),
                                    xlims=(0,120),
                                    ylims=(-8e-7,8e-7)):
    gs=fig.add_gridspec(nrows=3,ncols=1,hspace=0.3)
    for i in range(len(file_names)):
        file_name=file_names[i]
        filename2=file_names_calc_eddy[i]
        cross_loc=cross_shelf_loc[i]
        dTdt_3d, dTdt_2d, TF_surf_3d, TF_surf_2d, TF_crossshore_3d, TF_crossshore_2d=load_2d_3d_compare_ds_w_filename(loadpath,file_name)
        calc_eddy_flux=load_calc_eddy_flux(loadpath,filename2)
        calc_mean_flux=-TF_crossshore_3d-calc_eddy_flux
        ax=fig.add_subplot(gs[i,0])
        ax.plot(run_date,dTdt_3d,label='$dT/dt_{3d}$',color='k')
        # ax.plot(run_date,-TF_crossshore_3d,label='$TF_{o, 3d}$',color='r')
        ax.plot(run_date,calc_eddy_flux,label='$TF_{o-eddy, 3d}$',color='r')
        ax.plot(run_date,calc_mean_flux,label='$TF_{o-mean, 3d}$',color='navy')
        ax.axhline(y=TF_surf_3d,label='$TF_{s, 3d}$',color='orange')
        ax.plot(run_date,dTdt_2d,label='$dT/dt_{2d}$',color='k',linestyle='dashed')
        ax.plot(run_date,-TF_crossshore_2d,label='$TF_{o-mean, 2d}$',color='navy',linestyle='dashed')
        ax.axhline(y=TF_surf_2d,label='$TF_{s, 2d}$',color='orange',linestyle='dashed')
        ax.set_title('At {:02d} km from the coast'.format(cross_loc),fontsize=8)
        ax.grid(alpha=0.5)
        ax.set_xlim(xlims)
        ax.set_ylim(ylims)
        ax.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
        ax.yaxis.get_offset_text().set_fontsize(8)
        ax.tick_params(axis='both', which='major', labelsize=8)
        ax.text(0.02, 0.9, '{}'.format(chr(97+i)), transform=ax.transAxes,zorder=10,fontsize=8)

    plt.tight_layout()    
    ax.legend(bbox_to_anchor=(0.9, -0.35),ncol=4,fontsize=7)
    # plt.subplots_adjust(left=0.07, right=0.93, wspace=0.25, hspace=0.35)
    fig.text(0.05, 0.5, 'Temperature change rate $[^\circ Cs^{-1}]$', va='center', rotation='vertical',fontsize=8)
    fig.text(0.5, 0.05, 'Time [day]', ha='center',fontsize=8)

    return fig

def plot_temp_budget_compare_CV_w_details_complex(fig,loadpath,file_names,file_names_calc_eddy,
                                cross_shelf_loc,
                                    run_date=np.arange(120),
                                    xlims=(0,120),
                                    ylims=(-8e-7,8e-7)):
    gs=fig.add_gridspec(nrows=3,ncols=1,hspace=0.3)
    for i in range(len(file_names)):
        file_name=file_names[i]
        filename2=file_names_calc_eddy[i]
        cross_loc=cross_shelf_loc[i]
        dTdt_3d, dTdt_2d, TF_surf_3d, TF_surf_2d, TF_crossshore_3d, TF_crossshore_2d=load_2d_3d_compare_ds_w_filename(loadpath,file_name)
        calc_eddy_flux=load_calc_eddy_flux(loadpath,filename2)
        calc_mean_flux=-TF_crossshore_3d-calc_eddy_flux
        ax=fig.add_subplot(gs[i,0])
        ax.plot(run_date,dTdt_3d,label='$dT/dt_{3d}$',color='k')
        ax.plot(run_date,-TF_crossshore_3d,label='$TF_{o-total, 3d}$',color='r')
        ax.plot(run_date,calc_eddy_flux,label='$TF_{o-eddy, 3d}$',color='sandybrown')
        ax.plot(run_date,calc_mean_flux,label='$TF_{o-mean, 3d}$',color='navy')
        ax.axhline(y=TF_surf_3d,label='$TF_{s, 3d}$',color='orange')
        ax.plot(run_date,dTdt_2d,label='$dT/dt_{2d}$',color='k',linestyle='dashed')
        ax.plot(run_date,-TF_crossshore_2d,label='$TF_{o-mean, 2d}$',color='navy',linestyle='dashed')
        ax.axhline(y=TF_surf_2d,label='$TF_{s, 2d}$',color='orange',linestyle='dashed')
        ax.set_title('At {:02d} km from the coast'.format(cross_loc),fontsize=8)
        ax.grid(alpha=0.5)
        ax.set_xlim(xlims)
        ax.set_ylim(ylims)
        ax.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
        ax.yaxis.get_offset_text().set_fontsize(8)
        ax.tick_params(axis='both', which='major', labelsize=8)
        ax.text(0.02, 0.9, '{}'.format(chr(97+i)), transform=ax.transAxes,zorder=10,fontsize=8)

    plt.tight_layout()    
    ax.legend(bbox_to_anchor=(0.9, -0.35),ncol=4,fontsize=7)
    # plt.subplots_adjust(left=0.07, right=0.93, wspace=0.25, hspace=0.35)
    fig.text(0.05, 0.5, 'Temperature change rate $[^\circ Cs^{-1}]$', va='center', rotation='vertical',fontsize=8)
    fig.text(0.5, 0.05, 'Time [day]', ha='center',fontsize=8)

    return fig

def load_temp_flux_3d_dep_int(loadpath,run_number):
    '''
    the temperature flux loaded here have units of m^3 degree C s^-1
    '''
    ds=xr.open_dataset(loadpath+'run{:02d}_temp_flux_components.nc'.format(run_number))
    return ds

def load_temp_flux_3d_dep_ave(loadpath,run_number):
    '''
    the temperature flux loaded here have units of m degree C s^-1
    '''
    ds=xr.open_dataset(loadpath+'run{:02d}_temp_flux_pu_components.nc'.format(run_number))
    return ds

def obtain_base_3d_flux_vars(loadpath,cal_method,run_number=32):
    '''
    the temperature flux loaded here have units of m^3 degree C s^-1 or m degree C s^-1 
    '''
    if str(cal_method) == 'dep_ave':
        ds=load_temp_flux_3d_dep_ave(loadpath,run_number)
        temp_flux_mean=ds['uT_mean_sel_rm_initial_pu']
        temp_flux_total=ds['uT_total_sel_rm_initial_pu']
        temp_flux_eddy=ds['uT_eddy_sel_rm_initial_pu']

    elif str(cal_method) == 'dep_int':
        ds=load_temp_flux_3d_dep_int(loadpath,run_number)
        temp_flux_mean=ds['uT_mean_sel_rm_initial']
        temp_flux_total=ds['uT_total_sel_rm_initial']
        temp_flux_eddy=ds['uT_eddy_sel_rm_initial']

    else:
        print('method not recognized')
    
    time=ds['time']
    cross=ds['cross_shelf']

    
    return temp_flux_mean,temp_flux_total,temp_flux_eddy,time,cross

def plot_temp_flux_eddy_n_mean_w_days(fig,sel_day,loadpath,
                                      xlims=(150,300),
                                      ylims=(0,1e-2)):
    temp_flux_mean,temp_flux_total,temp_flux_eddy,time,cross=obtain_base_3d_flux_vars(loadpath,'dep_ave')
    if len(sel_day)==6:
        gs=fig.add_gridspec(nrows=2,ncols=3,hspace=0.3,wspace=0.15)
        for i in range(len(sel_day)):
            row=i//3
            col=i%3
            day=sel_day[i]
            ax=fig.add_subplot(gs[row,col])
            eddy=ax.plot(cross,np.abs(temp_flux_eddy[day,...]),color='r',zorder=10)
            mean=ax.plot(cross,np.abs(temp_flux_mean[day,...]),color='b')
            ax.legend(['eddy','mean'],fontsize=8,loc='upper left')
            ax.set_ylim(ylims)
            ax.set_xlim(xlims)
            ax.grid(alpha=0.5)
            ax.set_title('Day '+str(sel_day[i]),fontsize=8)
            ax.tick_params(axis='both',labelsize=8)
            ax.ticklabel_format(axis='y',style='sci',scilimits=(0,0))
            ax.yaxis.get_offset_text().set_fontsize(8)
    
    elif len(sel_day)==4:
        gs=fig.add_gridspec(nrows=2,ncols=2,hspace=0.3,wspace=0.15)
        for i in range(len(sel_day)):
            row=i//2
            col=i%2
            day=sel_day[i]
            ax=fig.add_subplot(gs[row,col])
            eddy=ax.plot(cross,np.abs(temp_flux_eddy[day,...]),color='r',zorder=10)
            mean=ax.plot(cross,np.abs(temp_flux_mean[day,...]),color='b')
            ax.legend(['eddy','mean'],fontsize=8,loc='upper left')
            ax.set_ylim(ylims)
            ax.set_xlim(xlims)
            ax.grid(alpha=0.5)
            ax.set_title('Day '+str(sel_day[i]),fontsize=8)
            ax.tick_params(axis='both',labelsize=8)
            ax.ticklabel_format(axis='y',style='sci',scilimits=(0,0))
            ax.yaxis.get_offset_text().set_fontsize(8)

    else:
        print('Input error: Please select 4 or 6 number of the "sel_day"')


    fig.text(0.05, 0.5, 'Temperature flux $[^\circ Cms^{-1}]$', va='center', rotation='vertical',fontsize=8)
    #set x super label
    fig.text(0.5, 0.05, 'Cross-shelf distance [km]', ha='center',fontsize=8)

    return fig

def plot_percent_mean_eddy_stackplot(fig,loadpath,cal_method,
                                     run_number=32,
                                     day_sel=[0,20,40,60],
                                     ylims=(0,100),
                                     xlims=(150,300)):
    temp_flux_mean,temp_flux_total,temp_flux_eddy,time,cross=obtain_base_3d_flux_vars(loadpath,cal_method,run_number)
    total_flux_abs=np.abs(temp_flux_eddy)+np.abs(temp_flux_mean)
    eddy_flux_percent=np.abs(temp_flux_eddy)/total_flux_abs*100
    mean_flux_percent=np.abs(temp_flux_mean)/total_flux_abs*100
    
    gs=fig.add_gridspec(nrows=2,ncols=2,hspace=0.05,wspace=0.25)

    for i in range(len(day_sel)):
        row=i//2
        col=i%2
        day=day_sel[i]
        ax=fig.add_subplot(gs[row,col])
        ax.stackplot(np.arange(300),eddy_flux_percent[day,...],mean_flux_percent[day,...],
                     colors=['r','b',],alpha=0.5)
        ax.legend(['eddy','mean'],fontsize=8,loc='upper left')
        ax.set_ylim(ylims)
        ax.set_xlim(xlims)
        ax.set_title('Day '+str(day_sel[i]),fontsize=8)
        ax.tick_params(axis='both',labelsize=8)
        ax.set_aspect('equal')
        ax.set_xlabel('Cross-shelf distance [km]', fontsize=8)
        ax.set_ylabel('% of the total flux', fontsize=8)

        # add subplot index
        ax.text(0.02, 0.9, '{}'.format(chr(97+i)), transform=ax.transAxes,zorder=10,fontsize=8)

    plt.tight_layout()

    # fig.text(0.05, 0.5, '% of the total temperature flux', va='center', rotation='vertical',fontsize=8)
    

    return fig

### Basecase 3d investigation
def ax_frameless(ax):
    f1=ax.tick_params(axis='x', which='both', bottom=False,top=False, labelbottom=False)
    f2=ax.tick_params(axis='y', which='both', right=False,left=False, labelleft=False)
    ax.yaxis.set_label_coords(-.3, .5)
    return f1, f2

def plot_meanflow_w_rho(ax,cross,depth,meanflow,rho,
            xlims,ylims,rho_levels=np.arange(999.8,1000.8,0.05),vmin=-0.05,vmax=0.05):
    '''
    plot surface density contourf with depth averaged velocity perturbation quiver (u' v')
    default density range is in [999.8,1000.8]
    noted that the quiver plot for the velocity use every 5km in space
    '''
    P=ax.pcolormesh(cross,depth,meanflow, cmap='RdBu_r',
                  vmin=vmin,vmax=vmax)
    # Cbar=plt.colorbar(P,cax=cax)
    C=ax.contour(cross,depth,rho,levels=rho_levels,colors='k',linewidths=0.5,)
    C2=ax.contour(cross,depth,rho,levels=[1000.2],colors='k',linewidths=2,)
    C3=ax.contour(cross,depth,rho,levels=[1000.5],colors='gold',linewidths=2,)
    ax.set_xlim(xlims)
    ax.set_ylim(ylims)
#     ax.set_ylabel('Depth [m]',fontsize=8)
    ax.patch.set_facecolor('grey')

    return P, C, 

def plot_cross_KE(ax,cross,EKE,MKE,xlims,ylims):
    '''
    plot per unit mass EKE and MKE as a function of cross-shelf location 
    '''
    EKE=ax.plot(cross,EKE,color='r',label='EKE',linewidth=0.75)
    MKE=ax.plot(cross,MKE,color='navy',label='MKE',linewidth=0.75)
    ax.set_xlim(xlims)
    ax.set_ylim(ylims)
    ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    # ax.set_xticklabels([])
    # ax.set_yticklabels([])
    # ax.set_ylabel('KE [$m^2/s^2$]',fontsize=8)
    ax.legend(loc='upper left',fontsize=6,frameon=False)
    return EKE, MKE

def prep_temp_flux_3d(loadpath,
                      cal_method='dep_ave',
                      run_number=32):
    temp_flux_mean,temp_flux_total,temp_flux_eddy,time,cross=obtain_base_3d_flux_vars(loadpath,cal_method,run_number=run_number)
    # subds1=xr.open_dataset(loadpath+'run{:02d}_EnergyBudget_VA_Domain_n_PEZ.nc'.format(run_number))
    # EKE_pu=subds1['EKE_domain_vol_ave']
    # MKE_pu=subds1['MKE_domain_vol_ave']
    
    ds_baseKE=xr.open_dataset(loadpath+'run{:02d}_KE_output.nc'.format(run_number))
    EKE_pu=ds_baseKE['unit_area_EKE']
    MKE_pu=ds_baseKE['unit_area_MKE']
    u_bar=ds_baseKE['u_mean']
    rho_bar=ds_baseKE['rho_mean']
    layer_depth=ds_baseKE['layer_depth']

    subds2=xr.open_dataset(loadpath+'run{:02d}_vel_V.nc'.format(run_number))
    v_bar=subds2['V_mean']
    
    return MKE_pu,EKE_pu,temp_flux_mean,temp_flux_eddy,v_bar,u_bar,rho_bar,layer_depth

def plot_cross_temp_flux(ax,cross,eddy_flux,mean_flux,xlims,ylims):

    eddy_flux=ax.plot(cross,eddy_flux,color='r',label='eddy flux',linewidth=0.75)
    mean_flux=ax.plot(cross,mean_flux,color='navy',label='mean flux',linewidth=0.75)
    ax.set_xlim(xlims)
    ax.set_ylim(ylims)
    ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))

    ax.legend(loc='upper left',fontsize=6,frameon=False)
    return eddy_flux, mean_flux

def plot_snapshots_temp_flux(fig,loadpath,
                                      sel_dates,
                                      run_number=32,
                                      run_date=np.arange(120),
                                      cross=np.arange(300),
                                      xlims=(150,300),
                                      ylims_KE=(0,2e-3),
                                      ylims_temp_flux=(0,1e-2),
                                      ylims_sideview=(-220,0),):
    MKE_pu,EKE_pu,temp_flux_mean,temp_flux_eddy,v_bar,u_bar,rho_bar,layer_depth=prep_temp_flux_3d(loadpath)
    subplot_index=['a','b','c','d','e','f','g','h','i','j','k','l',]
    gs = fig.add_gridspec(nrows=4, ncols=4, height_ratios=[0.5,0.5,0.75,0.75],
                width_ratios=[1,1,1,0.05],hspace=0.35,wspace=0.35)
    
    ax_1=fig.add_subplot(gs[0, 0])
    ax_1.set_ylabel('KE $[m^{2}s^{-2}]$',fontsize=8)
    f1,f2=ax_frameless(ax_1)
    ax_1.yaxis.set_label_coords(-.25, .5)

    ax_2=fig.add_subplot(gs[1, 0])
    f1,f2=ax_frameless(ax_2)
    ax_2.set_ylabel('Temp flux [$^\circ Cms^{-1}$]',fontsize=8)
    ax_2.yaxis.set_label_coords(-.25, .5)
    
    ax_3=fig.add_subplot(gs[2, 0])
    f1,f2=ax_frameless(ax_3)
    ax_3.set_ylabel('z [m]',fontsize=8)
    ax_3.yaxis.set_label_coords(-.3, .5)

    ax_4=fig.add_subplot(gs[3, 0])
    f1,f2=ax_frameless(ax_4)
    ax_4.set_ylabel('z [m]',fontsize=8)
    ax_4.yaxis.set_label_coords(-.3, .5)


    for col in range(3):
        i=int(sel_dates[col])
        ax1 = fig.add_subplot(gs[0, col])
        subplot_int1=int(3*0+col)
        EKE,MKE=plot_cross_KE(ax1,cross,EKE_pu[i,...],MKE_pu[i,...],xlims,ylims_KE)
        ax1.set_title('Day {:01d}'.format(i),fontsize=8)
        ax1.text(0.02, 0.88, subplot_index[subplot_int1], 
                transform=ax1.transAxes,zorder=10,fontsize=8)
        
        ax2=fig.add_subplot(gs[1,col])
        subplot_int2=int(3*1+col)
        # p2,q2=plot_cross_temp_flux(ax2,cross,temp_flux_eddy[i,...],temp_flux_mean[i,...],xlims,ylims_temp_flux)
        p2,q2=plot_cross_temp_flux(ax2,cross,np.abs(temp_flux_eddy[i,...]),np.abs(temp_flux_mean[i,...]),xlims,ylims_temp_flux)
        ax2.text(0.02, 0.88, subplot_index[subplot_int2], 
                transform=ax2.transAxes,zorder=10,fontsize=8)

        ax3=fig.add_subplot(gs[2,col])
        subplot_int3=int(3*2+col)
        p3,c3,=plot_meanflow_w_rho(ax3,cross,-layer_depth,v_bar[i,...],rho_bar[i,...],
                xlims,ylims_sideview,rho_levels=np.arange(999.0,1002.7,0.05),vmax=0.06,vmin=-0.06)
        ax3.text(0.02, 0.9, subplot_index[subplot_int3], 
                transform=ax3.transAxes,zorder=10,fontsize=8)
        
        ax4=fig.add_subplot(gs[3,col])
        subplot_int4=int(3*3+col)
        p4,c4,=plot_meanflow_w_rho(ax4,cross,-layer_depth,u_bar[i,...],rho_bar[i,...],
                xlims,ylims_sideview,rho_levels=np.arange(999.0,1002.7,0.05),vmax=0.01,vmin=-0.01)
        ax4.set_xlabel('Cross-shelf distance [km]',fontsize=8)
        ax4.text(0.02, 0.9, subplot_index[subplot_int4], transform=ax4.transAxes,zorder=10,fontsize=8)
        

    
    cax3=fig.add_subplot(gs[2,3])
    cbar3=plt.colorbar(p3,cax=cax3)
    cbar3.ax.set_title('$\overline{V} \;[ms^{-1}]$',fontsize=8,rotation=0)
    cax4=fig.add_subplot(gs[3,3])
    cbar4=plt.colorbar(p4,cax=cax4)
    cbar4.ax.set_title('$\overline{U} \;[ms^{-1}]$',fontsize=8,rotation=0)

    return fig

def plot_snapshots_temp_flux_U(fig,loadpath,
                                      sel_dates,
                                      run_number=32,
                                      run_date=np.arange(120),
                                      cross=np.arange(300),
                                      xlims=(150,300),
                                      ylims_KE=(0,2e-3),
                                      ylims_temp_flux=(0,1e-2),
                                      ylims_sideview=(-220,0),):
    MKE_pu,EKE_pu,temp_flux_mean,temp_flux_eddy,v_bar,u_bar,rho_bar,layer_depth=prep_temp_flux_3d(loadpath)
    subplot_index=['a','b','c','d','e','f','g','h','i','j','k','l',]
    gs = fig.add_gridspec(nrows=3, ncols=4, height_ratios=[0.5,0.5,0.75],
                width_ratios=[1,1,1,0.05],hspace=0.35,wspace=0.35)
    
    ax_1=fig.add_subplot(gs[0, 0])
    ax_1.set_ylabel('EKE $[m^{2}s^{-2}]$',fontsize=8)
    f1,f2=ax_frameless(ax_1)
    ax_1.yaxis.set_label_coords(-.15, .5)

    ax_2=fig.add_subplot(gs[1, 0])
    f1,f2=ax_frameless(ax_2)
    ax_2.set_ylabel('Temp flux [$^\circ Cms^{-1}$]',fontsize=8)
    ax_2.yaxis.set_label_coords(-.25, .5)
    
    ax_3=fig.add_subplot(gs[2, 0])
    f1,f2=ax_frameless(ax_3)
    ax_3.set_ylabel('z [m]',fontsize=8)
    ax_3.yaxis.set_label_coords(-.3, .5)



    for col in range(3):
        i=int(sel_dates[col])
        ax1 = fig.add_subplot(gs[0, col])
        subplot_int1=int(3*0+col)
        EKE,MKE=plot_cross_KE(ax1,cross,EKE_pu[i,...],MKE_pu[i,...],xlims,ylims_KE)
        ax1.set_title('Day {:01d}'.format(i),fontsize=8)
        ax1.text(0.02, 0.88, subplot_index[subplot_int1], 
                transform=ax1.transAxes,zorder=10,fontsize=8)
        
        ax2=fig.add_subplot(gs[1,col])
        subplot_int2=int(3*1+col)
        # p2,q2=plot_cross_temp_flux(ax2,cross,temp_flux_eddy[i,...],temp_flux_mean[i,...],xlims,ylims_temp_flux)
        p2,q2=plot_cross_temp_flux(ax2,cross,np.abs(temp_flux_eddy[i,...]),np.abs(temp_flux_mean[i,...]),xlims,ylims_temp_flux)
        ax2.text(0.02, 0.88, subplot_index[subplot_int2], 
                transform=ax2.transAxes,zorder=10,fontsize=8)
        
        ax3=fig.add_subplot(gs[2,col])
        subplot_int3=int(3*2+col)
        p3,c3,=plot_meanflow_w_rho(ax3,cross,-layer_depth,u_bar[i,...],rho_bar[i,...],
                xlims,ylims_sideview,rho_levels=np.arange(999.0,1002.7,0.05),vmax=0.02,vmin=-0.02)
        ax3.set_xlabel('Cross-shelf distance [km]',fontsize=8)
        ax3.text(0.02, 0.9, subplot_index[subplot_int3], transform=ax3.transAxes,zorder=10,fontsize=8)
        

    
    cax3=fig.add_subplot(gs[2,3])
    cbar3=plt.colorbar(p3,cax=cax3)
    cbar3.ax.set_title('$\overline{U} \;[ms^{-1}]$',fontsize=8,rotation=0)

    plt.tight_layout()

    return fig
### Case comparsion

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

def plot_case_compare_uT_w_EKE(fig,scenario,t0s,loadpath,
                                xlims=(150,300),
                                ylims_KEs=(0,3e-3),
                                ylims_temp_flux=(0,0.01),
                                cross=np.arange(300),along=np.arange(250),):
    subplot_index=['a','b','c','d','e','f','g','h','i','j','k','l',]
    run_number_sum,run_number_name_sum,run_number_name_sum2,run_idx_sum=obtain_scenario(str(scenario))
    gs = fig.add_gridspec(nrows=2, ncols=4, height_ratios=[1,1],
                width_ratios=[1,1,1,0.05],hspace=0.35,wspace=0.35)
    ax_1=fig.add_subplot(gs[0, 0])
    ax_1.set_ylabel('EKE $[m^{2}s^{-2}]$',fontsize=8)
    f1,f2=ax_frameless(ax_1)
    ax_1.yaxis.set_label_coords(-.15, .5)

    ax_2=fig.add_subplot(gs[1, 0])
    f1,f2=ax_frameless(ax_2)
    ax_2.set_ylabel('Temp flux [$^\circ Cms^{-1}$]',fontsize=8)
    ax_2.yaxis.set_label_coords(-.25, .5)
    
    for col in range(3):
        
        run_number=run_number_sum[col]
        run_number_name=run_number_name_sum[col]
        run_number_idx=run_idx_sum[col]
        t0=int(t0s[run_number_idx])

        MKE_pu,EKE_pu,temp_flux_mean,temp_flux_eddy,v_bar,u_bar,rho_bar,layer_depth=prep_temp_flux_3d(loadpath,run_number=run_number)
        ax1 = fig.add_subplot(gs[0, col])
        subplot_int1=int(3*0+col)
        EKE,MKE=plot_cross_KE(ax1,cross,EKE_pu[t0,...],MKE_pu[t0,...],xlims,ylims_KEs)
        ax1.set_title(r'Run {:01d}'.format(run_number_name),fontsize=8)
        ax1.text(0.02, 0.88, subplot_index[subplot_int1], 
                 transform=ax1.transAxes,zorder=10,fontsize=8)
        ax1.grid(alpha=0.5)

        ax2=fig.add_subplot(gs[1,col])
        subplot_int2=int(3*1+col)
        # p2,q2=plot_cross_temp_flux(ax2,cross,temp_flux_eddy[t0,...],temp_flux_mean[t0,...],xlims,ylims_temp_flux)
        p2,q2=plot_cross_temp_flux(ax2,cross,np.abs(temp_flux_eddy[t0,...]),np.abs(temp_flux_mean[t0,...]),xlims,ylims_temp_flux)
        ax2.text(0.02, 0.88, subplot_index[subplot_int2], 
                transform=ax2.transAxes,zorder=10,fontsize=8)
        ax2.grid(alpha=0.5)
    plt.tight_layout()

    return fig

def plot_case_compare_snapshots(fig,scenario,t0s,loadpath,
                                xlims=(150,300),
                                ylims_KEs=(0,3e-3),
                                ylims_temp_flux=(0,0.01),
                                ylims_sideview=(-220,0),
                                cross=np.arange(300),along=np.arange(250),
                                rho_max=1000.7,rho_min=1000.1,
                                Vbar_max=0.05,Vbar_min=-0.05,
                                Ubar_max=0.01,Ubar_min=-0.01):
    subplot_index=['a','b','c','d','e','f','g','h','i','j','k','l',]
    run_number_sum,run_number_name_sum,run_number_name_sum2,run_idx_sum=obtain_scenario(str(scenario))
    gs = fig.add_gridspec(nrows=4, ncols=4, height_ratios=[0.5,0.5,0.75,0.75],
                width_ratios=[1,1,1,0.05],hspace=0.35,wspace=0.35)
    ax_1=fig.add_subplot(gs[0, 0])
    ax_1.set_ylabel('EKE $[m^{2}s^{-2}]$',fontsize=8)
    f1,f2=ax_frameless(ax_1)
    ax_1.yaxis.set_label_coords(-.15, .5)

    ax_2=fig.add_subplot(gs[1, 0])
    f1,f2=ax_frameless(ax_2)
    ax_2.set_ylabel('Temp flux [$^\circ Cms^{-1}$]',fontsize=8)
    ax_2.yaxis.set_label_coords(-.25, .5)
    
    ax_3=fig.add_subplot(gs[2, 0])
    f1,f2=ax_frameless(ax_3)
    ax_3.set_ylabel('z [m]',fontsize=8)
    ax_3.yaxis.set_label_coords(-.3, .5)

    ax_4=fig.add_subplot(gs[3, 0])
    f1,f2=ax_frameless(ax_4)
    ax_4.set_ylabel('z [m]',fontsize=8)
    ax_4.yaxis.set_label_coords(-.3, .5)


    for col in range(3):
        
        run_number=run_number_sum[col]
        run_number_name=run_number_name_sum[col]
        run_number_idx=run_idx_sum[col]
        t0=int(t0s[run_number_idx])

        MKE_pu,EKE_pu,temp_flux_mean,temp_flux_eddy,v_bar,u_bar,rho_bar,layer_depth=prep_temp_flux_3d(loadpath,run_number=run_number)
        ax1 = fig.add_subplot(gs[0, col])
        subplot_int1=int(3*0+col)
        EKE,MKE=plot_cross_KE(ax1,cross,EKE_pu[t0,...],MKE_pu[t0,...],xlims,ylims_KEs)
        ax1.set_title(r'Run {:01d}'.format(run_number_name),fontsize=8)
        ax1.text(0.02, 0.88, subplot_index[subplot_int1], 
                 transform=ax1.transAxes,zorder=10,fontsize=8)

        ax2=fig.add_subplot(gs[1,col])
        subplot_int2=int(3*1+col)
        p2,q2=plot_cross_temp_flux(ax2,cross,temp_flux_eddy[t0,...],temp_flux_mean[t0,...],xlims,ylims_temp_flux)
        ax2.text(0.02, 0.88, subplot_index[subplot_int2], 
                transform=ax2.transAxes,zorder=10,fontsize=8)

        ax3=fig.add_subplot(gs[2,col])
        subplot_int3=int(3*2+col)
        p3,c3,=plot_meanflow_w_rho(ax3,cross,-layer_depth,v_bar[t0,...],rho_bar[t0,...],
                xlims,ylims_sideview,rho_levels=np.arange(999.0,1002.7,0.05),vmax=Vbar_max,vmin=Vbar_min)
        ax3.text(0.02, 0.9, subplot_index[subplot_int3], 
                transform=ax3.transAxes,zorder=10,fontsize=8)
        
        ax4=fig.add_subplot(gs[3,col])
        subplot_int4=int(3*3+col)
        p4,c4,=plot_meanflow_w_rho(ax4,cross,-layer_depth,u_bar[t0,...],rho_bar[t0,...],
                xlims,ylims_sideview,rho_levels=np.arange(999.0,1002.7,0.05),vmax=Ubar_max,vmin=Ubar_min)
        ax4.set_xlabel('Cross-shelf distance [km]',fontsize=8)
        ax4.text(0.02, 0.9, subplot_index[subplot_int4], transform=ax4.transAxes,zorder=10,fontsize=8)

        
    cax3=fig.add_subplot(gs[2,3])
    cbar3=plt.colorbar(p3,cax=cax3)
    cbar3.ax.set_title('$\overline{V} \;[ms^{-1}]$',fontsize=8,rotation=0)
    cax4=fig.add_subplot(gs[3,3])
    cbar4=plt.colorbar(p4,cax=cax4)
    cbar4.ax.set_title('$\overline{U} \;[ms^{-1}]$',fontsize=8,rotation=0)

    plt.tight_layout()

    return fig

'''
def plot_case_compare_snapshots_U(fig,scenario,t0s,loadpath,
                                xlims=(150,300),
                                ylims_KEs=(0,3e-3),
                                ylims_temp_flux=(0,0.01),
                                ylims_sideview=(-220,0),
                                cross=np.arange(300),
                                Ubar_max=0.01,Ubar_min=-0.01):
    subplot_index=['a','b','c','d','e','f','g','h','i','j','k','l',]
    run_number_sum,run_number_name_sum,run_number_name_sum2,run_idx_sum=obtain_scenario(str(scenario))
    gs = fig.add_gridspec(nrows=3, ncols=4, height_ratios=[0.5,0.5,0.75,],
                width_ratios=[1,1,1,0.05],hspace=0.35,wspace=0.35)
    ax_1=fig.add_subplot(gs[0, 0])
    ax_1.set_ylabel('EKE $[m^{2}/s^{2}]$',fontsize=8)
    f1,f2=ax_frameless(ax_1)
    ax_1.yaxis.set_label_coords(-.15, .5)

    ax_2=fig.add_subplot(gs[1, 0])
    f1,f2=ax_frameless(ax_2)
    ax_2.set_ylabel('Temp flux [$^\circ Cms^{-1}$]',fontsize=8)
    ax_2.yaxis.set_label_coords(-.25, .5)
    
    ax_3=fig.add_subplot(gs[2, 0])
    f1,f2=ax_frameless(ax_3)
    ax_3.set_ylabel('Depth [m]',fontsize=8)
    ax_3.yaxis.set_label_coords(-.3, .5)


    for col in range(3):
        
        run_number=run_number_sum[col]
        run_number_name=run_number_name_sum[col]
        run_number_idx=run_idx_sum[col]
        t0=int(t0s[run_number_idx])

        MKE_pu,EKE_pu,temp_flux_mean,temp_flux_eddy,v_bar,u_bar,rho_bar,layer_depth=prep_temp_flux_3d(loadpath,run_number=run_number)
        ax1 = fig.add_subplot(gs[0, col])
        subplot_int1=int(3*0+col)
        EKE,MKE=plot_cross_KE(ax1,cross,EKE_pu[t0,...],MKE_pu[t0,...],xlims,ylims_KEs)
        ax1.set_title(r'Run {:01d}'.format(run_number_name),fontsize=8)
        ax1.text(0.02, 0.88, subplot_index[subplot_int1], 
                 transform=ax1.transAxes,zorder=10,fontsize=8)

        ax2=fig.add_subplot(gs[1,col])
        subplot_int2=int(3*1+col)
        p2,q2=plot_cross_temp_flux(ax2,cross,temp_flux_eddy[t0,...],temp_flux_mean[t0,...],xlims,ylims_temp_flux)
        ax2.text(0.02, 0.88, subplot_index[subplot_int2], 
                transform=ax2.transAxes,zorder=10,fontsize=8)

        ax3=fig.add_subplot(gs[2,col])
        subplot_int4=int(3*2+col)
        p3,c3,=plot_meanflow_w_rho(ax3,cross,-layer_depth,u_bar[t0,...],rho_bar[t0,...],
                xlims,ylims_sideview,rho_levels=np.arange(999.0,1002.7,0.05),vmax=Ubar_max,vmin=Ubar_min)
        ax3.set_xlabel('Cross-shelf distance [km]',fontsize=8)
        ax3.text(0.02, 0.9, subplot_index[subplot_int4], transform=ax3.transAxes,zorder=10,fontsize=8)

        
    cax3=fig.add_subplot(gs[2,3])
    cbar3=plt.colorbar(p3,cax=cax3)
    cbar3.ax.set_title('$\overline{U}$',fontsize=8,rotation=0)

    plt.tight_layout()

    return fig'''

def plot_case_compare_snapshots_U(fig,scenario,t0s,loadpath,
                                xlims=(150,300),
                                ylims_KEs=(0,3e-3),
                                ylims_temp_flux=(0,0.01),
                                ylims_sideview=(-220,0),
                                cross=np.arange(300),
                                Ubar_max=0.01,Ubar_min=-0.01):
    subplot_index=['a','b','c','d','e','f','g','h','i','j','k','l',]
    run_number_sum,run_number_name_sum,run_number_name_sum2,run_idx_sum=obtain_scenario(str(scenario))
    gs = fig.add_gridspec(nrows=3, ncols=4, height_ratios=[0.5,0.5,0.75,],
                width_ratios=[1,1,1,0.05],hspace=0.35,wspace=0.35)
    ax_1=fig.add_subplot(gs[0, 0])
    ax_1.set_ylabel('KE $[m^{2}s^{-2}]$',fontsize=8)
    f1,f2=ax_frameless(ax_1)
    ax_1.yaxis.set_label_coords(-.15, .5)

    ax_2=fig.add_subplot(gs[1, 0])
    f1,f2=ax_frameless(ax_2)
    ax_2.set_ylabel('Temp flux [$^\circ Cms^{-1}$]',fontsize=8)
    ax_2.yaxis.set_label_coords(-.25, .5)
    
    ax_3=fig.add_subplot(gs[2, 0])
    f1,f2=ax_frameless(ax_3)
    ax_3.set_ylabel('z [m]',fontsize=8)
    ax_3.yaxis.set_label_coords(-.3, .5)


    for col in range(3):
        
        run_number=run_number_sum[col]
        run_number_name=run_number_name_sum[col]
        run_number_idx=run_idx_sum[col]
        t0=int(t0s[run_number_idx])

        MKE_pu,EKE_pu,temp_flux_mean,temp_flux_eddy,v_bar,u_bar,rho_bar,layer_depth=prep_temp_flux_3d(loadpath,run_number=run_number)
        ax1 = fig.add_subplot(gs[0, col])
        subplot_int1=int(3*0+col)
        EKE,MKE=plot_cross_KE(ax1,cross,EKE_pu[t0,...],MKE_pu[t0,...],xlims,ylims_KEs)
        ax1.set_title(r'Run {:01d}'.format(run_number_name),fontsize=8)
        ax1.text(0.02, 0.88, subplot_index[subplot_int1], 
                 transform=ax1.transAxes,zorder=10,fontsize=8)

        ax2=fig.add_subplot(gs[1,col])
        subplot_int2=int(3*1+col)
        # p2,q2=plot_cross_temp_flux(ax2,cross,temp_flux_eddy[t0,...],temp_flux_mean[t0,...],xlims,ylims_temp_flux)
        p2,q2=plot_cross_temp_flux(ax2,cross,np.abs(temp_flux_eddy[t0,...]),np.abs(temp_flux_mean[t0,...]),xlims,ylims_temp_flux)
        ax2.text(0.02, 0.88, subplot_index[subplot_int2], 
                transform=ax2.transAxes,zorder=10,fontsize=8)

        ax3=fig.add_subplot(gs[2,col])
        subplot_int4=int(3*2+col)
        p3,c3,=plot_meanflow_w_rho(ax3,cross,-layer_depth,u_bar[t0,...],rho_bar[t0,...],
                xlims,ylims_sideview,rho_levels=np.arange(999.0,1002.7,0.05),vmax=Ubar_max,vmin=Ubar_min)
        ax3.set_xlabel('Cross-shelf distance [km]',fontsize=8)
        ax3.text(0.02, 0.9, subplot_index[subplot_int4], transform=ax3.transAxes,zorder=10,fontsize=8)

        
    cax3=fig.add_subplot(gs[2,3])
    cbar3=plt.colorbar(p3,cax=cax3)
    cbar3.ax.set_title('$\overline{U} \;[ms^{-1}]$',fontsize=8,rotation=0)

    plt.tight_layout()

    return fig

def plot_case_compare_time_series_uT_at_25_n_50(fig,scenario,loadpath,t0s,
                                    cross_shelf_loc1=275,
                                    cross_shelf_loc2=250,
                                    run_date=np.arange(120),
                                    xlims=(0,120),
                                    ylims=(0,1e-2)):
    run_number_sum,run_number_name_sum,run_number_name_sum2,run_idx_sum=obtain_scenario(str(scenario))
 
    gs = fig.add_gridspec(nrows=2, ncols=1)
    color_group=['b','k','r']
    ax1=fig.add_subplot(211)
    ax2=fig.add_subplot(212)
    for k in range(len(run_number_sum)):
        run_number=run_number_sum[k]
        run_idx=int(run_idx_sum[k])

        temp_flux_mean,temp_flux_total,temp_flux_eddy,time,cross=obtain_base_3d_flux_vars(loadpath,'dep_ave',run_number)
        t0=int(t0s[run_idx])

        ax1.plot(run_date,temp_flux_eddy[:,cross_shelf_loc1],label='run '+ '{}'.format(run_number_name_sum2[k]),color=color_group[k])
        ax1.scatter(t0,temp_flux_eddy[t0,cross_shelf_loc1],marker='x',color=color_group[k],s=20,zorder=2,clip_on=False)

        ax2.plot(run_date,temp_flux_eddy[:,cross_shelf_loc2],label='run '+ '{}'.format(run_number_name_sum2[k]),color=color_group[k])
        ax2.scatter(t0,temp_flux_eddy[t0,cross_shelf_loc2],marker='x',color=color_group[k],s=20,zorder=2,clip_on=False)


    ax1.set_title('At {:02d} km offshore'.format(300-cross_shelf_loc1),fontsize=8)
    ax1.grid(alpha=0.5)
    ax1.set_xlim(xlims)
    ax1.set_ylim(ylims)
    ax1.legend(fontsize=8)
    ax1.tick_params(axis='both',which='major',labelsize=8)
    ax1.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    ax1.set_ylabel(r"Eddy flux $\langle \overline{u'T'} \rangle \; [^\circ C ms^{-1}]$",fontsize=8)
    ax1.set_xlabel('Time [day]',fontsize=8)


    ax2.set_title('At {:02d} km offshore'.format(300- cross_shelf_loc2),fontsize=8)
    ax2.grid(alpha=0.5)
    ax2.set_xlim(xlims)
    ax2.set_ylim(ylims)
    ax2.legend(fontsize=8)
    ax2.tick_params(axis='both',which='major',labelsize=8)
    ax2.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    ax2.set_ylabel(r"Eddy flux $\langle \overline{u'T'} \rangle \; [^\circ C ms^{-1}]$",fontsize=8)
    ax2.set_xlabel('Time [day]',fontsize=8)

    # add subplot label
    ax1.text(0.015, 0.90, '{}'.format(chr(97)), transform=ax1.transAxes,zorder=20,fontsize=9)
    ax2.text(0.015, 0.90, '{}'.format(chr(98)), transform=ax2.transAxes,zorder=20,fontsize=9)
    plt.tight_layout()

    return fig




# eddy diffusivity
def calc_parameterized_eddy_diffusivity(k,Bs,fs):
    run_date=np.arange(120)
    B=np.abs(Bs[k].values)
    f=fs[k].values
    kappa_param=B*run_date*86400/np.abs(f)
    return kappa_param

def obtain_peak_EKE_loc(loadpath):
    # ori_ds=xr.open_dataset(loadpath+'fitted_loc_w_wavelength.nc')
    # fitted_loc=ori_ds['fitted_loc']
    ds_loc=xr.open_dataset(loadpath+'fitted_m_real_peak_loc.nc')
    fitted_loc=ds_loc['modified_peak_EKE_loc']

    return fitted_loc

def obtain_eddy_diffusivity_from_fastest_mode(k,run_numbers,fitted_loc,loadpath,dy=1e3,Ny=250):
    run_number=int(run_numbers[k])
    # ds_KE=xr.open_dataset(loadpath+'run{:02d}_KE_output.nc'.format(run_number))
    # layer_depth=ds_KE['layer_depth']
    # dz=(layer_depth[1]-layer_depth[0]).values
    # rho_mean=ds_KE['rho_mean'][0,:,:]
    # ver_layers=np.count_nonzero(np.nan_to_num(rho_mean),axis=0)

    # EKE_max_loc=fitted_loc[k,...]
    # Nz=EKE_max_loc.copy()
    # Nz=Nz.astype(int)
    # Nz=ver_layers[Nz]
    # cross_section_area=dy*Ny*dz*Nz

    # ds_to_open=xr.open_dataset(loadpath+'run{:02d}_eddy_flux_para.nc'.format(run_number))
    # uT_eddy_sel_rm_initial=ds_to_open['uT_eddy_sel_rm_initial']
    # ds_to_open2=xr.open_dataset(loadpath+'run{:02d}_eddy_flux_components.nc'.format(run_number))
    # dTbar_dx=ds_to_open2['dTbar_dx_da_sel']
    # kappa_obs=np.abs(uT_eddy_sel_rm_initial/dTbar_dx)/cross_section_area


    ds_to_open=xr.open_dataset(loadpath+'run{:02d}_eddy_flux_para_update.nc'.format(run_number))
    uT_eddy_sel_rm_initial=ds_to_open['uT_eddy_sel_rm_initial']

    ds_to_open2=xr.open_dataset(loadpath+'run{:02d}_eddy_flux_components_update.nc'.format(run_number))
    dTbar_dx=ds_to_open2['dTbar_dx_da_sel']

    kappa_obs=np.abs(uT_eddy_sel_rm_initial/dTbar_dx)

    return kappa_obs # [m^2s^-s] from the fastest eddy growth

def obtain_run_number_n_names(k,run_numbers,run_number_names):
    run_number=int(run_numbers[k])
    run_number_name=run_number_names[k]
    
    return run_number,run_number_name

def obtain_kappa_Btf_fit(loadpath,run_date=np.arange(120)):
    run_numbers,run_number_names,t0s,tis,Bs,Slope_burgers,alphas,N0s,fs=obtain_run_case_summary(loadpath)
    fitted_loc=obtain_peak_EKE_loc(loadpath)
    kappa_param_sum=[]
    kappa_obs_sum=[]

    for k in range(12):
        run_number=int(run_numbers[k])
        start=int(tis[k])
        end=int(t0s[k])

        kappa_obs=obtain_eddy_diffusivity_from_fastest_mode(k,run_numbers,fitted_loc,loadpath)
        kappa_param=calc_parameterized_eddy_diffusivity(k,Bs,fs)

        kappa_param_sum.append(kappa_param[start:end])
        kappa_obs_sum.append(kappa_obs[start:end])


    kappa_param_develop=np.concatenate(kappa_param_sum)
    kappa_obs_develop=np.concatenate(kappa_obs_sum)

    kappa_Btf_fit=np.polyfit(kappa_param_develop,kappa_obs_develop,1)
    return kappa_Btf_fit

def plot_eddy_diffusivity_obs_vs_pred_V2_presentation(loadpath,fig,
                                      xlims=(1e-2,1e3),
                                      ylims=(1e-2,1e3)):
    color_group=['k','k','grey','grey','orange','orange','r','r','b','b','g','g']
    run_numbers,run_number_names,t0s,tis,Bs,Slope_burgers,alphas,N0s,fs=obtain_run_case_summary(loadpath)
    fitted_loc=obtain_peak_EKE_loc(loadpath)
    ax=fig.add_subplot(111)
    fit_kappa=obtain_kappa_Btf_fit(loadpath)

    for k in range(1,12,2):
        run_number,run_number_name=obtain_run_number_n_names(k,run_numbers,run_number_names)
        kappa_obs=obtain_eddy_diffusivity_from_fastest_mode(k,run_numbers,fitted_loc,loadpath)
        kappa_param=calc_parameterized_eddy_diffusivity(k,Bs,fs)
        start=int(tis[k])
        end=int(t0s[k])
        ax.scatter(fit_kappa[0]*kappa_param[start:end]+fit_kappa[1],kappa_obs[start:end],
                    color=color_group[k],marker='x',s=15,label='run{:02d}'.format(run_number_name))
    for k in range(0,12,2):
        run_number,run_number_name=obtain_run_number_n_names(k,run_numbers,run_number_names)
        kappa_obs=obtain_eddy_diffusivity_from_fastest_mode(k,run_numbers,fitted_loc,loadpath)
        kappa_param=calc_parameterized_eddy_diffusivity(k,Bs,fs)
        start=int(tis[k])
        end=int(t0s[k])
        ax.scatter(fit_kappa[0]*kappa_param[start:end]+fit_kappa[1],kappa_obs[start:end],
                    color=color_group[k],marker=',',s=10,label='run{:02d},  S={:.2f}'.format(run_number_name, Slope_burgers[k]))

    ax.set_xlabel(r'$K =0.034 Bt/f \;[m^2s^{-1}]$',fontsize=8)    
    ax. set_ylabel(r'$K_{obs} \; [m^2s^{-1}]$',fontsize=8)
    ax.legend(ncol=2,fontsize=8,loc='upper left')
    ax.tick_params(which='minor', labelleft=False,labelbottom=False)
    ax.tick_params(labelsize=8)
    ax.grid()
    ax.set_xlim(xlims)
    ax.set_ylim(ylims)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_aspect('equal')
    fit_x=np.linspace(1e-1,1e3,100)
    ax.plot(fit_x,fit_x,'k--',linewidth=0.5)
    ax.text(0.55,0.85,'$ K= {:.3f} (Bt)/f$'.format(fit_kappa[0]),
                fontsize=8,transform=ax.transAxes)
    
    return fig

def plot_eddy_diffusivity_obs_vs_pred(loadpath,fig,
                                      xlims=(1e2,1e4),
                                      ylims=(1e-2,1e4)):
    color_group=['k','k','grey','grey','orange','orange','r','r','b','b','g','g']
    run_numbers,run_number_names,t0s,tis,Bs,Slope_burgers,alphas,N0s,fs=obtain_run_case_summary(loadpath)
    fitted_loc=obtain_peak_EKE_loc(loadpath)
    ax=fig.add_subplot(111)
    for k in range(1,12,2):

        run_number,run_number_name=obtain_run_number_n_names(k,run_numbers,run_number_names)
        kappa_obs=obtain_eddy_diffusivity_from_fastest_mode(k,run_numbers,fitted_loc,loadpath)
        kappa_param=calc_parameterized_eddy_diffusivity(k,Bs,fs)
        start=int(tis[k])
        end=int(t0s[k])
        ax.scatter(kappa_param[start:end],kappa_obs[start:end],
                    color=color_group[k],marker='x',s=15,label='run{:02d}'.format(run_number_name))
    for k in range(0,12,2):
        run_number,run_number_name=obtain_run_number_n_names(k,run_numbers,run_number_names)
        kappa_obs=obtain_eddy_diffusivity_from_fastest_mode(k,run_numbers,fitted_loc,loadpath)
        kappa_param=calc_parameterized_eddy_diffusivity(k,Bs,fs)
        start=int(tis[k])
        end=int(t0s[k])
        ax.scatter(kappa_param[start:end],kappa_obs[start:end],
                    color=color_group[k],marker=',',s=10,label='run{:02d}'.format(run_number_name))

    ax.set_xlabel(r'$K \sim Bt/f \;[m^2s^{-1}]$',fontsize=8)    
    ax. set_ylabel(r'$K_{obs} \; [m^2s^{-1}]$',fontsize=8)
    ax.legend(ncol=2,fontsize=8,loc='upper left')
    ax.tick_params(which='minor', labelleft=False,labelbottom=False)
    ax.tick_params(labelsize=8)
    ax.grid()
    ax.set_xlim(xlims)
    ax.set_ylim(ylims)
    ax.set_xscale('log')
    ax.set_yscale('log')
    # ax.set_aspect('equal')
    fit_kappa=obtain_kappa_Btf_fit(loadpath)
    ax.text(0.75,0.65,'$ K_{obs}$'+'$= {:.3f} (Bt)/f$'.format(fit_kappa[0]),
                fontsize=8,transform=ax.transAxes)
    
    return fig