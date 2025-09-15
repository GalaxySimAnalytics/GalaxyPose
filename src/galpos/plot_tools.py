
from typing import TYPE_CHECKING, Optional
import numpy as np
import matplotlib.pyplot as plt 
from matplotlib.figure import Figure

from .plot_utils import sfr_virus_radial_evolution, hist_2d, SCIENCEPLOT


if TYPE_CHECKING:
    from pynbody.snapshot import SimSnap




def plot_sfr_evolution(
    current: "SimSnap", 
    birth_centered: "SimSnap", 
    birth_aligned: "SimSnap",
    sfh_color: str = "inferno",
    mass_color: str = "jet",
    r_max: Optional[float] = None
    ) -> Figure:

    plt.rcParams.update(SCIENCEPLOT)
    
    if r_max is None:
        r95 = float(np.percentile(current.s['r'],95)) # include 95% particles
    else:
        r95 = r_max
    region_r = float((r95//5+1)*5)
    
    r_range = (0, region_r)
    t_range = (0, 13.80272)

    # setup time bins
    age_bins_min = np.arange(0,14)
    age_bins_max = age_bins_min + 1
    tform_bins_min = 13.80272 - age_bins_max
    tform_bins_max = 13.80272 - age_bins_min
    
    # fig grid
    n_row =9 
    n_col = 16 # the final for colorbar

    
    frac = 1.2
    fig = plt.figure(dpi=150,figsize=(frac*n_col,frac*n_row))
    gs=fig.add_gridspec(n_row,n_col,wspace=0.0,hspace=0)


    not_use = plt.subplot(gs[:3,0]) 
    not_use.set_axis_off()



    sfh_ax = plt.subplot(gs[:3,1:-1])

    im = sfr_virus_radial_evolution(
        birth_aligned['tform'], birth_aligned['mass'], birth_aligned['r'], 
        r_range=r_range,t_range=t_range, r_nbins=30*3, t_nbins=140*3
        )
    
    sfh_im = sfh_ax.imshow(im, origin='lower', extent=(*t_range,*r_range), aspect='auto', norm='log', cmap=sfh_color)

    sfh_ax_facecolor = sfh_im.cmap(sfh_im.norm(sfh_im.get_clim()[0]))
    sfh_ax.set_facecolor(sfh_ax_facecolor)
    sfh_ax.tick_params(axis='both',which='both',direction='out')
    sfh_ax.xaxis.set_ticks_position('top')
    sfh_ax.xaxis.set_label_position('top')
    sfh_ax.set_xlim(0,14)
    sfh_ax.axvline(t_range[-1], color='r', linewidth=1)
    sfh_ax.set_xticks([0,1,2,3,4,5,6,7,8,9,10,11,12,13,13.8])
    _ = sfh_ax.set_xticklabels(list(map(str,[0,1,2,3,4,5,6,7,8,9,10,11,12,13,13.8])))
    sfh_ax.set_xlabel('Lookback time [Gyr]')
    sfh_ax.set_ylabel('R [kpc]')


    sfh_bar = plt.subplot(gs[:3, -1])
    fig.colorbar(
        sfh_im,
        shrink=3/4,
        cax=sfh_bar,
        label=r"$\Sigma_{SFR}\ [\rm M_\odot\ Gyr^{-1} \ kpc^{-2}]$",
        extend='both'
    )
    sfh_bar_pos = sfh_bar.get_position()
    sfh_bar.set_position((sfh_bar_pos.x0, sfh_bar_pos.y0, sfh_bar_pos.width*0.1, sfh_bar_pos.height))
    sfh_bar.tick_params(direction='in', right=True)



    mass_panels = np.array([[plt.subplot(gs[i+3,j]) for j in range(n_col-1)] for i in range(6)])

    x_range = (-region_r, region_r)
    y_range = (-region_r, region_r)
    nbins = 200


    for i in range(n_col-1):
        if i == 0:
            im_cur, _, _ = hist_2d(
                current.s['x'], current.s['y'],weights=current.s['mass'],
                x_range=x_range,y_range=y_range,nbins=nbins
                )
            
            face_vmax=np.percentile(im_cur[im_cur>0],99.95)
            face_vmin=np.percentile(im_cur[im_cur>0],0.1)
            
            mass_im = mass_panels[0,0].imshow(
                im_cur, origin='lower', extent=(*x_range,*y_range), norm='log', 
                cmap=mass_color, vmin=face_vmin, vmax=face_vmax
                )
            im, _, _ = hist_2d(
                current.s['x'], current.s['z'],weights=current.s['mass'],
                x_range=x_range,y_range=y_range,nbins=nbins
                )
            mass_panels[1,0].imshow(
                im, origin='lower', extent=(*x_range,*y_range), norm='log', 
                cmap=mass_color, vmin=face_vmin, vmax=face_vmax
                )
            continue
        if i >=16:
            continue
        sel = (current.s['age']>age_bins_min[i-1])&(current.s['age']<age_bins_max[i-1])
        selstar = current.s[sel]
        im,_,_ = hist_2d(
            selstar['x'], selstar['y'],weights=selstar['mass'],x_range=x_range,y_range=y_range,nbins=nbins
            )
        mass_panels[0,i].imshow(
            im, origin='lower', extent=(*x_range,*y_range), norm='log', 
            cmap=mass_color, vmin=face_vmin, vmax=face_vmax
            )
        im,_,_ = hist_2d(
            selstar['x'], selstar['z'],weights=selstar['mass'],
            x_range=x_range,y_range=y_range,nbins=nbins
            )
        mass_panels[1,i].imshow(
            im, origin='lower', extent=(*x_range,*y_range), norm='log', 
            cmap=mass_color, vmin=face_vmin, vmax=face_vmax
            )

    for i in range(n_col-1):
        if i == 0:
            im, _, _ = hist_2d(
                birth_aligned.s['x'], birth_aligned.s['y'],weights=birth_aligned.s['mass'],
                density=True,x_range=x_range,y_range=y_range,nbins=nbins
                )
            mass_panels[2,0].imshow(
                im, origin='lower', extent=(*x_range,*y_range), norm='log', 
                cmap=mass_color, vmin=face_vmin, vmax=face_vmax
                )
            im, _, _ = hist_2d(
                birth_aligned.s['x'], birth_aligned.s['z'],weights=birth_aligned.s['mass'],
                density=True,x_range=x_range,y_range=y_range,nbins=nbins
                )
            mass_panels[3,0].imshow(
                im, origin='lower', extent=(*x_range,*y_range), norm='log', 
                cmap=mass_color, vmin=face_vmin, vmax=face_vmax
                )
            continue
        if i >=16:
            continue
        sel = (birth_aligned.s['tform']>tform_bins_min[i-1])&(birth_aligned.s['tform']<tform_bins_max[i-1])
        selstar = birth_aligned.s[sel]
        im,_,_ = hist_2d(
            selstar['x'], selstar['y'],weights=selstar['mass'],density=True,
            x_range=x_range,y_range=y_range,nbins=nbins)
        mass_panels[2,i].imshow(
            im, origin='lower', extent=(*x_range,*y_range), norm='log', 
            cmap=mass_color, vmin=face_vmin, vmax=face_vmax)
        im,_,_ = hist_2d(
            selstar['x'], selstar['z'],weights=selstar['mass'],density=True,
            x_range=x_range,y_range=y_range,nbins=nbins)
        mass_panels[3,i].imshow(
            im, origin='lower', extent=(*x_range,*y_range), norm='log', 
            cmap=mass_color, vmin=face_vmin, vmax=face_vmax)
        
    for i in range(n_col-1):
        st = birth_centered
        if i == 0:
            im, _, _ = hist_2d(
                st.s['x'], st.s['y'],weights=st.s['mass'],density=True,
                x_range=x_range,y_range=y_range,nbins=nbins
                )
            mass_panels[4,0].imshow(
                im, origin='lower', extent=(*x_range,*y_range), norm='log', 
                cmap=mass_color, vmin=face_vmin, vmax=face_vmax
                )
            im, _, _ = hist_2d(
                st.s['x'], st.s['z'],weights=st.s['mass'],density=True,
                x_range=x_range,y_range=y_range,nbins=nbins)
            mass_panels[5,0].imshow(
                im, origin='lower', extent=(*x_range,*y_range), norm='log', 
                cmap=mass_color, vmin=face_vmin, vmax=face_vmax
                )
            continue
        if i >=16:
            continue
        sel = (st.s['tform']>tform_bins_min[i-1])&(st.s['tform']<tform_bins_max[i-1])
        selstar = st.s[sel]
        im,_,_ = hist_2d(
            selstar['x'], selstar['y'],weights=selstar['mass'],
            density=True,x_range=x_range,y_range=y_range,nbins=nbins
            )
        mass_panels[4,i].imshow(
            im, origin='lower', extent=(*x_range,*y_range), 
            norm='log', cmap=mass_color, vmin=face_vmin, vmax=face_vmax
            )
        im,_,_ = hist_2d(
            selstar['x'], selstar['z'],weights=selstar['mass'],density=True,
            x_range=x_range,y_range=y_range,nbins=nbins)
        mass_panels[5,i].imshow(
            im, origin='lower', extent=(*x_range,*y_range), 
            norm='log', cmap=mass_color, vmin=face_vmin, vmax=face_vmax
            )

    mass_panels_face_color = mass_im.cmap(mass_im.norm(mass_im.get_clim()[0]))
    for i in mass_panels.flatten():
        i.set_facecolor(mass_panels_face_color)
        
    for i in range(mass_panels.shape[0]):
        for j in range(mass_panels.shape[1]):
            if (i!=mass_panels.shape[0]-1) or (j%2 == 1):
                    mass_panels[i,j].set_xticklabels([])
            if j!=0:
                mass_panels[i,j].set_yticklabels([])
            mass_panels[i,j].tick_params(axis='both',which='both',direction='out')
            if i==0:
                mass_panels[i,j].tick_params(axis='x',which='both',direction='in')
            if j==mass_panels.shape[1]-1:
                mass_panels[i,j].tick_params(axis='y',which='both',direction='in')


    pos0 = mass_panels[0,0].get_position()
    pos1 = mass_panels[1,0].get_position()
    ycenter = (pos0.y0 + pos1.y1) / 2
    xleft = pos0.x0
    fig.text(xleft - 0.03, ycenter, 'Current', va='center', ha='right', rotation='vertical')


    pos0 = mass_panels[2,0].get_position()
    pos1 = mass_panels[3,0].get_position()
    ycenter = (pos0.y0 + pos1.y1) / 2
    xleft = pos0.x0
    fig.text(xleft - 0.03, ycenter, 'Birth Aligned', va='center', ha='right', rotation='vertical')

    pos0 = mass_panels[4,0].get_position()
    pos1 = mass_panels[5,0].get_position()
    ycenter = (pos0.y0 + pos1.y1) / 2
    xleft = pos0.x0
    fig.text(xleft - 0.03, ycenter, 'Birth Centered', va='center', ha='right', rotation='vertical')

    pos0 = mass_panels[-1,0].get_position()
    pos1 = mass_panels[-1,-1].get_position()
    xcenter = (pos0.x0 + pos1.x1) / 2
    yleft = pos0.y0
    fig.text(xcenter, yleft - 0.03, 'X [kpc]', va='top', ha='center')


    mass_bar = plt.subplot(gs[3:, -1])
    fig.colorbar(
        mass_im,
        shrink=3/4,
        cax=mass_bar,
        label=r"$\Sigma_* \ [\rm M_\odot/kpc^2]$",
        extend='both'
    )
    mass_bar_bar_pos = mass_bar.get_position()
    mass_bar.set_position(
        (mass_bar_bar_pos.x0, mass_bar_bar_pos.y0, mass_bar_bar_pos.width*0.1, mass_bar_bar_pos.height)
        )
    mass_bar.tick_params(direction='in', right=True)
    return fig