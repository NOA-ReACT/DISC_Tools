import seaborn as sns
sns.set_style('ticks')
sns.set_context('poster')

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize, ListedColormap, LinearSegmentedColormap, ColorConverter

from . import colormaps 

import numpy as np
import xarray as xr
import pandas as pd



def add_colorbar(ax, cm, label, on_left=False, horz_buffer=0.025, width_ratio="1.25%"):
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    
    if on_left:
        bbox_left = 0 - horz_buffer
    else:
        bbox_left = 1 + horz_buffer
       
    cax = inset_axes(ax,
                     width=width_ratio,  # percentage of parent_bbox width
                     height="100%",  # height : 50%
                     loc=3,
                     bbox_to_anchor=(bbox_left,0,1,1),
                     bbox_transform=ax.transAxes,
                     borderpad=0.0,
                     )       
    return plt.colorbar(cm, cax=cax, label=label)


def format_time(ax, format_string="%H:%M:%S", label='Time (UTC)'):
    import matplotlib.dates as mdates
    ax.set_xticklabels(ax.xaxis.get_majorticklabels(), rotation=0, ha='center')
    ax.xaxis.set_major_formatter(mdates.DateFormatter(format_string))
    ax.set_xlabel(label)
    

def format_height(ax, scale=1.0e3, label='Height [km]'):
    import matplotlib.ticker as ticker
    ticks_y = ticker.FuncFormatter(lambda x, pos: '${0:g}$'.format(x/scale))
    ax.yaxis.set_major_formatter(ticks_y)
    ax.set_ylabel(label)
    

def format_latitude(ax):
    import matplotlib.ticker as ticker
    latFormatter = ticker.FuncFormatter(lambda x, pos: "${:g}^\circ$S".format(-1*x) if x < 0 else "${:g}^\circ$N".format(x))
    ax.xaxis.set_major_formatter(latFormatter)


import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize, ListedColormap, LinearSegmentedColormap, ColorConverter


def make_colormap(seq):
    """Return a LinearSegmentedColormap
    seq: a sequence of floats and RGB-tuples. The floats should be increasing
    and in the interval (0,1).
    """
    seq = [(None,) * 3, 0.0] + list(seq) + [1.0, (None,) * 3]
    cdict = {'red': [], 'green': [], 'blue': []}
    for i, item in enumerate(seq):
        if isinstance(item, float):
            r1, g1, b1 = seq[i - 1]
            r2, g2, b2 = seq[i + 1]
            cdict['red'].append([item, r1, r2])
            cdict['green'].append([item, g1, g2])
            cdict['blue'].append([item, b1, b2])
    return LinearSegmentedColormap('CustomMap', cdict)

def add_colorbar(ax, cm, label, on_left=False, horz_buffer=0.025, width_ratio="1.25%"):
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    
    if on_left:
        bbox_left = 0 - horz_buffer
    else:
        bbox_left = 1 + horz_buffer
       
    cax = inset_axes(ax,
                     width=width_ratio,  # percentage of parent_bbox width
                     height="100%",  # height : 50%
                     loc=3,
                     bbox_to_anchor=(bbox_left,0,1,1),
                     bbox_transform=ax.transAxes,
                     borderpad=0.0,
                     )       
    return plt.colorbar(cm, cax=cax, label=label)

     
ACTC_category_colors = [sns.xkcd_rgb['silver'],         #unknown
                            sns.xkcd_rgb['reddish brown'],         #surface and subsurface
                            sns.xkcd_rgb['white'],         #clear
                            sns.xkcd_rgb['dull red'],      #rain in clutter
                            sns.xkcd_rgb['off blue'],     #snow in clutter
                            sns.xkcd_rgb['dull yellow'],   #cloud in clutter
                            sns.xkcd_rgb['dark red'],      #heavy rain',
                            sns.xkcd_rgb["navy blue"],   #heavy mixed-phase precipitation
                            sns.xkcd_rgb['light grey'],    #clear (poss. liquid) 
                            sns.xkcd_rgb['pale yellow'],   #liquid cloud
                            sns.xkcd_rgb['golden'],        #drizzling liquid
                            sns.xkcd_rgb['orange'],        #warm rain
                            sns.xkcd_rgb['bright red'],    #cold rain
                            sns.xkcd_rgb['easter purple'], # melting snow
                            sns.xkcd_rgb['dark sky blue'],        # snow (possible liquid)
                            sns.xkcd_rgb['bright blue'], # snow
                            sns.xkcd_rgb["prussian blue"],   # rimed snow (poss. liquid)
                            sns.xkcd_rgb['dark teal'],   # rimed snow and SLW
                            sns.xkcd_rgb['teal'],              # snow and SLW
                            sns.xkcd_rgb['light green'],   # supercooled liquid
                            sns.xkcd_rgb["sky blue"],      # ice (poss. liquid)
                            sns.xkcd_rgb['bright teal'],   # ice and SLW
                            sns.xkcd_rgb['light blue'],    # ice (no liquid)
                            sns.xkcd_rgb['pale blue'],     # strat. ice
                            sns.xkcd_rgb['neon green'],    # PSC I
                            sns.xkcd_rgb['greenish cyan'], # PSC II
                            sns.xkcd_rgb['ugly green'],    # insects
                            sns.xkcd_rgb['sand'],          # dust
                            sns.xkcd_rgb['pastel pink'],   # sea salt
                            sns.xkcd_rgb['dust'],          # continental pollution
                            sns.xkcd_rgb['purpley grey'],  # smoke
                            sns.xkcd_rgb['dark lavender'], # dusty smoke
                            sns.xkcd_rgb['dusty lavender'],# dusty mix
                            sns.xkcd_rgb['slate grey'],     # stratospheric aerosol 1 (ash)
                            sns.xkcd_rgb['brownish purple'],     # stratospheric aerosol 2 (sulphate)
                            sns.xkcd_rgb['dark grey'],     # stratospheric aerosol 3 (smoke)]
                  ]

ATC_category_colors = [sns.xkcd_rgb['grey'],      #missing data
                       sns.xkcd_rgb['reddish brown'],       #surface and subsurface
                       sns.xkcd_rgb['light grey'],      #noise in both Mie and Ray channels
                       sns.xkcd_rgb['white'],       #clear
                       sns.xkcd_rgb['pale yellow'],   #liquid cloud
                       sns.xkcd_rgb['light green'], # supercooled liquid
                       sns.xkcd_rgb['light blue'],    # ice (no liquid)
                       sns.xkcd_rgb['sand'],          # dust
                       sns.xkcd_rgb['pastel pink'],   # sea salt
                       sns.xkcd_rgb['dust'],   # continental pollution
                       sns.xkcd_rgb['purpley grey'],  # smoke
                       sns.xkcd_rgb['dark lavender'], # dusty smoke
                       sns.xkcd_rgb['dusty lavender'],# dusty mix
                       sns.xkcd_rgb['pale blue'],      # strat. ice
                       sns.xkcd_rgb['neon green'],     # PSC I
                       sns.xkcd_rgb['greenish cyan'],     # PSC II
                       sns.xkcd_rgb['slate grey'],     # stratospheric aerosol 1 (ash)
                       sns.xkcd_rgb['brownish purple'],     # stratospheric aerosol 2 (sulphate)
                       sns.xkcd_rgb['dark grey'],     # stratospheric aerosol 3 (smoke)]
                       sns.xkcd_rgb['pastel pink'], #'101: Unknown: Aerosol Target has a very low probability (no class assigned)',
                       sns.xkcd_rgb['light lavender'], #'102: Unknown: Aerosol classification outside of param space',
                       sns.xkcd_rgb['dull purple'], #'104: Unknown: Strat. Aerosol Target has a very low probability (no class assigned)',
                       sns.xkcd_rgb['denim'], #'105: Unknown: Strat. Aerosol classification outside of param space',
                       sns.xkcd_rgb['neon yellow'], #'106: Unknown: PSC Target has a very low probability (no class assigned)',
                       sns.xkcd_rgb['light neon green']  #'107: Unknown: PSC classification outside of param space'
                      ]

CTC_category_colors = [sns.xkcd_rgb['grey'],      #missing data
                       sns.xkcd_rgb['reddish brown'],       #surface and subsurface
                       sns.xkcd_rgb['white'],       #clear
                       sns.xkcd_rgb['pale yellow'],  #liquid cloud
                       sns.xkcd_rgb['golden'], # drizzling liquid cloud
                       sns.xkcd_rgb['orange'], # warm rain
                       sns.xkcd_rgb['bright red'],    #cold rain
                       sns.xkcd_rgb['easter purple'], # melting snow
                       sns.xkcd_rgb["prussian blue"],   # rimed snow (poss. liquid)
                       sns.xkcd_rgb['bright blue'], # snow
                       sns.xkcd_rgb['light blue'],    # ice (no liquid)
                       sns.xkcd_rgb['ice blue'],      # strat. ice
                       sns.xkcd_rgb['ugly green'],    # insects
                       sns.xkcd_rgb['dark red'],      # heavy rain likely 
                       sns.xkcd_rgb["royal blue"],   # mixed-phase precip. likely
                       sns.xkcd_rgb['dark red'],      # heavy rain
                       sns.xkcd_rgb["navy blue"],   #heavy mixed-phase precipitation
                       sns.xkcd_rgb['dull red'],     # rain in clutter 
                       sns.xkcd_rgb['off blue'],     #snow in clutter
                       sns.xkcd_rgb['dull yellow'],    # cloud in clutter 
                       sns.xkcd_rgb['light grey'],    # clear (poss. liquid) 
                       sns.xkcd_rgb['silver'],        # unknown
                      ]
        
    
def plot_ECL2_target_classification(ax, ds, varname, category_colors, 
                                       hmax=15e3, label_fontsize='xx-small', 
                                       processor=None, title=None, title_prefix=None,
                                       savefig=False, dstdir="./", show_latlon=True,
                                   use_latitude=False):
    
    if processor is None:
        processor = "-".join([t.replace("_","") for t in ds.encoding['source'].split("/")[-1][9:16].split('_', maxsplit=1)])
        
    if title is None:
        long_name = ds[varname].attrs['long_name'].split(" ")
        #Removing capitalizations unless it's an acronym
        for i, l in enumerate(long_name):
            if not l.isupper():
                long_name[i] = l.lower()
        if title_prefix:
            long_name = [title_prefix.strip()] + long_name
        long_name = " ".join(long_name)
        title = f"{processor} {title_prefix}{long_name}"
    
    cleanup_category = lambda s: s.strip().replace('possible', 'poss.').replace('supercooled', "s\'cooled").replace("stratospheric", 'strat.').replace('extinguished', 'ext.').replace('precipitation', 'precip.').replace('and', '&').replace('unknown', 'unk.').replace('precipitation', 'precip.')
    
    if "\n" in ds[varname].attrs['definition']:
        categories = [cleanup_category(s) for s in ds[varname].attrs['definition'].split('\n')]
    else:
        #C-TC uses comma-separated "definition" attribute, but also uses commas within definitions.
        categories = [cleanup_category(s) for s in ds[varname].attrs['definition'].replace("ground clutter,", "ground clutter;").split(',')]
    
    import pandas as pd
    if ':' in categories[0]:
        first_c = int(categories[0].split(":")[0])
        last_c  = int(categories[-1].split(":")[0])
        u = np.array([int(c.split(':')[0]) for c in categories])
        categories_formatted = [f"${c.split(':')[0]}$:{c.split(':')[1]}" for c in categories]
    elif '=' in categories[0]:
        first_c = int(categories[0].split("=")[0])
        last_c  = int(categories[-1].split("=")[0])
        u = np.array([int(c.split('=')[0]) for c in categories])
        categories_formatted = [f"${c.split('=')[0]}$:{c.split('=')[1]}" for c in categories]
    else:
        print("category values are not included within categories")
    
    bounds = np.concatenate(([u.min()-1], u[:-1]+np.diff(u)/2. ,[u.max()+1]))
    
    from matplotlib.colors import ListedColormap, BoundaryNorm
    norm = BoundaryNorm(bounds, len(bounds)-1)
    cmap = ListedColormap(sns.color_palette(category_colors[:len(u)]).as_hex())
    
    if use_latitude:
        _l, _h, _z = xr.broadcast(ds.latitude, ds.height, ds[varname])
        if (np.isnan(_h).sum() > 0):
            _cm = ax.pcolor(_l, _h, _z, norm=norm, cmap=cmap)
        else:
            _cm = ax.pcolormesh(_l, _h, _z, norm=norm, cmap=cmap)
    else:
        
        _t, _h, _z = xr.broadcast(ds.time, ds.height, ds[varname])
        if (np.isnan(_h).sum() > 0):
            _cm = ax.pcolor(_t, _h, _z, norm=norm, cmap=cmap)
        else:
            _cm = ax.pcolormesh(_t, _h, _z, norm=norm, cmap=cmap)
    
    _cb = add_colorbar(ax, _cm, '', horz_buffer=0.01)
    _cb.set_ticks(bounds[:-1]+np.diff(bounds)/2.)
    _cb.ax.set_yticklabels(categories_formatted, fontsize=label_fontsize)
    
    format_plot(ax, ds, title, hmax, use_latitude=use_latitude)
    
    if savefig:
        import os
        dstfile = f"{product_code}_{varname}.png"
        fig.savefig(os.path.join(dstdir,dstfile), bbox_inches='tight')
        
    
    
AAER_aerosol_classes = [10,11,12,13,14,15]
CCLD_ice_classes = [7,8,9,13,15,17]
CCLD_rain_classes = [3,4,5,12,14,16]


def format_plot(ax, ds, title, hmax, use_latitude=False):
    
    #Set title
    ax.set_title(title)
    
    ax.set_ylim(-500,hmax)
    format_height(ax)

    if use_latitude:
        #Trim to frame
        get_frame_edge = lambda n: min([67.5,22.5,-22.5,-67.5], key=lambda x:abs(x-n))

        #Lat/lon ticks snap to frame boundaries, then intervals of 5deg latitude
        lat_ticks = np.linspace(get_frame_edge(ds.latitude[0]), get_frame_edge(ds.latitude[-1]), 1*9+1) #37)
        ax.set_xticks(lat_ticks)
        
        #lat_ticks_minor = np.arange(get_frame_edge(ds.latitude[0]), get_frame_edge(ds.latitude[-1]), -0.25)
        lat_ticks_minor = np.linspace(get_frame_edge(ds.latitude[0]), get_frame_edge(ds.latitude[-1]), 4*9+1) #37)
        ax.set_xticks(lat_ticks_minor, minor=True)

        #Nice formatting for coordinates
        lon_ticks = ds.set_coords(['latitude']).longitude.swap_dims({'along_track':'latitude'}).sel(latitude=lat_ticks, method='nearest').values
        latlon_ticks = ["($%.1f^{\circ}$N, $%.1f^{\circ}$E)" %ll for ll in zip(lat_ticks, lon_ticks)]
        ax.set_xticklabels(latlon_ticks, fontsize='small', color='k')
    
    else:
        
        #Major ticks at 3-minute intervals
        time_ticks = pd.date_range(ds.time.to_index().round('60S')[0], ds.time.to_index().round('60S')[-1], freq='60S')
        ax.set_xticks(time_ticks)
        
        #Minor ticks at 1-minute intervals
        time_ticks_minor = pd.date_range(ds.time.to_index().round('60S')[0], ds.time.to_index().round('60S')[-1], freq='15S')
        ax.set_xticks(time_ticks_minor, minor=True)

        #Nice formatting for time
        format_time(ax, format_string="%H:%M:%S", label=f"Time (UTC) {ds.time[0].dt.year.values}-{ds.time[0].dt.month.values}-{ds.time[0].dt.day.values}")

        #Complement time axis ticks with lat/lon information
        _ax = ax.twiny()
        _ax.set_xlim(ax.get_xlim())

        #Trim to frame
        get_frame_edge = lambda n: min([67.5,22.5,-22.5,-67.5], key=lambda x:abs(x-n))

        #Lat/lon ticks snap to frame boundaries, then intervals of 5deg latitude
        lat_ticks = np.linspace(get_frame_edge(ds.latitude[0]), get_frame_edge(ds.latitude[-1]), 1*9+1)
        time_ticks = ds.set_coords(['latitude']).time.swap_dims({'along_track':'latitude'}).sel(latitude=lat_ticks, method='nearest').values
        _ax.set_xticks(time_ticks)
        
        #Lat/lon ticks snap to frame boundaries, then intervals of 5deg latitude
        lat_ticks_minor = np.linspace(get_frame_edge(ds.latitude[0]), get_frame_edge(ds.latitude[-1]), 5*9+1)
        time_ticks_minor = ds.set_coords(['latitude']).time.swap_dims({'along_track':'latitude'}).sel(latitude=lat_ticks_minor, method='nearest').values
        _ax.set_xticks(time_ticks_minor, minor=True)

        #Nice formatting for coordinates
        lon_ticks = ds.set_coords(['latitude']).longitude.swap_dims({'along_track':'latitude'}).sel(latitude=lat_ticks, method='nearest').values
        latlon_ticks = ["($%.1f^{\circ}$N, $%.1f^{\circ}$E)" %ll for ll in zip(lat_ticks, lon_ticks)]
        _ax.set_xticklabels(latlon_ticks, fontsize='xx-small', color='0.5')
        _ax.tick_params(axis='x', which='both', color='0.5')

        ax.set_xlim([time_ticks[0], time_ticks[-1]])
        _ax.set_xlim([time_ticks[0], time_ticks[-1]])
        
    #Specify the product filename
    product_code = ds.encoding['source'].split('/')[-1].split('.')[0]
    ax.text(0.9975,0.98, product_code, 
            ha='right', va='top', 
            fontsize='xx-small', color='0.5', transform=ax.transAxes)
    
    #Add orbit alongside title
    orbit = ds.encoding['source'].split("/")[-1].split(".")[0].split("_")[-1]
    ax.text(1, -0.215, f"EarthCARE orbit {orbit}", fontsize='small', 
            ha='right', va='bottom', color='0.33', transform=ax.transAxes)
    
    
    
def plot_ECL2(ax, ds, varname, label, 
              plot_where=True, scale_factor=1,
             hmax=15e3, plot_scale=None, plot_range=None, cmap=None,
             units=None, processor=None, title=None, title_prefix="",
             heightvar='height'):
    

    sns.set_style('ticks')
    sns.set_context('poster')
    
    import pandas as pd
    
    if plot_scale is None:
        plot_scale = ds[varname].attrs['plot_scale']
    
    if plot_range is None:
        plot_range = ds[varname].attrs['plot_range']
        
    if plot_scale == 'logarithmic':
        norm=LogNorm(plot_range[0], plot_range[-1])
    else:
        norm=Normalize(plot_range[0], plot_range[-1])
    
    if cmap is None:
        cmap=chiljet2()
    
    if processor is None:
        processor = "-".join([t.replace("_","") for t in ds.encoding['source'].split("/")[-1][9:16].split('_', maxsplit=1)])
    
    if units is None:
        units = ds[varname].attrs['units']
    
    if title is None:
        long_name = ds[varname].attrs['long_name'].split(" ")
        #Removing capitalizations unless it's an acronym
        for i, l in enumerate(long_name):
            if not l.isupper():
                long_name[i] = l.lower()
        if title_prefix:
            long_name = [title_prefix.strip()] + long_name
        long_name = " ".join(long_name)
    
        title = f"{processor} {long_name}"
        
    _t, _h, _z = xr.broadcast(ds.time, ds[heightvar].fillna(0.), scale_factor*ds[varname].fillna(0.))
    _cm = ax.pcolormesh(_t, _h, _z.where(plot_where), 
                        norm=norm, cmap=cmap)
    add_colorbar(ax, _cm, f"{label} [{units}]", horz_buffer=0.02)
    
    format_plot(ax, ds, title, hmax)
    
    
def add_subfigure_labels(axes, xloc=0.0, yloc=1.125, zorder=0, fontsize='medium',
                         label_list=[], flatten_order='F'):
    if label_list == []:
        import string
        labels = string.ascii_lowercase
    else:
        labels = label_list
        
    for i, ax in enumerate(axes.flatten(order=flatten_order)):
        if ax:
            ax.text(xloc, yloc, "%s)" %(labels[i]), va='baseline', fontsize=fontsize,
                    transform=ax.transAxes, fontweight='bold', zorder=zorder)


def add_surface(ax, ds, 
                elevation_var='surface_elevation', 
                land_var='land_flag'):

    ax.axhspan(-500,0,lw=0, color=sns.xkcd_rgb['sky blue'], zorder=-1)
    
    ax.fill_between(ds.time[:], ds[elevation_var][:], y2=-500,
                    lw=0, color=sns.xkcd_rgb['sky blue'], step='mid')
    
    ax.fill_between(ds.time[:], ds[elevation_var].where(ds[land_var]==1)[:], y2=-500,
                    lw=0, color=sns.xkcd_rgb['pale brown'], step='mid', zorder=1)

    
def add_ruler(ax, ds, dx=500, d0=100, x0=100, hmax=18e3):
    h0 = 0.9*hmax
    buffer = 0.015*hmax
    ax.plot(ds.time[[x0,x0+dx]], [h0,h0], color='k', lw=5, solid_capstyle='butt')
    rticks = np.arange(x0,x0+dx+1,d0)
    nticks = len(rticks)
    
    ax.plot(ds.time[rticks], nticks*[h0], color='k', lw=0, marker='|', markersize=6)
    
    shade_around_text(ax.text(ds.time[x0+dx//2], h0+buffer, f"scale [km]", fontsize='xx-small', 
                              va='bottom', ha='center', color='0.33'), alpha=0.5, fg='w')
    for i,rtick in enumerate(rticks):
        shade_around_text(ax.text(ds.time[rtick], h0-buffer, f"{rtick-x0}", 
                                  fontsize=10, va='top', ha='center', color='0.33'), 
                          alpha=0.5, fg='w')
        if (i%2 == 1) & (i < nticks-1):
            ax.plot(ds.time[[rtick,rtick+d0]], [h0,h0], color='0.95', lw=5, solid_capstyle='butt')

            
def shade_around_text(t, alpha=0.2, lw=2.5, fg='k'):
    import matplotlib.patheffects as PathEffects
    return t.set_path_effects([PathEffects.withStroke(linewidth=lw, foreground=fg, alpha=alpha)])


def add_temperature(ax, ds):
    
    _x, _y, _t = xr.broadcast(ds.time, ds.height_level, ds.temperature_level - 273.15)
    _cn = ax.contour(_x, _y, _t, levels=np.arange(-90,31,10),
                        colors='k', 
                        linewidths=[0.1, 0.1, 0.1, 0.33, 0.33, 1.0, 0.5, 1.0, 0.5, 2.0, 0.5, 1.0, 0.5], zorder=10)
    _cl = plt.clabel(_cn, [l for l in [-40,0] if l in _cn.levels], 
               inline=1, fmt='$%.0f^{\circ}$C', fontsize='xx-small', zorder=11)
    
    for t in _cl:
        t = shade_around_text(t, fg='w', alpha=0.5, lw=5)
        
    for l in _cn.labelTexts:
        l.set_rotation(0)
    return _cl


def add_humidity(ax, ds):
    
    _x, _y, _q = xr.broadcast(ds.time, ds.height_layer, ds.specific_humidity_layer_mean)
    contour_levels = [1e-5,1e-4,1e-3,1e-2]
    _cn = ax.contour(_x, _y, _q, 
                     levels=contour_levels, cmap='Blues', norm=LogNorm(1e-6,1e-2), 
                     linewidths=[3,3,3,3], alpha=0.67, zorder=10)
    _cl = plt.clabel(_cn, inline=1, fmt='%.1e', inline_spacing=15, colors='k',
                     fontsize='10', zorder=10)
    
    for i,t in enumerate(_cl):
        s = f"{int(np.log10(float(t.get_text()))):g}"
        t.set_text("$10^{" + s + "}$ kg/kg")
        t = shade_around_text(t, fg='w', alpha=0.5, lw=2.)
        
    for l in _cn.labelTexts:
        l.set_rotation(0)
    return _cl


def snap_xlims(axes):
    
    xlim = list(axes[0].get_xlim())
    for ax in axes[1:]:
        _xlim = list(ax.get_xlim())
        if _xlim[0] < xlim[0]:
            xlim[0] = _xlim[0]
        if _xlim[1] > _xlim[1]:
            xlim[1] = _xlim[1]
        ax.set_xlim(xlim[0], xlim[1])
    axes[0].set_xlim(xlim[0], xlim[1])

    
def add_extras(ax, CCLD, ACMCOM, hmax, 
               show_surface=True, show_ruler=True,
               show_humidity=True, show_temperature=True):
    
    if show_ruler:
        add_ruler(ax, CCLD, dx=500, d0=100, x0=100, hmax=hmax)
    
    if show_surface:
        add_surface(ax, CCLD)
    
    if show_temperature:
        add_temperature(ax, ACMCOM)
        
    if show_humidity:
        add_humidity(ax, ACMCOM)
    
    
def intercompare_target_classification(ATC, CTC, ACTC, ACMCOM, hmax=19e3, dstdir=None):
    
    nrows=5

    fig, axes = plt.subplots(figsize=(25,7*nrows), nrows=nrows, 
                             sharex=False, sharey=False, gridspec_kw={'hspace':0.5})

    plot_ECL2_target_classification(axes[0], ATC, 'classification_low_resolution', 
                                    ATC_category_colors, hmax=hmax, title_prefix="", label_fontsize=10)

    plot_ECL2_target_classification(axes[1], ACTC, 'ATLID_target_classification_low_resolution', 
                                    ATC_category_colors, hmax=hmax, title_prefix="", label_fontsize=10)

    plot_ECL2_target_classification(axes[2], CTC, 'hydrometeor_classification', 
                                    CTC_category_colors, hmax=hmax, title_prefix="", label_fontsize=10)

    plot_ECL2_target_classification(axes[3], ACTC, 'CPR_target_classification', 
                                    CTC_category_colors, hmax=hmax, title_prefix="", label_fontsize=10)

    plot_ECL2_target_classification(axes[4], ACTC, 'synergetic_target_classification_low_resolution', 
                                    ACTC_category_colors, hmax=hmax, title_prefix="", label_fontsize=10)

    for ax in axes:
        add_extras(ax, CTC, ACMCOM, hmax, show_surface=False, show_humidity=False)

    add_subfigure_labels(axes)
    snap_xlims(axes)
    
    
def intercompare_ice_water_content(AICE, CCLD, ACMCOM, ACMCAP,
                                  dstdir=None, hmax=19e3):

    cmap=colormaps.chiljet2()
    units = 'kg$~$m$^{-3}$'

    nrows=4
    fig, axes = plt.subplots(figsize=(25,7*nrows), nrows=nrows, 
                             sharex=False, sharey=False, gridspec_kw={'hspace':0.5})

    for ax in axes:
        add_extras(ax, CCLD, ACMCOM, hmax)

    i=0
    plot_ECL2(axes[i], AICE, 'ice_water_content', "IWC", scale_factor=1e-6,
              cmap=cmap, plot_scale=ACMCAP.ice_water_content.attrs['plot_scale'], 
              hmax=hmax, units=units,
              plot_range=ACMCAP.ice_water_content.attrs['plot_range'])

    i+=1
    plot_ECL2(axes[i], CCLD, 'water_content', "IWC", 
              plot_where=CCLD.hydrometeor_classification.isin(CCLD_ice_classes), 
              cmap=cmap, plot_scale=ACMCAP.ice_water_content.attrs['plot_scale'], 
              hmax=hmax, units=units,
              plot_range=ACMCAP.ice_water_content.attrs['plot_range'])

    i+=1
    plot_ECL2(axes[i], ACMCOM, 'ice_water_content', "IWC", 
              cmap=cmap, plot_scale=ACMCAP.ice_water_content.attrs['plot_scale'], 
              hmax=hmax, units=units,heightvar='height_layer',
              plot_range=ACMCAP.ice_water_content.attrs['plot_range'])

    i+=1
    plot_ECL2(axes[i], ACMCAP, 'ice_water_content', "IWC", 
              cmap=cmap, plot_scale=ACMCAP.ice_water_content.attrs['plot_scale'], 
              hmax=hmax, units=units,
              plot_range=ACMCAP.ice_water_content.attrs['plot_range'])

    add_subfigure_labels(axes)
    snap_xlims(axes)


def intercompare_snow_rate(CCLD, ACMCAP, hmax=19e3):
    label = "S"
    cmap=colormaps.chiljet2()
    units = 'mm$~$h$^{-1}$'
    plot_scale=ACMCAP.ice_mass_flux.attrs['plot_scale']
    plot_range=3600.*ACMCAP.ice_mass_flux.attrs['plot_range']

    nrows=2
    fig, axes = plt.subplots(figsize=(25,7*nrows), nrows=nrows, 
                             sharex=False, sharey=False, gridspec_kw={'hspace':0.5})

    for ax in axes:
        add_extras(ax, CCLD, ACMCOM, hmax)

    i=0
    plot_ECL2(axes[i], CCLD, 'mass_flux', label, scale_factor=3600.,
              plot_where=CCLD.hydrometeor_classification.isin(CCLD_ice_classes), 
              cmap=cmap, plot_scale=plot_scale, title_prefix="ice ",
              hmax=hmax, units=units,
              plot_range=plot_range)

    i+=1
    plot_ECL2(axes[i], ACMCAP, 'ice_mass_flux', label, scale_factor=3600.,
              cmap=cmap, plot_scale=plot_scale, 
              hmax=hmax, units=units,
              plot_range=plot_range)

    add_subfigure_labels(axes)
    snap_xlims(axes)

    
def intercompare_ice_effective_radius(AICE, ACMCOM, ACMCAP, CCLD,
                                      hmax=19e3):
    
    cmap=colormaps.chiljet2()
    label = "$r_{eff}$"
    units = "$\mu$m"
    plot_scale=ACMCAP.ice_effective_radius.attrs['plot_scale']
    plot_range=1e6*ACMCAP.ice_effective_radius.attrs['plot_range']

    nrows=3
    fig, axes = plt.subplots(figsize=(25,7*nrows), nrows=nrows, 
                             sharex=False, sharey=False, gridspec_kw={'hspace':0.5})

    for ax in axes:
        add_extras(ax, CCLD, ACMCOM, hmax)

    i=0
    plot_ECL2(axes[i], AICE, 'ice_effective_radius', label,
              plot_where=AICE.ice_water_content > 0,
              cmap=cmap, plot_scale=plot_scale, 
              hmax=hmax, units=units,
              plot_range=plot_range)

    i+=1
    plot_ECL2(axes[i], ACMCOM, 'ice_effective_radius', label,
              plot_where=ACMCOM.ice_water_content > 0, scale_factor=1e6,
              cmap=cmap, plot_scale=plot_scale, 
              hmax=hmax, units=units, heightvar='height_layer',
              plot_range=plot_range)

    i+=1
    plot_ECL2(axes[i], ACMCAP, 'ice_effective_radius', label,
              plot_where=ACMCAP.ice_water_content > 0, scale_factor=1e6,
              cmap=cmap, plot_scale=plot_scale, 
              hmax=hmax, units=units,
              plot_range=plot_range)

    add_subfigure_labels(axes)
    snap_xlims(axes)

    
def intercompare_rain_water_content(CCLD, ACMCOM, ACMCAP,
                                    hmax=19e3):

    cmap=colormaps.chiljet2()
    label="RWC"
    units = 'kg$~$m$^{-3}$'
    
    plot_scale=ACMCAP.rain_water_content.attrs['plot_scale']
    plot_range=ACMCAP.rain_water_content.attrs['plot_range']

    nrows=3
    fig, axes = plt.subplots(figsize=(25,7*nrows), nrows=nrows, 
                             sharex=False, sharey=False, gridspec_kw={'hspace':0.5})

    for ax in axes:
        add_extras(ax, CCLD, ACMCOM, hmax)

    i=0
    plot_ECL2(axes[i], CCLD, 'water_content', label,
              plot_where=CCLD.hydrometeor_classification.isin(CCLD_rain_classes), 
              cmap=cmap, plot_scale=plot_scale, 
              hmax=hmax, units=units, title_prefix="rain ",
              plot_range=plot_range)

    i+=1
    plot_ECL2(axes[i], ACMCOM, 'rain_water_content', label,
              cmap=cmap, plot_scale=plot_scale, 
              hmax=hmax, units=units,heightvar='height_layer',
              plot_range=plot_range)

    i+=1
    plot_ECL2(axes[i], ACMCAP, 'rain_water_content', label,
              cmap=cmap, plot_scale=plot_scale, 
              hmax=hmax, units=units,
              plot_range=plot_range)

    add_subfigure_labels(axes)
    snap_xlims(axes)

    
def intercompare_rain_rate(CCLD, ACMCAP, ACMCOM, hmax=19e3):
    cmap=colormaps.chiljet2()
    units = 'mm$~$h$^{-1}$'
    plot_scale=ACMCAP.rain_rate.attrs['plot_scale']
    plot_range=3600.*ACMCAP.rain_rate.attrs['plot_range']

    nrows=2
    fig, axes = plt.subplots(figsize=(25,7*nrows), nrows=nrows, 
                             sharex=False, sharey=False, gridspec_kw={'hspace':0.5})

    for ax in axes:
        add_extras(ax, CCLD, ACMCOM, hmax)

    i=0
    plot_ECL2(axes[i], CCLD, 'mass_flux', "R", scale_factor=3600.,
              plot_where=CCLD.hydrometeor_classification.isin(CCLD_rain_classes), 
              cmap=cmap, plot_scale=plot_scale, 
              hmax=hmax, units=units, title_prefix="rain ",
              plot_range=plot_range)
    
    i+=1
    plot_ECL2(axes[i], ACMCAP, 'rain_rate', "R", scale_factor=3600.,
              cmap=cmap, plot_scale=plot_scale, 
              hmax=hmax, units=units,
              plot_range=plot_range)

    add_subfigure_labels(axes)
    snap_xlims(axes)

    
def intercompare_liquid_water_content(CCLD, ACMCOM, ACMCAP):

    cmap=colormaps.chiljet2()
    label="LWC"
    units = 'kg$~$m$^{-3}$'
    hmax=19e3
    
    plot_range=ACMCAP.liquid_water_content.attrs['plot_range']
    plot_scale=ACMCAP.liquid_water_content.attrs['plot_scale']

    nrows=3
    fig, axes = plt.subplots(figsize=(25,7*nrows), nrows=nrows, 
                             sharex=False, sharey=False, gridspec_kw={'hspace':0.5})

    for ax in axes:
        add_extras(ax, CCLD, ACMCOM, hmax)

    i=0
    plot_ECL2(axes[i], CCLD, 'liquid_water_content', label,
              cmap=cmap, plot_scale=plot_scale,
              hmax=hmax, units=units, 
              plot_range=plot_range)

    i+=1
    plot_ECL2(axes[i], ACMCOM, 'liquid_water_content', label,
              cmap=cmap, plot_scale=plot_scale,
              hmax=hmax, units=units,heightvar='height_layer',
              plot_range=plot_range)

    i+=1
    plot_ECL2(axes[i], ACMCAP, 'liquid_water_content', label,
              cmap=cmap, plot_scale=plot_scale,
              hmax=hmax, units=units,
              plot_range=plot_range)

    add_subfigure_labels(axes)
    snap_xlims(axes)

    
def intercompare_liquid_effective_radius(CCLD, ACMCOM, ACMCAP):
    
    cmap=colormaps.chiljet2()
    label = "$r_{eff}$"
    units = "$\mu$m"
    hmax=19e3
    plot_scale=ACMCAP.liquid_effective_radius.attrs['plot_scale']
    plot_range=[0,20]

    nrows=2
    fig, axes = plt.subplots(figsize=(25,7*nrows), nrows=nrows, 
                             sharex=False, sharey=False, gridspec_kw={'hspace':0.5})

    for ax in axes:
        add_extras(ax, CCLD, ACMCOM, hmax)

    i=0
    plot_ECL2(axes[i], ACMCOM, 'liquid_effective_radius', label,
              plot_where=ACMCOM.liquid_water_content > 0, scale_factor=1e6,
              cmap=cmap, plot_scale=plot_scale, 
              hmax=hmax, units=units, heightvar='height_layer',
              plot_range=plot_range)

    i+=1
    plot_ECL2(axes[i], ACMCAP, 'liquid_effective_radius', label,
              plot_where=ACMCAP.liquid_water_content > 0, scale_factor=1e6,
              cmap=cmap, plot_scale=plot_scale, 
              hmax=hmax, units=units,
              plot_range=plot_range)

    add_subfigure_labels(axes)
    snap_xlims(axes)

    
def intercompare_aerosol_extinction(AAER, AEBD, ACMCOM, ACMCAP, CCLD):
    cmap=colormaps.chiljet2()
    label=r"$\alpha$"
    units = 'm$^{-1}$'
    hmax=19e3
    plot_scale=ACMCAP.aerosol_extinction.attrs['plot_scale']
    plot_range=[1e-6,1e-3]

    nrows=4
    fig, axes = plt.subplots(figsize=(25,7*nrows), nrows=nrows, 
                             sharex=False, sharey=False, gridspec_kw={'hspace':0.5})

    for ax in axes:
        add_extras(ax, CCLD, ACMCOM, hmax)
        
    i=0
    plot_ECL2(axes[i], AAER, 'particle_extinction_coefficient_355nm', label,
              plot_where=AAER.classification.isin(AAER_aerosol_classes),
              cmap=cmap, plot_scale=plot_scale, 
              hmax=hmax, units=units,
              plot_range=plot_range)

    i+=1
    plot_ECL2(axes[i], AEBD, 'particle_extinction_coefficient_355nm', label,
                plot_where=AEBD.simple_classification.isin([3]),
              cmap=cmap, plot_scale=plot_scale, 
              hmax=hmax, units=units,
              plot_range=plot_range)

    i+=1
    plot_ECL2(axes[i], ACMCOM, 'aerosol_extinction', label,
              cmap=cmap, plot_scale=plot_scale, 
              hmax=hmax, units=units, 
              heightvar='height_layer',
              plot_range=plot_range)

    i+=1
    plot_ECL2(axes[i], ACMCAP, 'aerosol_extinction', label,
              cmap=cmap, plot_scale=plot_scale, 
              hmax=hmax, units=units,
              plot_range=plot_range)

    add_subfigure_labels(axes)
    snap_xlims(axes)
    

def intercompare_lidar_ratio(AAER, AEBD, ACMCAP, CCLD):
    cmap=colormaps.chiljet2()
    label=r"$s$"
    units = 'sr'
    hmax=19e3
    plot_scale=ACMCAP.ATLID_bscat_extinction_ratio.attrs['plot_scale']
    plot_range=[0,100]

    #Inverting bscat-extinction ratio to get lidar ratio
    ACMCAP['ATLID_lidar_ratio'] = (1/ACMCAP.ATLID_bscat_extinction_ratio).where(ACMCAP.ATLID_bscat_extinction_ratio > 1e-9)
    ACMCAP['ATLID_lidar_ratio'].attrs['long_name'] = "forward-modelled ATLID extinction to backscatter ratio"
    
    nrows=3
    fig, axes = plt.subplots(figsize=(25,7*nrows), nrows=nrows, 
                             sharex=False, sharey=False, gridspec_kw={'hspace':0.5})

    for ax in axes:
        add_extras(ax, CCLD, ACMCOM, hmax)

    i=0
    plot_ECL2(axes[i], AAER, 'lidar_ratio_355nm', label,
              plot_where=AAER.classification > 0,
              #plot_where=AAER.classification.isin([10,11,12,13,14,15]),
              cmap=cmap, plot_scale=plot_scale, 
              hmax=hmax, units=units,
              plot_range=plot_range)

    i+=1
    plot_ECL2(axes[i], AEBD, 'lidar_ratio_355nm', label,
              plot_where=AEBD.simple_classification > 0,
              #plot_where=AEBD.simple_classification.isin([3]),
              cmap=cmap, plot_scale=plot_scale, 
              hmax=hmax, units=units,
              plot_range=plot_range)

    i+=1
    plot_ECL2(axes[i], ACMCAP, 'ATLID_lidar_ratio', label,
              plot_where=ACMCAP.ATLID_lidar_ratio > 0,
              cmap=cmap, plot_scale=plot_scale, 
              hmax=hmax, units=units,
              plot_range=plot_range)

    add_subfigure_labels(axes)
    snap_xlims(axes)

    
def intercompare_rain_median_diameter(CCLD, ACMCAP, ACMCOM):
    
    cmap=colormaps.chiljet2()
    label = "$D_{0}$"
    units = "m"
    hmax=19e3
    plot_scale=ACMCAP.rain_median_volume_diameter.attrs['plot_scale']
    plot_range=[1e-5,2e-3]

    nrows=2
    fig, axes = plt.subplots(figsize=(25,7*nrows), nrows=nrows, 
                             sharex=False, sharey=False, gridspec_kw={'hspace':0.5})

    for ax in axes:
        add_extras(ax, CCLD, ACMCOM, hmax)

    i=0
    plot_ECL2(axes[i], CCLD, 'characteristic_diameter', label,
              plot_where=CCLD.hydrometeor_classification.isin(CCLD_rain_classes),
              cmap=cmap, plot_scale=plot_scale, 
              hmax=hmax, units=units, 
              plot_range=plot_range)

    i+=1
    plot_ECL2(axes[i], ACMCAP, 'rain_median_volume_diameter', label,
              plot_where=ACMCAP.rain_water_content > 0, 
              cmap=cmap, plot_scale=plot_scale, 
              hmax=hmax, units=units,
              plot_range=plot_range)

    add_subfigure_labels(axes)
    snap_xlims(axes)

    
def intercompare_ice_median_diameter(CCLD, ACMCAP, ACMCOM):
    
    cmap=colormaps.chiljet2()
    label = "$D_{0}$"
    units = "m"
    hmax=19e3
    plot_scale=ACMCAP.ice_median_volume_diameter.attrs['plot_scale']
    plot_range=[1e-5,2e-3]

    nrows=2
    fig, axes = plt.subplots(figsize=(25,7*nrows), nrows=nrows, 
                             sharex=False, sharey=False, gridspec_kw={'hspace':0.5})

    for ax in axes:
        add_extras(ax, CCLD, ACMCOM, hmax)

    i=0
    plot_ECL2(axes[i], CCLD, 'characteristic_diameter', label,
              plot_where=CCLD.hydrometeor_classification.isin(CCLD_ice_classes),
              cmap=cmap, plot_scale=plot_scale, 
              hmax=hmax, units=units, 
              plot_range=plot_range)

    i+=1
    plot_ECL2(axes[i], ACMCAP, 'ice_median_volume_diameter', label,
              plot_where=ACMCAP.ice_water_content > 0, 
              cmap=cmap, plot_scale=plot_scale, 
              hmax=hmax, units=units,
              plot_range=plot_range)

    add_subfigure_labels(axes)
    snap_xlims(axes)
    

    
    
def plot_ACMCAP_ice(ACMCAP, ACMCOM, CCLD):
    
    cmap=colormaps.chiljet2()
    hmax=19e3

    nrows=6
    fig, axes = plt.subplots(figsize=(25,7*nrows), nrows=nrows, 
                             sharex=False, sharey=False, gridspec_kw={'hspace':0.5})
    
    for ax in axes:
        add_extras(ax, CCLD, ACMCOM, hmax)
    
    i=0    
    var = 'ice_water_content'
    label = "IWC"
    units = "kg$~$m$^{-1}$"
    plot_scale=ACMCAP[var].attrs['plot_scale']
    plot_range=ACMCAP[var].attrs['plot_range']
    plot_ECL2(axes[i], ACMCAP, var, label, 
              cmap=cmap, plot_scale=plot_scale, 
              hmax=hmax, units=units,
              plot_range=plot_range)
    
    i+=1    
    var = 'ice_mass_flux'
    label = "S"
    units = "mm$~$h$^{-1}$"
    plot_scale=ACMCAP[var].attrs['plot_scale']
    plot_range=[1e-4,2e1]
    plot_ECL2(axes[i], ACMCAP, var, label, scale_factor=3600.,
              cmap=cmap, plot_scale=plot_scale, 
              hmax=hmax, units=units,
              plot_range=plot_range)
    
    i+=1
    var = 'ice_effective_radius'
    label = "r$_{eff}$"
    units = "$\mu$m"
    plot_scale=ACMCAP[var].attrs['plot_scale']
    plot_range=[0,120]
    plot_ECL2(axes[i], ACMCAP, var, label,
              plot_where=ACMCAP.ice_water_content > 0, scale_factor=1e6,
              cmap=cmap, plot_scale=plot_scale, 
              hmax=hmax, units=units,
              plot_range=plot_range)
    
    i+=1
    var = 'ice_median_volume_diameter'
    label = "$D_{0}$"
    units = ACMCAP[var].attrs['units']
    plot_scale=ACMCAP[var].attrs['plot_scale']
    plot_range=[1e-5,1e-2]
    plot_ECL2(axes[i], ACMCAP, var, label,
              plot_where=ACMCAP.ice_water_content > 0, 
              cmap=cmap, plot_scale=plot_scale, 
              hmax=hmax, units=units,
              plot_range=plot_range)
    
    i+=1
    var = 'ice_lidar_bscat_extinction_ratio'
    label = "$s$"
    units = ACMCAP[var].attrs['units']
    plot_scale=ACMCAP[var].attrs['plot_scale']
    plot_range=ACMCAP[var].attrs['plot_range']
    plot_ECL2(axes[i], ACMCAP, var, label,
              plot_where=ACMCAP.ice_water_content > 0, 
              cmap=cmap, plot_scale=plot_scale, 
              hmax=hmax, units=units,
              plot_range=plot_range)
    
    i+=1
    var = 'ice_riming_factor'
    label = "r"
    units = ACMCAP[var].attrs['units']
    plot_scale=ACMCAP[var].attrs['plot_scale']
    plot_range=ACMCAP[var].attrs['plot_range']
    plot_ECL2(axes[i], ACMCAP, var, label,
              plot_where=ACMCAP.ice_water_content > 0, 
              cmap=cmap, plot_scale=plot_scale, 
              hmax=hmax, units=units,
              plot_range=plot_range)
    
    add_subfigure_labels(axes)
    snap_xlims(axes)
    
    