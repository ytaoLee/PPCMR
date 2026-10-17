"""Measured four-size scalability grid; all three query observations retained."""
import argparse
import csv
from pathlib import Path
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator,FuncFormatter,NullLocator

ROOT=Path(__file__).resolve().parent.parent
OUT=ROOT/'figures'
METHODS=('fresh_mpc','cached_mpc','cached_local_limb')
COLORS={'fresh_mpc':'#778690','cached_mpc':'#287BA7','cached_local_limb':'#B78651'}
NAMES={'fresh_mpc':'Fresh MPC','cached_mpc':'Cached MPC','cached_local_limb':'Cached local'}
SIZES=(1000,5000,10000,20000)
mpl.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','DejaVu Sans'],'font.size':7.5,
    'axes.labelsize':7.5,'axes.titlesize':7.5,'xtick.labelsize':7.5,'ytick.labelsize':7.5,'legend.fontsize':7.5,
    'axes.spines.top':True,'axes.spines.right':True,'axes.linewidth':.65,'axes.edgecolor':'#788A93',
    'axes.labelcolor':'#274351','text.color':'#274351','xtick.color':'#274351','ytick.color':'#274351',
    'xtick.major.size':2.5,'ytick.major.size':2.5,'xtick.major.width':.6,'ytick.major.width':.6,
    'pdf.fonttype':42,'svg.fonttype':'none','legend.frameon':False,'savefig.facecolor':'white'})


def rows(name):
    with (DATA/name).open(encoding='utf-8',newline='') as stream:return list(csv.DictReader(stream))


def positive(values):
    result=np.asarray(values,dtype=float)
    assert np.isfinite(result).all() and np.all(result>0)
    return result


def panel(ax,label,title):
    ax.text(0,1.10,label,transform=ax.transAxes,fontsize=8,fontweight='bold',va='bottom')
    ax.text(.14,1.10,title,transform=ax.transAxes,fontsize=7.5,va='bottom')
    ax.tick_params(which='both',pad=2,top=True,right=True,direction='in')
    ax.set_axisbelow(True);ax.grid(axis='both',which='both',color='#DDE5E9',linewidth=.45,alpha=.65)
    ax.set_xlim(.3,21.7);ax.set_xticks((1,5,10,20));ax.set_xlabel('Images (thousands)',labelpad=2)


def main():
    observed=rows('scale_observations.csv');summary=rows('scale_summary.csv');setup=rows('scale_setup.csv')
    assert len(observed)==72 and len(summary)==24 and len(setup)==12
    fig,axes=plt.subplots(2,2,figsize=(3.5,3.5))
    fig.subplots_adjust(left=.18,right=.975,bottom=.14,top=.815,wspace=.58,hspace=.72)
    x=np.asarray(SIZES)/1000
    all_times=positive([float(r['complete_query_seconds']) for r in observed])
    time_limits=(float(all_times.min())*.65,float(all_times.max())*1.4)
    for ax,direction,label,title in zip(axes[0],('image','text'),('a','b'),('Image to text','Text to image')):
        for j,method in enumerate(METHODS):
            centers=[];spread=[]
            for size,position in zip(SIZES,x):
                values=positive([float(r['complete_query_seconds']) for r in observed if int(r['images'])==size and r['method']==method and r['direction']==direction]);assert len(values)==3
                center=float(np.median(values));centers.append(center);spread.append((center-values.min(),values.max()-center))
                ax.scatter(position+np.array([-.16,0,.16]),values,s=5,alpha=.4,color=COLORS[method],edgecolors='none',zorder=2)
            ax.errorbar(x,centers,yerr=np.asarray(spread).T,color=COLORS[method],fmt='o-' if j<2 else 's-',
                markersize=3.5,markerfacecolor='none',markeredgewidth=.8,linewidth=1,capsize=1.6,zorder=3)
        panel(ax,label,title);ax.set_yscale('log');ax.set_ylabel('Query time (s)',labelpad=3)
        ax.set_ylim(*time_limits);ax.yaxis.set_major_locator(FixedLocator((.1,1,10,100)))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda value,_:f'{value:g}'))
        ax.yaxis.set_minor_locator(NullLocator())
    ax=axes[1,0]
    for method in METHODS[:2]:
        for direction,style in (('image','-'),('text','--')):
            values=positive([float(next(r for r in summary if int(r['images'])==size and r['method']==method and r['direction']==direction)['median_peer_dealer_MB']) for size in SIZES])
            ax.plot(x,values,linestyle=style,color=COLORS[method],marker='o' if direction=='image' else 's',markersize=2.8,linewidth=1)
    panel(ax,'c','MPC traffic');ax.set_yscale('log');ax.set_ylabel('IPC (MB/query)',labelpad=3)
    traffic_values=positive([float(r['median_peer_dealer_MB']) for r in summary if r['method']!='cached_local_limb'])
    ax.set_ylim(float(traffic_values.min())*.5,float(traffic_values.max())*3);ax.yaxis.set_major_locator(FixedLocator((.1,1,10,100,1000,10000)))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value,_:f'{value:g}'))
    ax.yaxis.set_minor_locator(NullLocator())
    # Direction labels sit in the shared margin, outside the measured curves.
    direction_handles=[Line2D([],[],color='#274351',linestyle='-',label='I2T'),Line2D([],[],color='#274351',linestyle='--',label='T2I')]
    fig.legend(handles=direction_handles,loc='upper center',bbox_to_anchor=(.54,.955),ncol=2,
        fontsize=7.5,handlelength=1.4,columnspacing=1.2,handletextpad=.4)
    ax=axes[1,1]
    for j,method in enumerate(METHODS):
        values=positive([float(next(r for r in setup if int(r['images'])==size and r['method']==method)['two_modality_setup_seconds']) for size in SIZES])
        ax.plot(x,values,color=COLORS[method],marker='o' if j<2 else 's',markersize=3,linewidth=1)
    panel(ax,'d','Setup cost');ax.set_yscale('log');ax.set_ylabel('Setup time (s)',labelpad=3)
    setup_values=positive([float(r['two_modality_setup_seconds']) for r in setup])
    ax.set_ylim(float(setup_values.min())*.65,float(setup_values.max())*1.4);ax.yaxis.set_major_locator(FixedLocator((1,10,100)))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value,_:f'{value:g}'))
    ax.yaxis.set_minor_locator(NullLocator())
    handles=[Line2D([],[],color=COLORS[m],marker='o' if i<2 else 's',markersize=3.1,linewidth=1,label=NAMES[m]) for i,m in enumerate(METHODS)]
    fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.54,1.01),ncol=3,handlelength=1.1,columnspacing=.75,handletextpad=.35)
    fig.savefig(OUT/'protocol_costs.pdf');fig.savefig(OUT/'protocol_costs.svg')
    fig.savefig(OUT/'protocol_costs.png',dpi=600);fig.savefig(OUT/'protocol_costs.tiff',dpi=600,pil_kwargs={'compression':'tiff_lzw'})
    plt.close(fig);print('Exported actual four-size protocol-cost figure',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data-dir',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True);a=p.parse_args()
    DATA=a.data_dir;OUT=a.output_dir;OUT.mkdir(parents=True,exist_ok=True);main()
