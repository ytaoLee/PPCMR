"""Compact measured panels and original vector system pictograms.

Keep retained measurements intact. Accuracy uses paired image-cluster 95%
bootstrap intervals; n=3 timing ranges are descriptive min--max, never CIs.
"""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Circle,Rectangle,FancyBboxPatch,Polygon,FancyArrowPatch,PathPatch
from matplotlib.path import Path as MPath
from matplotlib.transforms import Affine2D
from matplotlib.ticker import FixedLocator,FuncFormatter

ROOT=Path(__file__).resolve().parent.parent
OUT=ROOT/'figures'
METHODS=('fresh_mpc','cached_mpc','cached_local_limb')
COLORS={'fresh_mpc':'#778690','cached_mpc':'#287BA7','cached_local_limb':'#B78651'}
NAMES={'fresh_mpc':'Fresh MPC','cached_mpc':'Cached MPC','cached_local_limb':'Cached local'}
INK='#274351';BLUE='#287BA7';TEAL='#4A8175';PALE='#F2F7F9';GRAY='#738491'
mpl.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','DejaVu Sans'],
    'font.size':7,'axes.labelsize':7,'axes.titlesize':7,'xtick.labelsize':6.4,'ytick.labelsize':6.4,
    'legend.fontsize':6.5,'axes.spines.top':True,'axes.spines.right':True,'axes.linewidth':.65,
    'axes.edgecolor':'#788A93','axes.labelcolor':INK,'text.color':INK,'xtick.color':INK,'ytick.color':INK,
    'xtick.major.size':2.5,'ytick.major.size':2.5,'xtick.major.width':.6,'ytick.major.width':.6,
    'pdf.fonttype':42,'svg.fonttype':'none','legend.frameon':False,'savefig.facecolor':'white'})


def rows(name):
    with (DATA/name).open(encoding='utf-8',newline='') as stream:return list(csv.DictReader(stream))


def export(fig,name):
    fig.savefig(OUT/f'{name}.pdf')
    fig.savefig(OUT/f'{name}.svg')
    fig.savefig(OUT/f'{name}.png',dpi=600)
    fig.savefig(OUT/f'{name}.tiff',dpi=600,pil_kwargs={'compression':'tiff_lzw'})
    plt.close(fig)


def panel(ax,label,title):
    ax.text(0,1.11,label,transform=ax.transAxes,fontsize=8,fontweight='bold',va='bottom')
    ax.text(.12,1.11,title,transform=ax.transAxes,fontsize=7,va='bottom')
    ax.tick_params(which='both',pad=2,top=True,right=True,direction='in')
    ax.set_axisbelow(True)
    ax.grid(axis='both',which='both',color='#DDE5E9',linewidth=.45,alpha=.65)


def method_legend(fig,y=.99):
    handles=[Line2D([],[],color=COLORS[method],marker=('o','o','s')[i],markersize=3.1,
             linewidth=1.05,label=NAMES[method]) for i,method in enumerate(METHODS)]
    fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.54,y),ncol=3,
               handlelength=1.1,columnspacing=.75,handletextpad=.35)


def accuracy():
    data=rows('accuracy_ci.csv')
    fig,axes=plt.subplots(1,2,figsize=(3.5,1.65),sharey=True)
    fig.subplots_adjust(left=.14,right=.985,bottom=.27,top=.73,wspace=.26)
    for ax,direction,label,title in zip(axes,('image-to-text','text-to-image'),('a','b'),('Image to text','Text to image')):
        for dataset,color in (('COCO',BLUE),('Flickr30k','#B78651')):
            # FP32 curves visually overlap; the manuscript reports all differences
            # in a separate table. Preserve the source CSV for both modes.
            for arithmetic,style,marker in (('Exact','-','o'),):
                part=sorted([row for row in data if row['dataset']==dataset and row['direction']==direction and row['arithmetic']==arithmetic],key=lambda row:int(row['k']))
                x=np.array([int(row['k']) for row in part]);y=np.array([float(row['recall_percent']) for row in part])
                low=np.array([float(row['ci_low_percent']) for row in part]);high=np.array([float(row['ci_high_percent']) for row in part])
                assert len(y)==3 and np.all((0<=low)&(low<=high)&(high<=100))
                if arithmetic=='Exact':ax.fill_between(x,low,high,color=color,alpha=.15,linewidth=0)
                ax.plot(x,y,color=color,linestyle=style,linewidth=1.05,marker=marker,
                        markersize=4 if arithmetic=='Exact' else 2.6,markeredgewidth=.7,
                        markerfacecolor=color,label=dataset)
        panel(ax,label,title);ax.set_xticks((1,5,10));ax.set_xlim(.2,10.8);ax.set_ylim(20,103)
        ax.set_yticks((20,60,100));ax.set_xlabel('Retrieval cutoff k',labelpad=2)
    axes[0].set_ylabel('Recall (%)',labelpad=3)
    handles,labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.54,1.02),ncol=2,
               handlelength=1.45,columnspacing=.95,handletextpad=.4,labelspacing=.4)
    export(fig,'retrieval_accuracy')


if __name__=='__main__':
    p=argparse.ArgumentParser(description='Plot Exact recall; FP32 differences belong in the companion audit table')
    p.add_argument('--data-dir',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True);a=p.parse_args()
    DATA=a.data_dir
    OUT=a.output_dir;OUT.mkdir(parents=True,exist_ok=True)
    accuracy()
