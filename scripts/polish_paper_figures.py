"""Publication figures from frozen supplied records; no imputation or smoothing."""
from pathlib import Path
import csv, json, hashlib, sys
sys.path.insert(0, str(Path.home()/'.codex/skills/nature-figure/scripts'))
from audit_panel_alignment import require_matplotlib_panel_alignment
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'docs/pibefast-paper/figures'
DATA=ROOT/'tmp/figure-audits'
plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','DejaVu Sans'],'font.size':7.5,'svg.fonttype':'none','axes.linewidth':.6,'xtick.major.width':.6,'ytick.major.width':.6,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'ps.fonttype':42})
COLORS=['#397493','#C07A42']
def read(name):
    with (ROOT/name).open() as f:return list(csv.DictReader(f))
def save(fig,name):
    fig.canvas.draw()
    require_matplotlib_panel_alignment(fig,json_out=DATA/(name+'.alignment.json'),tolerance_pt=1.5,gutter_tolerance_pt=1.5,strict=True)
    fig.savefig(OUT/(name+'.pdf'),bbox_inches='tight',pad_inches=.06)
    fig.savefig(OUT/(name+'.svg'),bbox_inches='tight',pad_inches=.06)
    plt.close(fig)

def main():
    OUT.mkdir(exist_ok=True);DATA.mkdir(parents=True,exist_ok=True)
    events=read('research_data/pipc_pilot/events.csv')
    groups=[('Earlier',{'run_001','run_002','run_003'}),('Later',{'run_004','run_005','run_006'})]
    counts=[]
    fig,axs=plt.subplots(1,2,figsize=(7.16,2.35),layout='constrained')
    for ax,truth,components,title in zip(axs,['true','false'],[['B','S'],['B','S','none']],['a  Positive-event detection','b  Negative-interval triggers']):
        for j,(group,runs) in enumerate(groups):
            vals=[];labels=[]
            for c in components:
                rows=[r for r in events if r['run_id'] in runs and r['ground_truth_component']==c and r['ground_truth_event']==truth]
                n=len(rows);hit=sum(r['trigger_generated']=='True' for r in rows)
                vals.append(100*hit/n);labels.append(f'{hit}/{n}')
                counts.append(dict(group=group,component=c,truth=truth,triggers=hit,denominator=n,source_rows=[r['source_row'] for r in rows]))
            x=np.arange(len(components))+(j-.5)*.34
            ax.bar(x,vals,width=.31,color=COLORS[j],label=group)
            for pos,val,label in zip(x,vals,labels):ax.text(pos,val+3,label,ha='center',fontsize=8)
        ax.set_xticks(np.arange(len(components)),['B','S'] if len(components)==2 else ['B','S','No component'])
        ax.set_ylim(0,115);ax.set_yticks([0,25,50,75,100]);ax.set_ylabel('Recorded intervals (%)');ax.set_title(title,loc='left',fontsize=8,fontweight='bold',pad=10)
        ax.set_axisbelow(True)
    fig.legend(*axs[0].get_legend_handles_labels(),loc='outside upper center',frameon=False,ncol=2,fontsize=7)
    save(fig,'pilot-counts')
    (DATA/'plot_counts.json').write_text(json.dumps(counts,indent=2)+'\n')

    resources=[r for r in read('research_data/pipc_pilot/resources.csv') if r['run_id']=='run_008']
    x=np.array([float(r['elapsed_s'])/3600 for r in resources])
    fig,axs=plt.subplots(2,1,figsize=(7.16,2.65),sharex=True,layout='constrained')
    for ax,key,label,color in zip(axs,['pi_cpu_pct','pi_temperature_c'],['Pi CPU (%)','Temperature (°C)'],COLORS):
        y=[float(r[key]) if r[key] and r['pi_online']=='True' else np.nan for r in resources]
        ax.plot(x,y,color=color,lw=.7,marker='.',ms=1.2)
        # Keep offline source values visible as isolated crosses, never connected.
        off=[i for i,r in enumerate(resources) if r['pi_online']!='True' and r[key]]
        if off:ax.scatter(x[off],[float(resources[i][key]) for i in off],marker='x',color='black',s=22,zorder=4)
        for i,r in enumerate(resources):
            if r['data_loss_observed']=='True':ax.axvline(x[i],color='#9b3748',ls=':',lw=.9)
        ax.set_ylabel(label);ax.grid(axis='y',color='#E6E9EC',lw=.45);ax.set_xlim(0,8)
    axs[0].set_ylim(0,100);axs[1].set_ylim(0,90)
    axs[0].set_title('a  Processor utilization',fontsize=8,fontweight='bold',loc='left',pad=8)
    axs[1].set_title('b  Device temperature',fontsize=8,fontweight='bold',loc='left',pad=8)
    axs[1].set_xlabel('Elapsed run time (h)')
    from matplotlib.lines import Line2D
    fig.legend(handles=[Line2D([],[],marker='x',ls='',color='black',label='Recorded while offline'),Line2D([],[],ls=':',color='#9b3748',label='Data-loss flag')],fontsize=7,ncol=2,frameon=False,loc='outside upper center')
    save(fig,'resource-trace')

    speakers=[r for r in read('experiments/software_validation/mdsc_speakers.csv') if r['split']=='test']
    fig,ax=plt.subplots(figsize=(3.5,2.35),layout='constrained')
    rates=[100*(int(r['tp'])+int(r['tn']))/int(r['recordings']) for r in speakers]
    ax.barh(np.arange(6),rates,color=[COLORS[int(r['label'])] for r in speakers],height=.6)
    ax.set_yticks(np.arange(6),[r['speaker'] for r in speakers]);ax.invert_yaxis();ax.set_xlim(0,100)
    ax.set_xticks([0,25,50,75,100]);ax.set_xlabel('Correctly classified recordings (%)')
    for i,(r,val) in enumerate(zip(speakers,rates)):ax.text(val-1.5 if val>70 else val+1.2,i,f"{int(r['tp'])+int(r['tn'])}/405",va='center',ha='right' if val>70 else 'left',color='white' if val>70 else 'black',fontsize=7)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=COLORS[0],label='Control'),Patch(color=COLORS[1],label='Dysarthric')],loc='lower left',fontsize=6.7,frameon=False,ncol=2,bbox_to_anchor=(0,1.0))
    ax.set_axisbelow(True)
    save(fig,'mdsc-speaker-results')

    healthy=json.loads((ROOT/'research_data/human_repeatability/analysis_report.json').read_text())['modules']
    fig,ax=plt.subplots(figsize=(3.5,2.1),layout='constrained')
    for i,c in enumerate('BEFA'):
        r=healthy[c];v=r['icc_1_1'];lo,hi=r['icc_1_1_participant_bootstrap_95pct']
        ax.errorbar(v,i,xerr=[[v-lo],[hi-v]],fmt='o',color=COLORS[0],capsize=3)
        ax.text(1.04,i,f'{v:.3f}',ha='left',va='center',fontsize=7,clip_on=False)
    ax.set_yticks(range(4),['Balance','Eyes','Face','Arms']);ax.invert_yaxis();ax.set_xlim(0,1)
    ax.set_xlabel('ICC(1,1)')
    ax.grid(axis='x',color='#E6E9EC',lw=.45);ax.set_axisbelow(True)
    save(fig,'healthy-repeatability')


if __name__=='__main__':main()
