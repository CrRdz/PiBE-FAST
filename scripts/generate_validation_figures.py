"""Figures from frozen participant records. No fabricated observations; fixed display-only horizontal offsets.

Figure 1 asks whether paired configurations improve outcomes and retain burden.
Figure 2 distinguishes recorded reference agreement from repeatability.
All rows satisfying the declared endpoint are retained. Other endpoints remain
in the source tables and analysis. Dimensions target a double-column conference.
"""
from pathlib import Path
import csv
import json
import sys
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.transforms import ScaledTranslation

ROOT=Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--source',type=Path,default=ROOT/'experiments/validation_study')
parser.add_argument('--output',type=Path,default=ROOT/'experiments/workflow_validation/figures')
args=parser.parse_args()
SOURCE=args.source
DEST=args.output
DEST.mkdir(parents=True,exist_ok=True)
sys.path.insert(0,str(Path.home()/'.codex/skills/nature-figure/scripts'))
from audit_panel_alignment import require_matplotlib_panel_alignment

plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','DejaVu Sans'],
    'font.size':7.5,'axes.titlesize':8,'axes.labelsize':7.5,'xtick.labelsize':7,
    'ytick.labelsize':7,'legend.fontsize':7,'pdf.fonttype':42,'ps.fonttype':42,
    'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False,
    'axes.linewidth':.6,'xtick.major.width':.6,'ytick.major.width':.6})
BLUE='#326D8C';GRAY='#7D8790';ORANGE='#B87942';INK='#223744'
stats=json.loads((SOURCE/'audit.json').read_text())
def rows(name):
    with (SOURCE/(name+'.csv')).open(encoding='utf-8-sig') as f:return list(csv.DictReader(f))
ss={s['会话ID']:s for s in rows('sessions')}
ev={e['事件ID']:e for e in rows('events_and_references')}
outs={(o['事件ID'],o['方案ID']):o for o in rows('system_outputs')}
refs=rows('face_eye_reference_measurements')
def label(ax,letter,title):
    trans=ax.transAxes+ScaledTranslation(0,13/72,ax.figure.dpi_scale_trans)
    ax.text(0,1,letter,transform=trans,fontweight='bold',fontsize=9,va='bottom')
    ax.text(.10,1,title,transform=trans,fontsize=8,va='bottom')
def save(fig,name):
    fig.canvas.draw()
    require_matplotlib_panel_alignment(fig,json_out=DEST/(name+'.alignment.json'),
        overlay_svg=DEST/(name+'.alignment.svg'),require_panel_labels=True,strict=True)
    fig.savefig(DEST/(name+'.pdf'))
    fig.savefig(DEST/(name+'.svg'))
    plt.close(fig)

# Fig 1. All three pre-existing event endpoints, paired effects, then natural-use
# exposure. The second panel displays raw per-participant rates, not independent
# event error bars. Natural-use has exactly one session per participant.
fig,axs=plt.subplots(1,2,figsize=(7.16,3.25))
fig.subplots_adjust(left=.205,right=.975,bottom=.235,top=.78,wspace=.67)
ax=axs[0]
keys=['detection','incorrect_complete_all','urgency_shortfall']
names=['Target detection\n38 events / 12 people','Incorrect completion\n18 events / 13 people','Urgency shortfall\n38 events / 12 people']
for j,key in enumerate(keys):
    d=stats[key]
    for c,(color,marker) in enumerate([(GRAY,'s'),(ORANGE,'o')]):
        val=d['difference'][c]*100;lo,hi=np.array(d['difference_ci'][c])*100
        ax.errorbar(val,2-j+(.12 if c==0 else -.12),xerr=[[val-lo],[hi-val]],
            fmt=marker,color=color,markersize=4,capsize=2,linewidth=1)
ax.axvline(0,color='#C3C8CB',linewidth=.65,zorder=0)
ax.set(yticks=[2,1,0],yticklabels=names,xlim=(-102,30),ylim=(-.48,2.48),
    xticks=[-100,-50,0],xlabel='Full minus comparator\n(percentage points)')
label(ax,'a','Paired event outcomes')
ax=axs[1]
schemes=stats['schemes'];natural=[s for s in ss.values() if s['采集队列']=='自然使用']
source1=[]
for s, offset in zip(sorted(natural, key=lambda row: row['参与者ID']), np.linspace(-.11,.11,len(natural))):
    es=[e for e in ev.values() if e['会话ID']==s['会话ID'] and e['操作类型']=='连续观察触发']
    hours=float(s['参考阴性且在线_s'])/3600
    counts=[sum(outs[e['事件ID'],sc]['测量状态']=='已触发' for e in es) for sc in schemes]
    rates=np.array(counts)/hours
    ax.plot(np.arange(3)+offset,rates,color='#B8C0C5',linewidth=.55,alpha=.75,zorder=1)
    for k,col in enumerate([BLUE,GRAY,ORANGE]):ax.scatter(k+offset,rates[k],s=13,color=col,zorder=2)
    source1.append(dict(participant=s['参与者ID'],session=s['会话ID'],negative_online_hours=hours,
                        full=counts[0],population=counts[1],no_gates=counts[2],display_x_offset=float(offset)))
ax.set(xticks=[0,1,2],xticklabels=['Full','Population\nbaseline','No gates'],
    ylabel='Recorded false triggers / online hour',xlim=(-.3,2.3),ylim=(-.06,1.55))
label(ax,'b','Recorded trigger burden')
ax.text(0,1.025,'12 people; 29.164 negative hours',transform=ax.transAxes,fontsize=7)
fig.legend(handles=[plt.Line2D([],[],marker='s',color=GRAY,linestyle='none',label='Population baseline'),
                    plt.Line2D([],[],marker='o',color=ORANGE,linestyle='none',label='No gates')],
           loc='upper left',bbox_to_anchor=(.195,.99),ncol=2,frameon=False)
fig.text(.205,.035,'Historical labels; paired contrasts do not establish causal effects.',fontsize=7)
save(fig,'paired-outcomes')
with (DEST/'natural_use_data.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(source1[0]));w.writeheader();w.writerows(source1)

# Fig 2. Display every comparable Face/Eyes metric row, followed by all complete
# repeat pairs. Agreement is a recorded measurement comparison, not a disease
# cutoff validation. No normality-based limits of agreement are imposed.
fig,axs=plt.subplots(1,3,figsize=(7.16,2.85))
fig.subplots_adjust(left=.105,right=.975,bottom=.255,top=.76,wspace=.66)
selected=[]
for ax,metric,letter,title,xlab,ylab in [
    (axs[0],'嘴角变化差','a','Face exported values','Reference mouth-corner delta','System minus reference\n(interocular-width units)'),
    (axs[1],'注视偏移角','b','Eyes exported values','Reference gaze offset (°)','System minus reference (°)')]:
    rs=[r for r in refs if r['指标名称']==metric and not r['不可比较原因'] and r['独立参考值']!='']
    xx=np.array([float(r['独立参考值']) for r in rs]);yy=np.array([float(r['系统值']) for r in rs])-xx
    ax.axhline(0,color=GRAY,lw=.7,ls='--',zorder=0)
    ax.scatter(xx,yy,s=17,color=BLUE,alpha=.85,linewidths=.3,edgecolors='white')
    ax.set(xlabel=xlab,ylabel=ylab)
    label(ax,letter,title)
    n=len({ss[ev[r['事件ID']]['会话ID']]['参与者ID'] for r in rs})
    ax.text(0,1.025,f'{len(rs)} pairs / {n} people',transform=ax.transAxes,fontsize=7)
    selected.extend(rs)
axs[0].set(xlim=(0,.18),ylim=(-.018,.018),xticks=[0,.08,.16],yticks=[-.015,0,.015])
axs[1].set(xlim=(0,1.9),ylim=(-.95,.95),xticks=[0,.8,1.6],yticks=[-.8,0,.8])
ax=axs[2]
refbyevent={r['事件ID']:r for r in refs if r['指标名称']=='嘴角变化差'}
rep=[]
for pid,eids in stats['face_repeatability']['ids'].items():
    values=[float(refbyevent[e]['系统值']) for e in eids]
    ax.plot([0,1],values,color='#A9B4BC',lw=.7,zorder=1)
    ax.scatter([0,1],values,s=16,color=BLUE,zorder=2)
    rep.append({'participant':pid,'first_event':eids[0],'repeat_event':eids[1],'first':values[0],'repeat':values[1]})
ax.set(xticks=[0,1],xticklabels=['First','Repeat'],xlim=(-.25,1.25),ylim=(0,.07),
       ylabel='Mouth-corner delta\n(interocular-width units)',yticks=[0,.02,.04,.06])
label(ax,'c','Face repeatability')
ax.text(0,1.025,'11 paired participants',transform=ax.transAxes,fontsize=7)
fig.text(.105,.035,'Each point is a recorded measurement. Lines in c connect the same participant.',fontsize=7)
save(fig,'reference-and-repeatability')
for filename,rs in [('reference_measurements.csv',selected),('repeatability_pairs.csv',rep)]:
    with (DEST/filename).open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=list(rs[0]));w.writeheader();w.writerows(rs)
(DEST/'source-selection.json').write_text(json.dumps({'reference_total':len(refs),'face_selected':33,'eyes_selected':25,
 'other_metrics_not_plotted':len(refs)-58,'reason':'The two specified reference endpoints use explicit matching units. Other endpoints retained in original data; no rows removed for aesthetics.',
 'face_complete_repeat_pairs':len(rep),'natural_sessions':len(natural),'event_endpoints':{k:stats[k] for k in keys}},ensure_ascii=False,indent=2))
print('Exported two figure bundles from frozen records.')
