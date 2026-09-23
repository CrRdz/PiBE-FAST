"""Pictorial Fig. 1; original vector artwork with separate research/evidence paths."""
from pathlib import Path
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Rectangle, Polygon, Arc
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(Path.home()/'.codex/skills/nature-figure/scripts'))
from audit_panel_alignment import require_matplotlib_panel_alignment
OUT=ROOT/'docs/pibefast-paper/figures/system-overview'
QA=ROOT/'experiments/fig1_visual_revision'
plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','DejaVu Sans'],
 'font.size':6.3,'pdf.fonttype':42,'svg.fonttype':'none'})
fig=plt.figure(figsize=(7.16,3.25));ax=fig.add_axes([0,0,1,1]);ax.set(xlim=(0,7.16),ylim=(0,3.25));ax.axis('off')
ink='#273B4D';blue='#2D638B';green='#397B64';purple='#796394';gold='#A77D28';red='#A24F4C'
colors=[blue,green,purple,gold,red,blue]
fills=['#ECF3FA','#EBF5EF','#F1EDF8','#FCF7E7','#FAEEEE','#ECF3FA']
xs=[.10+1.18*i for i in range(6)];W=1.06

def txt(x,y,s,size=6.3,color=ink,weight='normal',**kw):
 return ax.text(x,y,s,ha='center',va='center',fontsize=size,color=color,fontweight=weight,linespacing=1.35,**kw)
def rounded(x,y,w,h,fc,ec=None,lw=.6,r=.055):
 p=FancyBboxPatch((x,y),w,h,boxstyle=f'round,pad=0,rounding_size={r}',facecolor=fc,edgecolor=ec or fc,lw=lw);ax.add_patch(p);return p
def line(points,color=ink,lw=1.2):
 ax.plot(*zip(*points),color=color,lw=lw,solid_capstyle='round',solid_joinstyle='round')
def arrow(points,color=blue,lw=1.25,scale=9):
 if len(points)>2:line(points[:-1],color,lw)
 ax.add_patch(FancyArrowPatch(points[-2],points[-1],arrowstyle='-|>',mutation_scale=scale,color=color,lw=lw,shrinkA=0,shrinkB=1.6))
def circle(x,y,r,fc,ec=None,lw=.8):
 ax.add_patch(Circle((x,y),r,facecolor=fc,edgecolor=ec or fc,lw=lw))
def person(cx,cy,s=.2,arms='down',color=ink):
 circle(cx,cy+s*.82,s*.18,color)
 line([(cx,cy+s*.60),(cx,cy-s*.10)],color,1.5)
 if arms=='up':pts=[(cx-s*.48,cy+s*.68),(cx-s*.43,cy+s*.23),(cx+s*.43,cy+s*.23),(cx+s*.48,cy+s*.68)]
 else:pts=[(cx-s*.43,cy-s*.08),(cx-s*.30,cy+s*.38),(cx+s*.30,cy+s*.38),(cx+s*.43,cy-s*.08)]
 line(pts,color,1.25);line([(cx-s*.35,cy-s*.65),(cx,cy-s*.10),(cx+s*.35,cy-s*.65)],color,1.4)
def eye(cx,cy,s=.15):
 ax.add_patch(Arc((cx,cy-.02),2*s,s*1.22,theta1=8,theta2=172,color=ink,lw=1.1))
 ax.add_patch(Arc((cx,cy+.02),2*s,s*1.22,theta1=188,theta2=352,color=ink,lw=1.1))
 circle(cx,cy,s*.39,'#A9C1D0',ink,.65);circle(cx,cy,s*.16,ink)
def face(cx,cy,s=.12):
 circle(cx,cy,s,'#F5EEE2',ink,.8);circle(cx-s*.35,cy+s*.2,s*.08,ink);circle(cx+s*.35,cy+s*.2,s*.08,ink)
 ax.add_patch(Arc((cx,cy),s*1.2,s*.9,theta1=205,theta2=335,color=ink,lw=.8))
def mic(cx,cy,s=.25):
 rounded(cx-s*.21,cy-s*.28,s*.42,s*.92,'#63869E',ink,.9,r=s*.2)
 ax.add_patch(Arc((cx,cy+s*.06),s*.75,s*1.18,theta1=175,theta2=365,color=ink,lw=1.1))
 line([(cx,cy-s*.54),(cx,cy-s*.84)],ink,1.1);line([(cx-s*.26,cy-s*.84),(cx+s*.26,cy-s*.84)],ink,1.1)
 for dy in [.02,.16,.30]:line([(cx-s*.14,cy+s*dy),(cx+s*.14,cy+s*dy)],'#B9CFDD',.55)
def clock(cx,cy,s=.11):
 circle(cx,cy,s,'white',ink,.8);line([(cx,cy+s*.58),(cx,cy),(cx+s*.46,cy)],ink,.8)
def pill(x,y,w,h,label,fc,color):
 rounded(x,y,w,h,fc,r=.07);txt(x+w/2,y+h/2,label,6.0,color)
# Six softly colored stages, matching the supplied visual grammar.
titles=['Sensing','Edge control','BE-FAST check','Evidence paths','Report assessment','User interaction']
subtitles=['Home environment','Raspberry Pi','Guided measurements','Parallel processing','Two independent axes','Display and acknowledge']
for i,(x,title,sub) in enumerate(zip(xs,titles,subtitles)):
 rounded(x+.012,.49,W,2.38,'#EEF0F2',r=.07)
 rounded(x,.51,W,2.38,fills[i],colors[i],.65,r=.07)
 circle(x+W/2,2.97,.12,colors[i]);txt(x+W/2,2.97,str(i+1),9,'white','bold')
 txt(x+W/2,2.66,title,7.2,colors[i],'bold');txt(x+W/2,2.44,sub,5.6)
# 1: camera and microphone, with a visual audio trace.
x=xs[0];cx=x+.31;cy=1.91
circle(cx,cy,.20,'#68869C',ink,1);circle(cx,cy,.15,'#C5D6E2',ink,.7);circle(cx,cy,.106,ink)
circle(cx,cy,.061,'#4386B0');circle(cx-.022,cy+.024,.021,'#DCEEF6')
line([(cx,cy-.2),(cx,cy-.36)],ink,2);rounded(cx-.16,cy-.40,.32,.06,ink,r=.025)
mic(x+.78,1.91,.27)
txt(x+.31,1.34,'Camera',6.2);txt(x+.78,1.34,'Microphone',6.0)
for j in range(21):
 h=.04+.07*(np.sin(j*.8)**2);line([(x+.17+j*.035,1.16-h/2),(x+.17+j*.035,1.16+h/2)],blue,.85)
rounded(x+.10,.67,.86,.38,'#DDEAF5');txt(x+.53,.86,'Video and audio\nPassive B / S',6.2,blue)
# 2: original single-board-computer icon, no logo.
x=xs[1];rounded(x+.14,1.59,.77,.54,'#4F9073','#32644D',.8,r=.04)
for px in [x+.20,x+.85]:
 for py in [1.65,2.07]:circle(px,py,.027,'#EDD797')
rounded(x+.36,1.76,.23,.21,'#344B49',r=.012)
for k in range(5):
 line([(x+.35,1.78+k*.036),(x+.30,1.78+k*.036)],'#BED6A9',.6)
 line([(x+.60,1.78+k*.036),(x+.65,1.78+k*.036)],'#BED6A9',.6)
for py in [1.67,1.93]:
 rounded(x+.76,py,.19,.15,'#C9D3D4','#57716B',.7,r=.01);rounded(x+.82,py+.035,.105,.08,'#647778',r=.003)
for j in range(10):ax.add_patch(Rectangle((x+.26+j*.035,2.05),.015,.055,facecolor='#E0C66F',edgecolor='none'))
rounded(x+.12,.70,.82,.60,'#DEEEE5');txt(x+.53,1.0,'Retain triggers\nManage retry queue\nRequest active check',6.15,green)
# 3: task pictograms; T is separately identified as onset metadata.
x=xs[2]
for col,row,code,name in [(0,0,'B','Balance'),(1,0,'E','Eyes'),(0,1,'F','Face'),(1,1,'A','Arms'),(0,2,'S','Speech'),(1,2,'T','Onset')]:
 xx=x+.07+col*.48;yy=1.84-row*.51
 rounded(xx,yy,.44,.46,'#FCFBFE',r=.04);cx=xx+.22;cy=yy+.29
 if code=='B':person(cx,cy-.015,.12,color=purple)
 elif code=='E':eye(cx,cy,.11)
 elif code=='F':face(cx,cy,.104)
 elif code=='A':person(cx,cy-.025,.115,'up',purple)
 elif code=='S':mic(cx,cy,.125)
 else:clock(cx,cy,.10)
 txt(cx,yy+.075,code+' '+name,5.65,purple)
txt(x+.53,.64,'Video BEFA · Audio S',5.65,purple)
# 4: research outputs are a terminal branch; only lifecycle feeds the report.
x=xs[3]
rounded(x+.07,1.75,.92,.56,'#F3EEDB','#D8C997',.5,r=.045)
txt(x+.53,2.18,'Research\nrepresentation',6.0,gold,'bold')
for j,c in enumerate([blue,green,purple,red]):
 rounded(x+.19+j*.17,1.94,.13,.095,c,r=.015)
txt(x+.53,1.82,'Metrics · quality · masks',5.6,gold)
rounded(x+.07,.70,.92,.79,'#FFFDF5','#D8C997',.5,r=.045)
txt(x+.53,1.36,'Evidence lifecycle',6.0,gold,'bold')
for j in range(3):
 rounded(x+.21+j*.20,1.095,.15,.13,'#E7D9AE','#AA8E43',.6,r=.017)
 if j<2:arrow([(x+.365+j*.20,1.16),(x+.405+j*.20,1.16)],gold,.65,5)
txt(x+.53,.86,'Updates + corrections\nSymptoms + onset',5.8,gold)
# 5: independent urgency and completeness, never four mutually exclusive states.
x=xs[4];txt(x+.53,2.23,'Urgency',6.8,red,'bold')
for yy,label,fc,c in [(1.95,'!   Urgent','#F2D9D8','#A73E3B'),(1.68,'!   Warning','#F5E4CC','#986018'),(1.41,'—   None','#E4ECE7','#4F7061')]:pill(x+.11,yy,.84,.22,label,fc,c)
txt(x+.53,1.20,'Completeness',6.4,red,'bold')
pill(x+.11,.89,.84,.20,'Complete','#E4ECE7','#4F7061');pill(x+.11,.64,.84,.20,'Incomplete','#E7E8EB','#57636F')
# 6: local monitor plus user silhouette and acknowledgement.
x=xs[5];rounded(x+.115,1.68,.83,.60,'#526D80',ink,.9,r=.035);rounded(x+.165,1.74,.73,.48,'#F5F9FC',r=.01)
txt(x+.53,2.08,'Current report',6.0,blue,'bold')
for j,w in enumerate([.42,.32,.36]):rounded(x+.24,1.94-j*.065,w,.023,'#B2C9D9',r=.009)
line([(x+.53,1.68),(x+.53,1.54)],ink,2);rounded(x+.34,1.49,.38,.045,ink,r=.018)
circle(x+.79,1.47,.115,blue);rounded(x+.64,1.20,.30,.23,blue,r=.085)
rounded(x+.10,.70,.86,.39,'#DDEAF5');txt(x+.53,.895,'Versioned history\nUser acknowledgement',5.9,blue)
# Horizontal control path, with an explicit fork around the research branch.
arrow([(xs[0]+W,1.79),(xs[1],1.79)])
arrow([(xs[1]+W,1.79),(xs[2],1.79)])
# Acquired measurements split to representation and lifecycle; T/symptom metadata enters only lifecycle.
arrow([(xs[2]+W,2.00),(xs[3]+.065,2.00)],purple,1.0,7)
arrow([(xs[2]+W,1.16),(xs[3]+.065,1.16)],purple,1.0,7)
arrow([(xs[3]+W,1.10),(xs[4],1.10)],gold,1.15,8)
arrow([(xs[4]+W,1.79),(xs[5],1.79)],red,1.15,8)
# Feedback loop below all panels; no arrows through labels.
line([(xs[5]+.53,.50),(xs[5]+.53,.22),(4.85,.22)],'#7F95A5',1.8)
arrow([(2.31,.22),(xs[0]+.53,.22),(xs[0]+.53,.50)],'#7F95A5',1.8,9)
txt(3.58,.22,'Resume monitoring after session / cooldown',6.4,blue)
fig.canvas.draw()
require_matplotlib_panel_alignment(fig,json_out=str(QA/'overview.alignment.json'),tolerance_pt=1.5,gutter_tolerance_pt=1.5,strict=True)
fig.savefig(str(OUT)+'.pdf',facecolor='white')
fig.savefig(str(OUT)+'.svg',facecolor='white')
plt.close(fig)
