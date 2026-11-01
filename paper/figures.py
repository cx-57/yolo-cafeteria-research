"""Deterministic, source-backed figures for the cafeteria object-detection paper."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib import rcParams
import numpy as np

ROOT = Path(__file__).resolve().parent
NAVY, BLUE, LIGHT, ORANGE, GRAY = "#152B45", "#27639A", "#EFF4F9", "#C78033", "#516378"
rcParams.update({"font.family":"DejaVu Sans","font.size":9,"savefig.dpi":220})

def box(ax, x, y, width, height, title, subtitle="", fill="white"):
    p = FancyBboxPatch((x,y),width,height,boxstyle="round,pad=0.011,rounding_size=0.02",
                       linewidth=1.25,edgecolor=BLUE,facecolor=fill)
    ax.add_patch(p)
    ax.text(x+width/2,y+height*0.63,title,ha="center",va="center",
            fontsize=9,fontweight="bold",color=NAVY)
    if subtitle:
        ax.text(x+width/2,y+height*.31,subtitle,ha="center",va="center",
                fontsize=7.6,color=GRAY)
    return (x,y,width,height)

def arrow(ax,x1,y1,x2,y2):
    ax.add_patch(FancyArrowPatch((x1,y1),(x2,y2),arrowstyle="-|>",mutation_scale=11,
                  lw=1.1,color=BLUE,connectionstyle="arc3"))

fig,ax=plt.subplots(figsize=(11.8,3.6))
ax.set_xlim(0,12.3); ax.set_ylim(0,3.8);ax.axis("off")
box(ax,.12,1.57,2.0,.72,"Cafeteria image","Multi-object scene")
labels=[
("YOLO11n","Food / hinge / carton"),
("YOLO11n","Plate / wrapper"),
("YOLO11x","Utensil"),
("YOLO11s","Bottle")]
ys=[2.94,2.07,1.20,.33]
for (name,description),y in zip(labels,ys):
    box(ax,3.0,y,2.65,.64,name,description,LIGHT)
    arrow(ax,2.12,1.93,3.0,y+.31)
    arrow(ax,5.65,y+.31,7.0,1.93)
box(ax,7.0,1.56,2.04,.74,"Seven-class boxes","Object + confidence")
box(ax,9.88,1.56,2.18,.74,"Suggestion policy","Compost / recycle / review")
arrow(ax,9.04,1.93,9.88,1.93)
ax.text(6.05,3.62,"Four independent specialist checkpoints",ha="center",color=GRAY,fontsize=9)
fig.savefig(ROOT/"fig-pipeline.png",bbox_inches="tight",pad_inches=.12)
plt.close(fig)

fig,ax=plt.subplots(figsize=(10.2,3.55))
x=np.arange(4)
precision=[93.00,97.03,93.42,97.22]
recall=[82.11,71.43,78.35,68.18]
xticks=["Validation\nbaseline","Validation\nclasswise","Historical\nbaseline","Historical\nclasswise"]
ax.plot(x,precision,marker="o",markersize=7,color=BLUE,lw=2.4,label="Micro precision")
ax.plot(x,recall,marker="s",markersize=6,color=ORANGE,lw=2.1,label="Micro recall")
for i,y in enumerate(precision):
    ax.annotate(f"{y:.2f}%",(i,y),xytext=(0,8),textcoords="offset points",ha="center",
                fontsize=9,color=BLUE,fontweight="bold")
for i,y in enumerate(recall):
    ax.annotate(f"{y:.2f}%",(i,y),xytext=(0,-16),textcoords="offset points",ha="center",
                fontsize=9,color=ORANGE,fontweight="bold")
ax.set_xticks(x,xticks)
ax.set_ylim(60,103)
ax.set_xlim(-.3,3.3)
ax.set_ylabel("Object detection (%)",color=NAVY)
ax.grid(axis="y",color="#DCE5EC",lw=.7)
ax.spines["top"].set_visible(False);ax.spines["right"].set_visible(False)
ax.legend(loc="lower left",frameon=False,ncol=2,bbox_to_anchor=(.01,1.0))
ax.axvline(1.5,color="#A7B2BE",linestyle="--",lw=.85)
ax.text(.48,60.7,"186-photo validation",ha="center",fontsize=8,color=GRAY)
ax.text(2.5,60.7,"248-photo exposed cohort",ha="center",fontsize=8,color=GRAY)
fig.tight_layout()
fig.savefig(ROOT/"fig-precision.png",bbox_inches="tight",pad_inches=.1)
plt.close(fig)
print("Built fig-pipeline.png and fig-precision.png")
