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

# Paired, unconnected comparisons: validation and previously exposed historical images
# are separate cohorts, so connecting their scores would imply a false trajectory.
fig, ax = plt.subplots(figsize=(10.2, 3.65))
positions = [0, 1, 2.55, 3.55]
precision = [93.00, 97.03, 93.42, 97.22]
recall = [82.11, 71.43, 78.35, 68.18]
labels = ["Uniform", "Classwise", "Uniform", "Classwise"]
bar_width = 0.32

p_bars = ax.bar([v - bar_width / 2 for v in positions], precision, bar_width,
                label="Precision", color=BLUE, alpha=0.92)
r_bars = ax.bar([v + bar_width / 2 for v in positions], recall, bar_width,
                label="Recall", color=ORANGE, alpha=0.90)
for bars in [p_bars, r_bars]:
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f"{height:.2f}%", (bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 4), textcoords="offset points", ha="center", va="bottom",
                    fontsize=8, fontweight="bold", color=NAVY)
ax.set_ylim(0, 110)
ax.set_yticks([0, 20, 40, 60, 80, 100])
ax.set_xticks(positions, labels)
ax.set_ylabel("Object detection (%)", color=NAVY)
ax.axvline(1.78, color="#A7B2BE", linestyle="--", linewidth=0.8)
ax.text(0.5, -0.13, "Validation · 186 images", ha="center", fontsize=9,
        transform=ax.get_xaxis_transform(), color=GRAY)
ax.text(3.05, -0.13, "Historical development · 248 images", ha="center",
        fontsize=9, transform=ax.get_xaxis_transform(), color=GRAY)
ax.grid(axis="y", color="#DCE5EC", linewidth=0.6)
ax.set_axisbelow(True)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.04),
          ncol=2, frameon=False)
fig.subplots_adjust(bottom=0.27, top=0.85, left=0.09, right=0.98)
fig.savefig(ROOT / "fig-precision.png", bbox_inches="tight", pad_inches=0.12)
plt.close(fig)

print("Built fig-pipeline.png and fig-precision.png")
