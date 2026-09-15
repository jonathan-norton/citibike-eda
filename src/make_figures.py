"""
Five figures for the Citi Bike EDA.

Each answers a stated question rather than displaying a variable.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

plt.rcParams.update({
    "figure.dpi": 130,
    "savefig.dpi": 130,
    "font.size": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linewidth": 0.6,
    "figure.facecolor": "white",
})

MEMBER_C = "#1f4e79"
CASUAL_C = "#d17b0f"
OUT = "reports/figures"
os.makedirs(OUT, exist_ok=True)

df = pd.read_parquet("trips_slim.parquet")
valid = df[~df.incomplete_trip & ~df.depot_trip]

MONTH_LABEL = {"202010": "Oct 2020", "202101": "Jan 2021", "202102": "Feb 2021"}


# --------------------------------------------------------------- figure 1
# Q: does the cleaning decision on long trips actually matter?
def fig1():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.4))

    d_all = df.duration_min
    d_val = valid.duration_min

    # range= rather than clip() so the tail is excluded, not piled on the edge
    ax1.hist(d_val, bins=60, range=(0, 60), color=MEMBER_C, alpha=.85)
    ax1.axvline(d_val.median(), color="black", lw=1.2)
    ax1.axvline(d_all.mean(), color="crimson", lw=1.2, ls="--")
    ax1.text(d_val.median() + 1.5, ax1.get_ylim()[1] * .90,
             f"median {d_val.median():.1f}", fontsize=8)
    ax1.text(d_all.mean() + 1.5, ax1.get_ylim()[1] * .76,
             f"raw mean {d_all.mean():.1f}", fontsize=8, color="crimson")
    ax1.set_xlabel("trip duration (minutes; tail beyond 60 not shown)")
    ax1.set_ylabel("trips")
    ax1.set_title("Most trips are short; the raw mean\nfalls where almost no trips are",
                  loc="left", fontsize=9.5)

    stats = {"raw mean": d_all.mean(), "p99": d_all.quantile(.99),
             "cleaned mean": d_val.mean(), "median": d_val.median()}
    cols = {"raw mean": "crimson", "p99": "grey",
            "cleaned mean": MEMBER_C, "median": "black"}
    keys = list(stats)[::-1]
    ax2.barh(keys, [stats[k] for k in keys],
             color=[cols[k] for k in keys], height=.6)
    for i, k in enumerate(keys):
        ax2.text(stats[k] + 1.5, i, f"{stats[k]:.1f}", va="center", fontsize=8)
    ax2.set_xlabel("minutes")
    ax2.set_xlim(0, max(stats.values()) * 1.22)
    n = int(df.incomplete_trip.sum())
    ax2.set_title(f"{n:,} incomplete trips (0.3% of rows) inflate\n"
                  f"the mean from {d_val.mean():.1f} to {d_all.mean():.1f} minutes",
                  loc="left", fontsize=9.5)

    fig.tight_layout()
    fig.savefig(f"{OUT}/fig1_duration_outliers.png", bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------- figure 2
# Q: do members and casual riders use the system at different times of day?
def fig2():
    h = (valid.groupby(["start_hour", "member_casual"], observed=True)
              .size().unstack(fill_value=0))
    share = h.div(h.sum(axis=0), axis=1) * 100

    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    ax.plot(share.index, share["member"], color=MEMBER_C, lw=2, label="member")
    ax.plot(share.index, share["casual"], color=CASUAL_C, lw=2, label="casual")
    ax.fill_between(share.index, share["member"], color=MEMBER_C, alpha=.10)
    ax.fill_between(share.index, share["casual"], color=CASUAL_C, alpha=.10)

    for x in (8,):
        ax.axvline(x, color="grey", lw=.8, ls=":", zorder=0)
    
    ax.annotate("members only", xy=(8, share["member"][8]),
                xytext=(4.2, share.values.max()*0.93), fontsize=8, color=MEMBER_C,
                arrowprops=dict(arrowstyle="->", color=MEMBER_C, lw=.9))

    ax.set_xticks(range(0, 24, 2))
    ax.set_xlabel("hour of day")
    ax.set_ylabel("% of that group's trips")
    ax.set_title("Both peak at 5pm, but only members show a morning commute spike",
                 loc="left", fontsize=9.5)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig2_hourly_by_type.png", bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------- figure 3
# Q: how does demand vary by hour and weekday, and does the commute
#    signature disappear on weekends?
def fig3():
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.2), sharey=True)
    for ax, grp in zip(axes, ["member", "casual"]):
        sub = valid[valid.member_casual == grp]
        m = (sub.groupby(["start_dow", "start_hour"], observed=True)
                .size().unstack(fill_value=0))
        m = m.div(m.values.sum()) * 100
        im = ax.imshow(m.values, aspect="auto", cmap="magma_r", origin="lower")
        ax.set_xticks(range(0, 24, 3))
        ax.set_xticklabels(range(0, 24, 3))
        ax.set_yticks(range(7))
        ax.set_yticklabels(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])
        ax.set_xlabel("hour of day")
        ax.grid(False)
        ax.set_title(f"{grp}", loc="left", fontsize=9.5)
        ax.axhline(4.5, color="white", lw=1.4)
    fig.colorbar(im, ax=axes, shrink=.85, label="% of group's trips")
    fig.suptitle("Weekday commute peaks vs weekend midday block",
                 x=.09, ha="left", fontsize=10)
    fig.savefig(f"{OUT}/fig3_hour_dow_heatmap.png", bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------- figure 4
# Q: how much does ridership collapse in winter, and who stops riding?
def fig4():
    order = ["202010", "202101", "202102"]
    tot = df.groupby("year_month", observed=True).size().reindex(order)
    cas = (df.groupby("year_month", observed=True)
             .member_casual.apply(lambda s: (s == "casual").mean())
             .reindex(order) * 100)

    fig, ax = plt.subplots(figsize=(7, 3.4))
    x = np.arange(len(order))
    ax.bar(x, tot / 1e6, color=MEMBER_C, width=.55)
    for i, v in enumerate(tot):
        ax.text(i, v / 1e6 + .07, f"{v/1e6:.2f}M", ha="center", fontsize=8.5,
                fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([MONTH_LABEL[m] for m in order])
    ax.set_ylabel("trips (millions)")
    ax.set_ylim(0, 2.7)

    ax2 = ax.twinx()
    ax2.plot(x, cas, color=CASUAL_C, lw=2, marker="o", ms=5)
    for i, v in enumerate(cas):
        ax2.text(i, v - 2.2, f"{v:.1f}%", fontsize=8.5, color=CASUAL_C, ha="center")
    ax2.set_ylabel("casual share of trips (%)", color=CASUAL_C)
    ax2.tick_params(axis="y", colors=CASUAL_C)
    ax2.set_ylim(0, 32)
    ax2.grid(False)

    ax.set_title("Winter cuts ridership by 72%, and casual riders leave first",
                 loc="left", fontsize=9.5)
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig4_seasonal.png", bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------- figure 5
# Q: which stations dominate, and do member and casual demand concentrate
#    in the same places?
def fig5():
    import clean_citibike as cc
    names = cc.canonical_station_names(valid)

    top = valid.start_station_id.value_counts().head(12).index[::-1]
    mem = valid[valid.member_casual == "member"].start_station_id.value_counts()
    cas = valid[valid.member_casual == "casual"].start_station_id.value_counts()

    labels = [names.get(s, s) for s in top]
    m = np.array([mem.get(s, 0) for s in top])
    c = np.array([cas.get(s, 0) for s in top])

    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    y = np.arange(len(top))
    ax.barh(y, m, color=MEMBER_C, label="member")
    ax.barh(y, c, left=m, color=CASUAL_C, label="casual")
    for i in range(len(top)):
        pct = c[i] / (m[i] + c[i]) * 100
        ax.text(m[i] + c[i] + 400, i, f"{pct:.0f}% casual",
                va="center", fontsize=7.6, color=CASUAL_C)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("trips started")
    ax.set_xlim(0, (m + c).max() * 1.28)
    ax.set_title("Busiest origin stations, split by rider type",
                 loc="left", fontsize=9.5)
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig5_top_stations.png", bbox_inches="tight")
    plt.close(fig)


for f in (fig1, fig2, fig3, fig4, fig5):
    f()
    print("built", f.__name__)
print("\nfigures in", OUT)
