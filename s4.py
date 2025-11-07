import matplotlib as mpl
import pandas as pd
from matplotlib import pyplot as plt

mpl.use("pgf")
mpl.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False,
})

df = pd.read_csv(
    filepath_or_buffer="data/s4/s4.csv",
    names=["time, $t$ (s)", "w_plain", "w_salted", "w_gelatin"],
    index_col=0
)

with open("tables/s4.tex", "w") as f:
    f.write(df.to_latex(formatters=[lambda x: f"{x:.1f}"] * len(df.columns),
                        header=["\\% swelling, plain", "\\% swelling, salted", "\\% swelling, gelatin"], index=True))

fig, ax = plt.subplots()

ax.plot(df.index, df["w_plain"], label="Plain agarose", color="black", marker="x", linestyle="solid")
ax.plot(df.index, df["w_salted"], label="Salted agarose", color="black", marker="x", linestyle="dashed")
ax.plot(df.index, df["w_gelatin"], label="Gelatin", color="black", marker="x", linestyle="dotted")

ax.set_xlim(0, 3600)
ax.set_ylim(0, None)

ax.set_xlabel(r"time, $t$ (s)")
ax.set_ylabel(r"Percentage swelling")
ax.legend()
ax.minorticks_on()
ax.grid(which='minor', alpha=0.2)
ax.grid(which='major', alpha=0.5)

ax.set_title(r"Percentage of swelling against immersion time")
fig.tight_layout()

fig.savefig("plots/s4/s4.pgf")

exit(0)
