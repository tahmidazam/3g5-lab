import matplotlib as mpl
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

mpl.use("pgf")
mpl.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False,
})

df_plain = pd.read_csv(
    filepath_or_buffer="data/s2/s2-plain.csv",
    names=["time", "temperature", "absorbance"],
    index_col=0
)

df_salted = pd.read_csv(
    filepath_or_buffer="data/s2/s2-salted.csv",
    names=["time", "temperature", "absorbance"],
    index_col=0
)


def plot(df: pd.DataFrame, filename: str, title: str, threshold_pct: int = 15) -> None:
    mask = (df.index >= 60) & (df.index <= 800)
    t_window = df.index[mask]

    # Horizontal line: take the mean of absorbance in the window
    mean_absorbance = df.loc[mask, "absorbance"].mean()

    # Evaluate "regression" for all times (just a constant)
    reg_plain_full = np.full_like(df.index, mean_absorbance, dtype=float)

    # Only consider times after 60
    mask_post60 = df.index > 60

    # Compute residuals instead of percentage error
    residuals = df["absorbance"] - reg_plain_full

    mask_baseline = (df.index >= 60) & (df.index <= 800)
    residuals_baseline = df["absorbance"][mask_baseline] - reg_plain_full[mask_baseline]
    threshold_residual = 3 * np.std(residuals_baseline)

    # Find the first index where the residual exceeds a threshold
    a_index_plain = residuals[mask_post60][np.abs(residuals[mask_post60]) > threshold_residual].index[0]

    # Get corresponding absorbance and temperature
    a_absorbance = df["absorbance"].loc[a_index_plain]
    a_temp = df["temperature"].loc[a_index_plain]

    fig, (temperature_ax, absorbance_ax, error_ax) = plt.subplots(3, 1, sharex=True,
                                                                  gridspec_kw={"height_ratios": [1, 3, 1]},
                                                                  figsize=(6, 8))

    temperature_ax.plot(df.index, df["temperature"], color="black")
    temperature_ax.axvline(a_index_plain, color="black", linestyle="dotted")
    temperature_ax.axhline(a_temp, color="black", linestyle="dotted")

    temperature_ax.annotate(
        rf"$t_g$ = {a_index_plain:.0f} s",
        xy=(a_index_plain, temperature_ax.get_ylim()[1]),
        xytext=(0, 5),
        textcoords="offset points",
        ha="left",
        va="bottom",
    )

    temperature_ax.annotate(
        rf"$T_g$ = {a_temp:.2f} $^\circ$C",
        xy=(df.index.max(), a_temp),
        xytext=(5, 0),
        textcoords="offset points",
        ha="left", va="center",
    )

    temperature_ax.set_ylabel(r"temperature, $T$ ($^\circ$C)")

    temperature_ax.set_xlim(0, 1810)
    temperature_ax.minorticks_on()
    temperature_ax.grid(which='minor', alpha=0.2)
    temperature_ax.grid(which='major', alpha=0.5)

    absorbance_ax.plot(df.index, df["absorbance"], color="black")
    absorbance_ax.plot(df.index, reg_plain_full, color="black", linestyle="dashed")
    absorbance_ax.axvline(a_index_plain, color="black", linestyle="dotted")
    absorbance_ax.axhline(a_absorbance, color="black", linestyle="dotted")

    absorbance_ax.annotate(
        f"$A_g$ = {a_absorbance:.2f}",
        xy=(df.index.max(), a_absorbance),
        xytext=(5, 4),
        textcoords="offset points",
        ha="left", va="center",
    )

    absorbance_ax.annotate(
        rf"baseline = {mean_absorbance:.2f}",
        xy=(df.index.max(), mean_absorbance),
        xytext=(5, -4),
        textcoords="offset points",
        ha="left", va="center",
    )

    absorbance_ax.set_ylabel(r"absorbance at 400 nm")
    absorbance_ax.set_xlim(0, 1810)
    absorbance_ax.set_ylim(0, None)
    absorbance_ax.minorticks_on()
    absorbance_ax.grid(which='minor', alpha=0.2)
    absorbance_ax.grid(which='major', alpha=0.5)

    error_ax.plot(df.index, residuals, color="black")
    error_ax.axhline(threshold_residual, color="black", linestyle="dotted")
    error_ax.axvline(a_index_plain, color="black", linestyle="dotted")

    error_ax.annotate(
        rf"threshold = {threshold_residual:.3f}",
        xy=(df.index.max(), threshold_residual),
        xytext=(5, 0),
        textcoords="offset points",
        ha="left", va="center",
    )

    error_ax.set_ylabel(r"residual")
    error_ax.set_xlabel(r"time, $t$ (s)")
    error_ax.set_xlim(0, 1810)
    error_ax.minorticks_on()
    error_ax.grid(which='minor', alpha=0.2)
    error_ax.grid(which='major', alpha=0.5)

    fig.suptitle(title)
    fig.tight_layout()

    fig.savefig(filename)


plot(df_plain, "plots/s2/s2_plain.pgf", "Plain agarose hydrogel absorbance and temperature curves")
plot(df_salted, "plots/s2/s2_salted.pgf", "Salted agarose hydrogel absorbance and temperature curves", threshold_pct=75)
