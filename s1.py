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

df = pd.read_csv(
    filepath_or_buffer="data/s1/s1.csv",
    names=["time, $t$ (s)", "T_plain", "T_salted"],
    index_col=0
)

t_a_plain = df["T_plain"].iloc[-1]
t_a_salted = df["T_salted"].iloc[-1]

df["delta_T_plain"] = df["T_plain"] - t_a_plain
df["delta_T_salted"] = df["T_salted"] - t_a_salted

df["log_delta_T_plain"] = np.log(df["delta_T_plain"])
df["log_delta_T_salted"] = np.log(df["delta_T_salted"])

mask = (df.index >= 60) & (df.index <= 150)
t_window = df.index[mask]

coeffs_plain = np.polyfit(t_window, df.loc[mask, "log_delta_T_plain"], 1)
coeffs_salted = np.polyfit(t_window, df.loc[mask, "log_delta_T_salted"], 1)

reg_plain_full = np.polyval(coeffs_plain, df.index)
reg_salted_full = np.polyval(coeffs_salted, df.index)

mask_post60 = df.index > 60


def plot(index: pd.Index, log_delta_t_series: pd.Series, t_series: pd.Series, title: str, filename: str,
         coeffs: np.ndarray):
    residuals = log_delta_t_series - reg_plain_full
    residuals_baseline = log_delta_t_series[mask] - reg_plain_full[mask]
    threshold_residual = 3 * np.std(residuals_baseline)
    tg_index = residuals[mask_post60][np.abs(residuals[mask_post60]) > threshold_residual].index[0]
    tg = t_series.loc[tg_index]

    fig, (log_temp_ax, residual_ax) = plt.subplots(2, 1, sharex=True, figsize=(6, 4),
                                                   gridspec_kw={"height_ratios": [3, 1]})

    log_temp_ax.plot(index, log_delta_t_series, color="black")
    log_temp_ax.plot(index, reg_plain_full, color="black", linestyle="dashed")
    log_temp_ax.axvline(tg_index, color="black", linestyle="dotted")
    log_temp_ax.set_ylabel(r"$\ln \Delta T$ ($^\circ$C)")
    log_temp_ax.minorticks_on()
    log_temp_ax.grid(which='minor', alpha=0.2)
    log_temp_ax.grid(which='major', alpha=0.5)
    log_temp_ax.set_xlim(0, 1800)

    tg_label = f"$t_g$ = {tg_index:.0f} s, $T_g = {tg:.3f}~^\\circ$C"

    log_temp_ax.annotate(
        tg_label,
        xy=(tg_index, log_temp_ax.get_ylim()[1]),
        xytext=(0, 5),
        textcoords="offset points",
        ha="left",
        va="bottom",
    )

    slope, intercept = coeffs
    eq_text = f"$y = {slope:.4f} x + {intercept:.3f}$"

    log_temp_ax.text(
        0.95, 0.95,
        eq_text,
        ha="right",
        va="top",
        transform=log_temp_ax.transAxes,
    )

    residual_ax.plot(df.index, residuals, color="black")
    residual_ax.axhline(threshold_residual, color="black", linestyle="dotted")
    residual_ax.axvline(tg_index, color="black", linestyle="dotted")
    residual_ax.set_xlabel(r"time, $t$ (s)")
    residual_ax.set_ylabel("residual ($^\circ$C)")
    residual_ax.minorticks_on()
    residual_ax.grid(which='minor', alpha=0.2)
    residual_ax.grid(which='major', alpha=0.5)
    residual_ax.set_xlim(0, 1800)

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(filename)

    return tg


plain_tg = plot(
    index=df.index,
    log_delta_t_series=df["log_delta_T_plain"],
    t_series=df["T_plain"],
    title="Plain agarose gel cooling curve",
    filename="plots/s1/s1_plain.pgf",
    coeffs=coeffs_plain
)

salted_tg = plot(
    index=df.index,
    log_delta_t_series=df["log_delta_T_salted"],
    t_series=df["T_salted"],
    title="Salted agarose gel cooling curve",
    filename="plots/s1/s1_salted.pgf",
    coeffs=coeffs_salted
)

model_parameters = pd.DataFrame(
    [plain_tg, salted_tg],
    columns=["$T_g$ ($^\circ$C)"],
    index=["Plain agarose", "Salted agarose"])

with open("tables/s1_tg.tex", "w") as f:
    f.write(model_parameters.to_latex(
        formatters=[lambda x: f"{x:.3}"],
        header=["$T_g$ ($^\circ$C)"],
        index=True
    ))
