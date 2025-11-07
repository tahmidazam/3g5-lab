import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

from s1 import model_parameters

mpl.use("pgf")
mpl.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False,
})


def standard_form(num, precision=3):
    s = f"{num:.{precision}e}"
    mantissa, exp = s.split("e")
    exp = int(exp)
    return rf"{mantissa} \times 10^{{{exp}}}"


def maxwell_kevin_voigt_model(x, e_1, e_2, eta_1, eta_2):
    return 1 / e_1 + (1 / e_2) * (1 - np.exp(-e_2 * x / eta_2)) + x / eta_1


def plot_s3(filename: str, title: str):
    df = pd.read_csv(filename)
    x = df.iloc[:, 0].to_numpy()
    y = df.iloc[:, 1].to_numpy()

    popt, pcov = curve_fit(maxwell_kevin_voigt_model, x, y)
    e_1, e_2, eta_1, eta_2 = popt
    y_fit = maxwell_kevin_voigt_model(x, *popt)

    m = 1 / eta_1
    c = 1 / e_2 + 1 / e_1
    y_linear = m * x + c

    fig, ax = plt.subplots(figsize=(6, 4))

    ax.plot(x, y_linear, linestyle="--", linewidth=1, color="black")
    ax.axhline(y=1 / e_1, linewidth=1, linestyle="--", color="black")
    ax.axhline(y=1 / e_2 + 1 / e_1, linewidth=1, linestyle="--", color="black")
    ax.plot(x, y_fit, color="black", linestyle="--")
    ax.plot(x, y, marker="x", color="black")

    ax.minorticks_on()
    ax.grid(which='minor', alpha=0.2)
    ax.grid(which='major', alpha=0.5)

    ax.annotate(
        rf"$\frac{{1}}{{E_1}} = {standard_form(1 / e_1)}$ Pa$^{{-1}}$",
        xy=(x.max(), 1 / e_1),
        xytext=(10, 0),
        textcoords="offset points",
        ha="left", va="center",
    )
    ax.annotate(
        rf"$\frac{{1}}{{E_1}} + \frac{{1}}{{E_2}} = {standard_form(1 / e_1 + 1 / e_2)}$ Pa$^{{-1}}$",
        xy=(x.max(), 1 / e_1 + 1 / e_2),
        xytext=(10, 0),
        textcoords="offset points",
        ha="left", va="center",
    )
    ax.annotate(
        rf"$m=\frac{{1}}{{\eta_1}} = {standard_form(1 / eta_1)}$ Pa$^{{-1}}$s$^{{-1}}$",
        xy=(x.max(), y_linear[-1]),
        xytext=(10, 0),
        textcoords="offset points",
        ha="left", va="center",
    )

    ax.set_title(title)
    ax.set_xlim(0, 360)
    ax.set_ylim(bottom=0)
    ax.set_xlabel(r"time, $t$ (s)")
    ax.set_ylabel(r"creep compliance function, $J(t)$ (Pa$^{-1}$)")

    fig.tight_layout()

    fig.savefig(filename.replace('.csv', '.pgf').replace("data/", "plots/"), dpi=300)

    return (e_1, e_2, eta_1, eta_2), x, y


s3_plain_180_parameters, s3_plain_180_t_values, s3_plain_180_values = plot_s3(
    filename="data/s3/plain-180.csv",
    title=r"Plain agarose hydrogel under 385 g load",
)
s3_plain_720_parameters, s3_plain_720_t_values, s3_plain_720_values = plot_s3(
    filename="data/s3/plain-720.csv",
    title=r"Plain agarose hydrogel under 925 g load",
)
s3_salted_180_parameters, s3_salted_180_t_values, s3_salted_180_values = plot_s3(
    filename="data/s3/salted-180.csv",
    title=r"Salted agarose hydrogel under 385 g load",
)
s3_salted_720_parameters, s3_salted_720_t_values, s3_salted_720_values = plot_s3(
    filename="data/s3/salted-720.csv",
    title=r"Salted agarose hydrogel under 925 g load",
)

values_data = [
    s3_plain_180_values,
    s3_plain_720_values,
    s3_salted_180_values,
    s3_salted_720_values
]
df_values = pd.DataFrame(
    data=np.array(values_data).T,
    columns=[
        "plain, 385 g load",
        "plain, 925 g load",
        "salted, 385 g load",
        "salted, 925 g load"
    ],
    index=s3_plain_180_t_values
)
df_values.index.name = r"time, $t$ (s)"

df_model_parameters = pd.DataFrame(
    data=[
        s3_plain_180_parameters,
        s3_plain_720_parameters,
        s3_salted_180_parameters,
        s3_salted_720_parameters
    ],
    columns=[
        "e_1",
        "e_2",
        "eta_1",
        "eta_2"
    ],
    index=[
        "Plain, 385 g load",
        "Plain, 925 g load",
        "Salted, 385 g load",
        "Salted, 925 g load"
    ]
)


def plot_fitted_curves_right_labels(model_parameters: pd.DataFrame, t_max: float = 360):
    t = np.linspace(0, t_max, 500)  # fine time points for smooth curves

    fig, ax = plt.subplots(figsize=(6, 4))

    # Define colors and line styles for each condition
    style_map = {
        "Plain, 385 g load": {"color": "black", "linestyle": "-"},
        "Plain, 925 g load": {"color": "orange", "linestyle": "-"},
        "Salted, 385 g load": {"color": "black", "linestyle": ":"},
        "Salted, 925 g load": {"color": "orange", "linestyle": ":"},
    }

    for idx, row in model_parameters.iterrows():
        e_1, e_2, eta_1, eta_2 = row
        J_t = maxwell_kevin_voigt_model(t, e_1, e_2, eta_1, eta_2)
        ax.plot(t, J_t, label=idx, **style_map.get(idx, {}))
        # ax.text(t_max + 2, J_t[-1], idx, color=style_map[idx]["color"],
        #         va="center", fontsize=9)

    ax.set_xlim(0, t_max)
    ax.set_ylim(bottom=0)
    ax.set_xlabel(r"time, $t$ (s)")
    ax.set_ylabel(r"creep compliance, $J(t)$ (Pa$^{-1}$)")
    ax.set_title("4-component Burgers model")
    legend = ax.legend()

    frame = legend.get_frame()
    frame.set_edgecolor('black')

    ax.minorticks_on()
    ax.grid(which='minor', alpha=0.2)
    ax.grid(which='major', alpha=0.5)
    fig.tight_layout()

    fig.savefig("plots/s3/fitted.pgf", dpi=300)


plot_fitted_curves_right_labels(df_model_parameters, t_max=360)

with open("tables/s3-model-parameters.tex", "w") as f:
    f.write(df_model_parameters.to_latex(
        formatters=[lambda x: f"{x / 1_000_000:.3f}", lambda x: f"{x / 1_000_000:.3f}",
                    lambda x: f"{x / 1_000_000_000:.3f}",
                    lambda x: f"{x / 1_000_000:.3f}"],
        header=["$E_1$ (kPa)", "$E_2$ (MPa)", "$\\eta_1$ (GPa s)", "$\\eta_2$ (MPa s)"],
        index=True
    ))

with open("tables/s3_raw_values_new.tex", "w") as f:
    df = pd.read_csv("data/s3/s3_raw.csv")
    df.set_index("time, $t$ (s)", inplace=True)
    f.write(df.to_latex(
        formatters=[lambda x: f"{x:.3f}"] * len(df.columns),
        header=[
            "plain, 385 g",
            "plain, 925 g",
            "salted, 385 g",
            "salted, 925 g"
        ],
        index=True
    ))

with open("tables/s3_j_values_new.tex", "w") as f:
    f.write(df_values.to_latex(
        formatters=[lambda x: f"{x * 1_000_000:.2f}"] * len(df.columns),
        header=[
            "plain, 385 g",
            "plain, 925 g",
            "salted, 385 g",
            "salted, 925 g"
        ],
        index=True
    ))

exit(0)
