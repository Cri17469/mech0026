import argparse

import matplotlib

# Use a non-interactive backend so the script can run in headless environments
# (e.g., automated tests or remote servers without a display).
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import glob
import os

# =========================================================
# USER INPUT: remote stress sigma_infinity (largest applied stress)
# =========================================================
sigma_inf = 1000.0    # <<< CHANGE THIS to your applied σ∞

a = 1.0               # hole radius (normalized)

# =========================================================
# Angle mapping from filename
# =========================================================
theta_map = {
    "0": 0,
    "025": np.pi/4,
    "05": np.pi/2
}

# =========================================================
# Load Abaqus CSV files in the new 2-column format
# =========================================================
def load_abaqus_csv(folder_path="."):
    data = {"RR": {}, "TT": {}, "RT": {}}

    for filepath in glob.glob(os.path.join(folder_path, "*.csv")):
        filename = os.path.basename(filepath)

        # Detect stress component
        if "RR" in filename:
            comp = "RR"
        elif "TT" in filename:
            comp = "TT"
        elif "RT" in filename:
            comp = "RT"
        else:
            continue

        # Detect angle
        theta = None
        for key in theta_map:
            if filename.replace(".csv", "").endswith(key):
                theta = theta_map[key]
                break
        if theta is None:
            continue
        
        # Load CSV (2 columns). The provided files use whitespace delimiters
        # and include a blank line followed by a header row, so we skip the
        # first two lines and let NumPy split on whitespace.
        arr = np.genfromtxt(filepath, skip_header=2)

        # Guard against empty or malformed files
        if arr.size == 0:
            continue

        r_vals = arr[:, 0]               # first column: radial position
        stress_vals = arr[:, 1] / sigma_inf   # normalize

        data[comp][theta] = {
            "r": r_vals,
            "stress": stress_vals,
            "filename": filename
        }

    return data


# =========================================================
# Theoretical stress equations for σ1/σ2 = 0.5
# =========================================================
sigma2 = sigma_inf
sigma1 = 0.5 * sigma_inf

def sigma_rr(r, theta):
    return (0.5 * (sigma1 + sigma2) * (1 - (a*a/r/r))
            + 0.5 * (sigma1 - sigma2) * (1 - 4*(a*a/r/r) + 3*(a**4/r**4)) * np.cos(2*theta))

def sigma_tt(r, theta):
    return (0.5 * (sigma1 + sigma2) * (1 + (a*a/r/r))
            - 0.5 * (sigma1 - sigma2) * (1 + 3*(a**4/r**4)) * np.cos(2*theta))

def tau_rt(r, theta):
    return (-0.5 * (sigma1 - sigma2) * (1 + 2*(a*a/r/r) - 3*(a**4/r**4)) * np.sin(2*theta))


# r domain for theory
r_theory = np.linspace(a, 5*a, 500)

angles = [0, np.pi/4, np.pi/2]
angle_labels = {0: "θ = 0", np.pi/4: "θ = π/4", np.pi/2: "θ = π/2"}

# =========================================================
# Plotting function (theory + FE overlay)
# =========================================================
def plot_with_overlay(r_theory, y_theory, comp, theta, fe_data, show=True):
    plt.figure()
    plt.plot(r_theory/a, y_theory, label="Theory")

    # Overlay FE curve
    if theta in fe_data[comp]:
        r_fe = fe_data[comp][theta]["r"]
        s_fe = fe_data[comp][theta]["stress"]
        plt.plot(r_fe/a, s_fe, "--", label="Abaqus")

    plt.xlabel("r/a")
    plt.ylabel(f"{comp} / σ∞")
    plt.title(f"{comp} vs r/a  ({angle_labels[theta]})")
    plt.grid(True)
    plt.legend()

    if show:
        plt.show()
    else:
        plt.close()

# =========================================================
# Generate all 9 plots
# =========================================================
def generate_plots(fe_data, show=True):
    for theta in angles:
        y = sigma_rr(r_theory, theta) / sigma_inf
        plot_with_overlay(r_theory, y, "RR", theta, fe_data, show=show)

    for theta in angles:
        y = sigma_tt(r_theory, theta) / sigma_inf
        plot_with_overlay(r_theory, y, "TT", theta, fe_data, show=show)

    for theta in angles:
        y = tau_rt(r_theory, theta) / sigma_inf
        plot_with_overlay(r_theory, y, "RT", theta, fe_data, show=show)


def main():
    parser = argparse.ArgumentParser(description="Plot theoretical and FE stresses.")
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Generate plots without displaying them (useful for headless testing).",
    )
    parser.add_argument(
        "--folder",
        default=".",
        help="Folder containing the FE CSV files (default: current directory).",
    )
    args = parser.parse_args()

    abaqus_data = load_abaqus_csv(args.folder)
    print("Loaded:")
    for comp in abaqus_data:
        for theta in abaqus_data[comp]:
            print(f"  {abaqus_data[comp][theta]['filename']} (θ={theta})")

    generate_plots(abaqus_data, show=not args.no_show)


if __name__ == "__main__":
    main()
