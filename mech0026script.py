import numpy as np
import matplotlib.pyplot as plt
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
        
        # Load CSV (2 columns)
        arr = np.genfromtxt(filepath, delimiter=",", skip_header=1)
        r_vals = arr[:, 0]               # first column: radial position
        stress_vals = arr[:, 1] / sigma_inf   # normalize

        data[comp][theta] = {
            "r": r_vals,
            "stress": stress_vals,
            "filename": filename
        }

    return data


# Load FE CSV
abaqus_data = load_abaqus_csv(".")
print("Loaded:")
for comp in abaqus_data:
    for theta in abaqus_data[comp]:
        print(f"  {abaqus_data[comp][theta]['filename']} (θ={theta})")

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
def plot_with_overlay(r_theory, y_theory, comp, theta):
    plt.figure()
    plt.plot(r_theory/a, y_theory, label="Theory")

    # Overlay FE curve
    if theta in abaqus_data[comp]:
        r_fe = abaqus_data[comp][theta]["r"]
        s_fe = abaqus_data[comp][theta]["stress"]
        plt.plot(r_fe/a, s_fe, "--", label="Abaqus")

    plt.xlabel("r/a")
    plt.ylabel(f"{comp} / σ∞")
    plt.title(f"{comp} vs r/a  ({angle_labels[theta]})")
    plt.grid(True)
    plt.legend()
    plt.show()

# =========================================================
# Generate all 9 plots
# =========================================================
for theta in angles:
    y = sigma_rr(r_theory, theta) / sigma_inf
    plot_with_overlay(r_theory, y, "RR", theta)

for theta in angles:
    y = sigma_tt(r_theory, theta) / sigma_inf
    plot_with_overlay(r_theory, y, "TT", theta)

for theta in angles:
    y = tau_rt(r_theory, theta) / sigma_inf
    plot_with_overlay(r_theory, y, "RT", theta)
