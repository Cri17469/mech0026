import numpy as np
import matplotlib.pyplot as plt
import glob
import os

# =========================================================
# USER INPUT
# =========================================================
sigma_inf = 2000.0     # applied stress σ∞
a = 0.2                # hole radius = 0.2 m   <<< IMPORTANT

# =========================================================
# Angle mapping based on filename suffix
# =========================================================
theta_map = {
    "0": 0,
    "025": np.pi/4,
    "05": np.pi/2
}

# =========================================================
# Load Abaqus CSV files (whitespace separated)
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

        # Detect angle tag
        theta = None
        for key in theta_map:
            if filename.replace(".csv", "").endswith(key):
                theta = theta_map[key]
                break
        if theta is None:
            continue

        # Load whitespace-separated CSV
        arr = np.genfromtxt(
            filepath,
            delimiter=None,
            comments=None,
            autostrip=True
        )

        # Remove blank rows
        arr = arr[~np.isnan(arr).any(axis=1)]
        arr = np.atleast_2d(arr)

        r_vals = arr[:, 0]             # this is Δr from hole surface
        stress_vals = arr[:, 1] / sigma_inf

        data[comp][theta] = {
            "r": r_vals,
            "stress": stress_vals,
            "filename": filename
        }

    return data


# Load FE data
abaqus_data = load_abaqus_csv(".")
print("Loaded:")
for comp in abaqus_data:
    for theta in abaqus_data[comp]:
        print(f"  {abaqus_data[comp][theta]['filename']}   (θ={theta})")

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


# r domain (but we restrict later)
r_theory = np.linspace(a, 5*a, 500)

angles = [0, np.pi/4, np.pi/2]
angle_labels = {0: "θ = 0", np.pi/4: "θ = π/4", np.pi/2: "θ = π/2"}

# =========================================================
# Individual plotting (1 angle per plot → 9 plots total)
# =========================================================
def plot_component_single_angle(comp, theory_func):
    # hole radius
    r0 = a

    for theta in angles:

        # ---- Set r-range depending on angle ----
        if theta == np.pi/4:
            r_min = r0
            r_max = 2* r0 * np.sqrt(2)    # 0.2 → 0.282842...
        else:
            r_min = r0
            r_max = 0.4

        # restrict theory domain
        r_th = r_theory[(r_theory >= r_min) & (r_theory <= r_max)]

        plt.figure()

        # ---- THEORY (θ2 = θ1 - π/2) ----
        theta_theory = theta - np.pi/2
        y_theory = theory_func(r_th, theta_theory) / sigma_inf
        plt.plot(r_th, y_theory, label=f"Theory {angle_labels[theta]} (θ₂ = θ - π/2)")

        # ---- FE DATA ----
        fe_curve = abaqus_data.get(comp, {}).get(theta)
        if fe_curve:
            # FE actual radial coordinate = a + Δr
            r_fe = a + fe_curve["r"]
            s_fe = fe_curve["stress"]

            mask = (r_fe >= r_min) & (r_fe <= r_max)
            plt.plot(r_fe[mask], s_fe[mask], "--", label=f"Abaqus {angle_labels[theta]}")

        plt.xlabel("r (m)")
        plt.ylabel(f"{comp} / σ∞")
        plt.title(f"{comp} vs r   ({angle_labels[theta]})")
        plt.grid(True)
        plt.legend()
        plt.show()


# =========================================================
# Run plots
# =========================================================
plot_component_single_angle("RR", sigma_rr)
plot_component_single_angle("TT", sigma_tt)
plot_component_single_angle("RT", tau_rt)
