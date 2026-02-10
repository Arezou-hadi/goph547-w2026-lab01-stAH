import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import savemat, loadmat

from goph547lab01.gravity import (
    gravity_potential_point,
    gravity_effect_point,
)

# -------------------------------------------------
# Physics & Survey Generation
# -------------------------------------------------
def generate_survey_data(mm_sub, xm_sub, ym_sub, zm_sub):
    """Simulates vertical gravity effect survey at z = {0, 1, 100, 110} m"""
    print("Survey data file not found, running survey (this may take a moment)...")
    
    # survey grids
    x_5, y_5 = np.meshgrid(
        np.linspace(-100.0, 100.0, 41), 
        np.linspace(-100.0, 100.0, 41)
    )
    zp = [0.0, 1.0, 100.0, 110.0]

    U_5 = np.zeros((x_5.shape[0], x_5.shape[1], len(zp)))
    g_5 = np.zeros((x_5.shape[0], x_5.shape[1], len(zp)))
    xs = x_5[0, :]
    ys = y_5[:, 0]

    for km, (mm_k, xx_k, yy_k, zz_k) in enumerate(zip(mm_sub, xm_sub, ym_sub, zm_sub)):
        xm_k = [xx_k, yy_k, zz_k]
        for k, zz in enumerate(zp):
            for j, xx in enumerate(xs):
                for i, yy in enumerate(ys):
                    obs_pos = [xx, yy, zz]
                    U_5[i, j, k] += gravity_potential_point(obs_pos, xm_k, mm_k)
                    g_5[i, j, k] += gravity_effect_point(obs_pos, xm_k, mm_k)

    # Save to disk
    savemat(
        "examples/anomaly_survey_data.mat",
        mdict={"x_5": x_5, "y_5": y_5, "zp": zp, "g_5": g_5, "U_5": U_5},
    )
    print("Survey data generated and saved!")

# -------------------------------------------------
# Main Logic
# -------------------------------------------------
def main():
    # 1. Load Density Data
    if not os.path.exists("examples/anomaly_data.mat"):
        raise FileNotFoundError("Missing file anomaly_data.mat in examples/ directory")

    data = loadmat("examples/anomaly_data.mat")
    rho, xm, ym, zm = data["rho"], data["x"], data["y"], data["z"]

    # Statistics
    vcell = 2.0**3
    mm = rho * vcell
    mtot = np.sum(mm)
    xbar = np.sum(xm * mm) / mtot
    ybar = np.sum(ym * mm) / mtot
    zbar = np.sum(zm * mm) / mtot

    # 2. Extract Subregion
    kx_min, kx_max = 40, 60
    ky_min, ky_max = 44, 56
    kz_min, kz_max = 7, 13
    
    mm_sub = mm[ky_min:ky_max+1, kx_min:kx_max+1, kz_min:kz_max+1].flatten()
    xm_sub = xm[ky_min:ky_max+1, kx_min:kx_max+1, kz_min:kz_max+1].flatten()
    ym_sub = ym[ky_min:ky_max+1, kx_min:kx_max+1, kz_min:kz_max+1].flatten()
    zm_sub = zm[ky_min:ky_max+1, kx_min:kx_max+1, kz_min:kz_max+1].flatten()

    # 3. Plot Density Distribution (Fixing Colorbar Errors)
    plt.figure(figsize=(8, 9))
    r_min, r_max = 0.0, 0.6
    
    # Subplot Helper to reduce repetition and ensure 'mappable' is passed
    def plot_density_slice(pos, X, Y, data_slice, xlabel, ylabel, title, x_mark, y_mark, x_lim, y_lim, box_coords):
        ax = plt.subplot(3, 1, pos)
        cf = ax.contourf(X, Y, data_slice, cmap="viridis_r", levels=np.linspace(r_min, r_max, 200))
        ax.plot(x_mark, y_mark, "xk", markersize=5)
        ax.plot(box_coords[0], box_coords[1], "--k")
        cbar = plt.colorbar(cf, ax=ax, ticks=np.linspace(r_min, r_max, 7))
        cbar.set_label(r"$\bar{\rho}$ [$kg/m^3$]")
        ax.set_xlabel(xlabel); ax.set_ylabel(ylabel); ax.set_title(title)
        ax.set_xlim(x_lim); ax.set_ylim(y_lim)

    # X-Z View (Mean along Y)
    box_xz = ([xm[0, kx_min, 0], xm[0, kx_min, 0], xm[0, kx_max, 0], xm[0, kx_max, 0], xm[0, kx_min, 0]],
              [zm[0, 0, kz_min], zm[0, 0, kz_max], zm[0, 0, kz_max], zm[0, 0, kz_min], zm[0, 0, kz_min]])
    plot_density_slice(1, xm[0,:,:], zm[0,:,:], np.mean(rho, axis=0), "x [m]", "z [m]", "Mean density along y-axis", xbar, zbar, (-30, 30), (-20, 0), box_xz)

    # Y-Z View (Mean along X)
    box_yz = ([ym[ky_min, 0, 0], ym[ky_min, 0, 0], ym[ky_max, 0, 0], ym[ky_max, 0, 0], ym[ky_min, 0, 0]],
              [zm[0, 0, kz_min], zm[0, 0, kz_max], zm[0, 0, kz_max], zm[0, 0, kz_min], zm[0, 0, kz_min]])
    plot_density_slice(2, ym[:,0,:], zm[:,0,:], np.mean(rho, axis=1), "y [m]", "z [m]", "Mean density along x-axis", ybar, zbar, (-30, 30), (-20, 0), box_yz)

    # X-Y View (Mean along Z)
    box_xy = ([xm[0, kx_min, 0], xm[0, kx_min, 0], xm[0, kx_max, 0], xm[0, kx_max, 0], xm[0, kx_min, 0]],
              [ym[ky_min, 0, 0], ym[ky_max, 0, 0], ym[ky_max, 0, 0], ym[ky_min, 0, 0], ym[ky_min, 0, 0]])
    plot_density_slice(3, xm[:,:,0], ym[:,:,0], np.mean(rho, axis=2), "x [m]", "y [m]", "Mean density along z-axis", xbar, ybar, (-30, 30), (-30, 30), box_xy)

    plt.subplots_adjust(hspace=0.5)
    plt.savefig("examples/anomaly_mean_density.png", dpi=300)
    plt.close()

    # 4. Survey Data Handling
    if not os.path.exists("examples/anomaly_survey_data.mat"):
        generate_survey_data(mm_sub, xm_sub, ym_sub, zm_sub)

    survey = loadmat("examples/anomaly_survey_data.mat")
    x_5, y_5, zp = survey["x_5"], survey["y_5"], survey["zp"][0]
    g_5, U_5 = survey["g_5"], survey["U_5"]
    dx, dy = x_5[0, 1] - x_5[0, 0], y_5[1, 0] - y_5[0, 0]

    # 5. Compute Derivatives
    dgdz = np.stack(((g_5[:,:,1]-g_5[:,:,0])/(zp[1]-zp[0]), (g_5[:,:,3]-g_5[:,:,2])/(zp[3]-zp[2])), axis=-1)
    
    d2gdz2_0 = -( (g_5[2:, 1:-1, 0] - 2*g_5[1:-1, 1:-1, 0] + g_5[:-2, 1:-1, 0])/dy**2 + 
                  (g_5[1:-1, 2:, 0] - 2*g_5[1:-1, 1:-1, 0] + g_5[1:-1, :-2, 0])/dx**2 )
    d2gdz2_100 = -( (g_5[2:, 1:-1, 2] - 2*g_5[1:-1, 1:-1, 2] + g_5[:-2, 1:-1, 2])/dy**2 + 
                    (g_5[1:-1, 2:, 2] - 2*g_5[1:-1, 1:-1, 2] + g_5[1:-1, :-2, 2])/dx**2 )
    d2gdz2 = np.stack((d2gdz2_0, d2gdz2_100), axis=-1)

    # 6. Final Derivative Plotting (Fixing colorbars)
    plt.figure(figsize=(10, 10))
    
    def quick_plot(pos, X, Y, Z, levels, label, title_text):
        ax = plt.subplot(2, 2, pos)
        cf = ax.contourf(X, Y, Z, cmap="viridis", levels=levels)
        cbar = plt.colorbar(cf, ax=ax)
        cbar.set_label(label)
        ax.text(-90, 70, title_text, weight="bold", bbox=dict(facecolor="white"))
        ax.set_xlim(-100, 100); ax.set_ylim(-100, 100)

    quick_plot(1, x_5, y_5, dgdz[:,:,0], np.linspace(-1.4e-10, 0.2e-10, 50), r"$\partial g_z / \partial z$", "z = 0.0 m")
    quick_plot(3, x_5, y_5, dgdz[:,:,1], np.linspace(-2.8e-13, 0.0, 50), r"$\partial g_z / \partial z$", "z = 100.0 m")
    quick_plot(2, x_5[1:-1, 1:-1], y_5[1:-1, 1:-1], d2gdz2[:,:,0], np.linspace(-0.4e-11, 2.4e-11, 50), r"$\partial^2 g_z / \partial z^2$", "z = 0.0 m")
    quick_plot(4, x_5[1:-1, 1:-1], y_5[1:-1, 1:-1], d2gdz2[:,:,1], np.linspace(-1.0e-15, 8.0e-15, 50), r"$\partial^2 g_z / \partial z^2$", "z = 100.0 m")

    plt.tight_layout()
    plt.savefig("examples/anomaly_survey_derivatives.png", dpi=300)
    print("All plots saved successfully.")

if __name__ == "__main__":
    main()
