import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat, savemat

# فرض بر این است که پکیج شما نصب شده یا در PYTHONPATH قرار دارد
from goph547lab01.gravity import (
    gravity_potential_point,
    gravity_effect_point,
)

# -------------------------------------------------
# Core computation utilities
# -------------------------------------------------
def compute_gravity_fields(xg, yg, zp, masses, xm):
    nx, ny = xg.shape
    nz = len(zp)

    U = np.zeros((nx, ny, nz))
    g = np.zeros((nx, ny, nz))

    xs = xg[0, :]
    ys = yg[:, 0]

    for k, z in enumerate(zp):
        for i, y in enumerate(ys):
            for j, x in enumerate(xs):
                obs = [x, y, z]
                # اصل برهم‌نهی خطی برای محاسبه اثر کل مجموعه جرم‌ها 
                for m, xsrc in zip(masses, xm):
                    U[i, j, k] += gravity_potential_point(obs, xsrc, m)
                    g[i, j, k] += gravity_effect_point(obs, xsrc, m)

    return U, g

def make_grid(npts):
    x = np.linspace(-100.0, 100.0, npts)
    return np.meshgrid(x, x)

# -------------------------------------------------
# Mass anomaly generation 
# -------------------------------------------------
def generate_mass_anomaly_sets():
    """
    تولید ۳ مجموعه جرم تصادفی با محدودیت مجموع جرم و مرکز ثقل یکسان [cite: 8, 9]
    """
    mtot = 1.0e7  # مجموع جرم کل 
    m_mean = mtot / 5  # میانگین جرم هر نقطه 
    m_std = mtot / 100  # انحراف معیار جرم 

    # میانگین و انحراف معیار مکان‌ها 
    xbar_target = np.array([0.0, 0.0, -10.0]) 
    xsig = np.array([20.0, 20.0, 2.0])

    if not os.path.exists("examples"):
        os.makedirs("examples")

    for k in range(3):
        valid_set = False
        while not valid_set:
            # ۱. تولید ۴ جرم اول به صورت تصادفی 
            m = np.random.normal(m_mean, m_std, (5, 1))
            xm = np.zeros((5, 3))
            
            for i in range(3):
                xm[:-1, i] = np.random.normal(xbar_target[i], xsig[i], 4)
            
            # ۲. محاسبه جرم پنجم برای حفظ مجموع جرم 
            m[-1] = mtot - np.sum(m[:-1])
            
            # ۳. محاسبه مکان پنجم برای حفظ مرکز ثقل 
            for i in range(3):
                xm[-1, i] = (xbar_target[i] * mtot - np.dot(m[:-1, 0], xm[:-1, i])) / m[-1, 0]
            
            # ۴. بررسی شرط z <= -1 برای تمام جرم‌ها 
            if np.all(xm[:, 2] <= -1.0):
                valid_set = True
                savemat(f"examples/mass_set_{k}.mat", {"m": m, "xm": xm})

# -------------------------------------------------
# Plotting
# -------------------------------------------------
def plot_gravity_potential_and_effect(kk, x25, y25, U25, g25, x5, y5, U5, g5):
    def plot_block(x, y, U, g, grid):
        fig, axes = plt.subplots(3, 2, figsize=(10, 12), constrained_layout=True)
        
        # محدوده‌های رنگی مشابه پارت A برای مقایسه بهتر 
        Umin, Umax = 0.0, 8.0e-5
        gmin, gmax = 0.0, 7.0e-6
        z_elevations = [0.0, 10.0, 100.0]

        fig.suptitle(
            f"Mass Set {kk} | mtot = 1.0e7 kg | Center of Mass z = -10 m\nGrid Spacing = {grid} m",
            weight="bold", fontsize=14
        )

        for i, z_val in enumerate(z_elevations):
            # Potential Plot
            axU = axes[i, 0]
            cfU = axU.contourf(x, y, U[:, :, i], cmap="viridis_r", levels=np.linspace(Umin, Umax, 50))
            fig.colorbar(cfU, ax=axU).set_label(r"U [$m^2/s^2$]")
            axU.set_title(f"Potential (U) at z = {z_val} m")
            axU.set_ylabel("y [m]")
            if grid == 25.0: axU.plot(x, y, "xk", markersize=2) # نمایش نقاط شبکه درشت 

            # Gravity Effect Plot
            axg = axes[i, 1]
            cfg = axg.contourf(x, y, g[:, :, i], cmap="magma", levels=np.linspace(gmin, gmax, 50))
            fig.colorbar(cfg, ax=axg).set_label(r"$g_z$ [$m/s^2$]")
            axg.set_title(f"Gravity Effect ($g_z$) at z = {z_val} m")
            if grid == 25.0: axg.plot(x, y, "xk", markersize=2)

        axes[2, 0].set_xlabel("x [m]")
        axes[2, 1].set_xlabel("x [m]")
        return fig

    # ذخیره نمودارها برای هر دو فاصله شبکه‌ای 
    fig25 = plot_block(x25, y25, U25, g25, 25.0)
    fig25.savefig(f"examples/multi_mass_grid_25_set_{kk}.png", dpi=300)
    plt.close(fig25)

    fig5 = plot_block(x5, y5, U5, g5, 5.0)
    fig5.savefig(f"examples/multi_mass_grid_5_set_{kk}.png", dpi=300)
    plt.close(fig5)

# -------------------------------------------------
# Main workflow
# -------------------------------------------------
def main():
    # تولید داده‌ها در صورت عدم وجود
    generate_mass_anomaly_sets()

    x25, y25 = make_grid(9)   # شبکه با فاصله ۲۵ متر 
    x5, y5 = make_grid(41)    # شبکه با فاصله ۵ متر 
    zp = [0.0, 10.0, 100.0]   # ارتفاعات مورد نظر 

    for k in range(3):
        print(f"--- Processing Mass Set {k} ---")
        data = loadmat(f"examples/mass_set_{k}.mat")
        m = data["m"][:, 0]
        xm = data["xm"]

        # تایید محدودیت‌ها در خروجی 
        print(f"Verified Total Mass: {np.sum(m):.2e} kg")
        print(f"Verified Center of Mass: {np.dot(m, xm) / np.sum(m)}")
        print(f"All masses below -1m: {np.all(xm[:, 2] <= -1.0)}")

        # محاسبات فیلدها
        U25, g25 = compute_gravity_fields(x25, y25, zp, m, xm)
        U5, g5 = compute_gravity_fields(x5, y5, zp, m, xm)

        # رسم و ذخیره
        plot_gravity_potential_and_effect(k, x25, y25, U25, g25, x5, y5, U5, g5)

if __name__ == "__main__":
    main()