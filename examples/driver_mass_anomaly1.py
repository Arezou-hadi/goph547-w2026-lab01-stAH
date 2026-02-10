import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import os

# -------------------------------------------------
# ۱. توابع فیزیکی (Physics)
# -------------------------------------------------
G = 6.67430e-11

def gravity_effect_point(obs, src, m):
    """محاسبه شتاب گرانش عمودی (gz) [cite: 71]"""
    r_vec = np.array(obs) - np.array(src)
    r = np.linalg.norm(r_vec)
    if r == 0: return 0
    return (G * m * abs(r_vec[2])) / (r**3)

def compute_second_vertical_derivative(g_z, spacing):
    """محاسبه مشتق دوم عمودی با استفاده از معادله لاپلاس [cite: 78, 80]"""
    d2g_dx2 = np.zeros_like(g_z)
    d2g_dy2 = np.zeros_like(g_z)
    # تفاضل محدود مرکزی
    d2g_dx2[:, 1:-1] = (g_z[:, 2:] - 2*g_z[:, 1:-1] + g_z[:, :-2]) / (spacing**2)
    d2g_dy2[1:-1, :] = (g_z[2:, :] - 2*g_z[1:-1, :] + g_z[:-2, :]) / (spacing**2)
    return -(d2g_dx2 + d2g_dy2)

# -------------------------------------------------
# ۲. بدنه اصلی تحلیل ناهنجاری
# -------------------------------------------------
def main():
    # اصلاح مسیر فایل (Path Handling) 
    script_dir = os.path.dirname(__file__)
    file_path = os.path.join(script_dir, 'anomaly_data.mat')
    
    if not os.path.exists(file_path):
        print(f"Error: Could not find {file_path}.")
        return

    # بارگذاری داده‌ها [cite: 38]
    data = sio.loadmat(file_path)
    x, y, z, rho = data['x'], data['y'], data['z'], data['rho']
    
    # گام ۱: محاسبات جرمی و مرکز ثقل 
    vol_cell = 8.0 # هر سلول 2m x 2m x 2m است 
    m_cells = rho * vol_cell
    total_mass = np.sum(m_cells)
    
    x_bar = np.sum(m_cells * x) / total_mass
    y_bar = np.sum(m_cells * y) / total_mass
    z_bar = np.sum(m_cells * z) / total_mass
    
    print(f"Total Mass: {total_mass:.4e} kg")
    print(f"Barycentre: ({x_bar:.2f}, {y_bar:.2f}, {z_bar:.2f})")
    print(f"Max Density: {np.max(rho):.2f} kg/m^3") 

    # گام ۳: رسم مقاطع میانگین چگالی 
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    planes = [
        (np.mean(rho, axis=2), x[:,:,0], y[:,:,0], 'x [m]', 'y [m]', 'XY Plane (axis=2)'), 
        (np.mean(rho, axis=1), x[:,0,:], z[:,0,:], 'x [m]', 'z [m]', 'XZ Plane (axis=1)'), 
        (np.mean(rho, axis=0), y[0,:,:], z[0,:,:], 'y [m]', 'z [m]', 'YZ Plane (axis=0)')   
    ]
    
    for i, (p_data, x_g, y_g, xl, yl, title) in enumerate(planes):
        im = axes[i].contourf(x_g, y_g, p_data, cmap='viridis')
        axes[i].plot(x_bar if 'x' in xl else y_bar, z_bar if 'z' in yl else y_bar, 'xk', markersize=3) 
        axes[i].set_title(title)
        axes[i].set_xlabel(xl)
        axes[i].set_ylabel(yl)
        plt.colorbar(im, ax=axes[i], label=r'$\rho$ [kg/m$^3$]')
    plt.tight_layout()
    plt.show()

    # گام ۴: مدل‌سازی پیشرو (Forward Modelling) 
    spacing = 5.0
    x_s = np.arange(-100, 101, spacing)
    y_s = np.arange(-100, 101, spacing)
    X, Y = np.meshgrid(x_s, y_s)
    
    # فیلتر کردن نقاط برای افزایش سرعت
    mask = rho > 0.1
    m_act, x_act, y_act, z_act = m_cells[mask], x[mask], y[mask], z[mask]
    
    def get_gz(elev):
        gz = np.zeros_like(X)
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                obs = [X[i,j], Y[i,j], elev]
                # فاصله اقلیدسی
                dist = np.sqrt((x_act-obs[0])**2 + (y_act-obs[1])**2 + (z_act-obs[2])**2)
                gz[i,j] = np.sum((G * m_act * abs(z_act-elev)) / (dist**3))
        return gz

    print("Calculating gravity fields at different elevations...")
    gz_0 = get_gz(0.0) # 
    gz_100 = get_gz(100.0) # 
    
    # گام ۶: مشتق دوم عمودی (Step 6) 
    d2gz_dz2_0 = compute_second_vertical_derivative(gz_0, spacing)
    
    plt.figure()
    plt.contourf(X, Y, d2gz_dz2_0, cmap='RdBu_r', levels=50)
    plt.title(r"Second Vertical Derivative $\partial^2 g_z / \partial z^2$ at z=0m")
    plt.colorbar(label=r"$s^{-2}$")
    plt.show()

if __name__ == "__main__":
    main()