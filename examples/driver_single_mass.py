import numpy as np
import matplotlib.pyplot as plt
from goph547lab01.gravity import gravity_potential_point, gravity_effect_point

def run_simulation(grid_spacing):
    # پارامترهای مسئله طبق دستورالعمل 
    
    m = 1.0e7  # جرم: 10^7 کیلوگرم
    xm = np.array([0, 0, -10])  # مرکز جرم در مختصات (0, 0, -10)
    
    # ایجاد شبکه (Grid) از -100 تا +100 متر 
    x_range = np.arange(-100, 100 + grid_spacing, grid_spacing)
    y_range = np.arange(-100, 100 + grid_spacing, grid_spacing)
    X, Y = np.meshgrid(x_range, y_range)
    
    # ارتفاع‌های مورد نظر برای نقشه برداری 
    z_levels = [0, 10, 100]
    
    fig, axes = plt.subplots(3, 2, figsize=(12, 15), constrained_layout=True)
    fig.suptitle(f'Gravity Survey Simulation (Spacing: {grid_spacing}m)', fontsize=16)

    # ذخیره مقادیر برای یکسان‌سازی محدوده رنگ‌ها (Colorbar Limits) 
    u_min, u_max = 0, 7e-5
    gz_min, gz_max = -1e-8, 7e-6

    for i, z in enumerate(z_levels):
        U_grid = np.zeros_like(X)
        GZ_grid = np.zeros_like(X)
        
        # محاسبه مقادیر برای هر نقطه در شبکه
        for row in range(X.shape[0]):
            for col in range(X.shape[1]):
                survey_point = np.array([X[row, col], Y[row, col], z])
                U_grid[row, col] = gravity_potential_point(survey_point, xm, m)
                GZ_grid[row, col] = gravity_effect_point(survey_point, xm, m)
        
        # رسم پتانسیل (U) - ستون اول 
        
        cf1 = axes[i, 0].contourf(X, Y, U_grid, levels=20, cmap='viridis', vmin=u_min, vmax=u_max)
        axes[i, 0].plot(X, Y, 'xk', markersize=2, alpha=0.3) # نمایش نقاط نمونه‌برداری 
        axes[i, 0].set_title(f'Potential (U) at z={z}m')
        plt.colorbar(cf1, ax=axes[i, 0], label='U [J/kg]')
        
        # رسم اثر گرانشی (gz) - ستون دوم 
        
        
        cf2 = axes[i, 1].contourf(X, Y, GZ_grid, levels=20, cmap='plasma', vmin=gz_min, vmax=gz_max)
        axes[i, 1].plot(X, Y, 'xk', markersize=2, alpha=0.3)
        axes[i, 1].set_title(f'Gravity Effect (gz) at z={z}m')
        plt.colorbar(cf2, ax=axes[i, 1], label='gz [m/s^2]')

    # تنظیم لیبل محورها 
    
    for ax in axes.flat:
        ax.set_xlabel('X [m]')
        ax.set_ylabel('Y [m]')

    plt.savefig(f'single_mass_spacing_{grid_spacing}.png')
    plt.show()

if __name__ == "__main__":
    # اجرای شبیه‌سازی برای هر دو فاصله شبکه‌بندی 
    
    run_simulation(5.0)
    run_simulation(25.0)