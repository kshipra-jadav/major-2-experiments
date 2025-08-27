import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from matplotlib.widgets import Slider, Button
import pandas as pd
from pathlib import Path

def make_interactive_3d_plot(df, x_col, y_col, z_col):
    # Create the figure and the 3D axis
    fig = plt.figure(figsize=(16, 14))
    ax = fig.add_subplot(111, projection='3d')
    
    # Adjust the main plot position to make room for sliders
    plt.subplots_adjust(bottom=0.3)
    
    # Extract the data
    x = df[df[z_col] < 150][x_col]
    y = df[df[z_col] < 150][y_col] 
    z = df[df[z_col] < 150][z_col]

    # Store original data ranges for zoom functionality
    x_range = x.max() - x.min()
    y_range = y.max() - y.min()
    z_range = z.max() - z.min()
    x_mid = (x.max() + x.min()) / 2
    y_mid = (y.max() + y.min()) / 2
    z_mid = (z.max() + z.min()) / 2

    # Create scatter plot with color gradient based on soil moisture
    scatter = ax.scatter(x, y, z, 
                        c=z,            # Color by soil moisture values
                        cmap='plasma',  # Different colormap
                        alpha=0.8,      # Slightly more opaque
                        s=60,           # Larger points
                        edgecolor='black',  # Black edges for better visibility
                        linewidth=0.5)

    # Labels and title with better formatting
    ax.set_xlabel(f'{x_col} (dB)', fontsize=14, fontweight='bold', labelpad=15)
    ax.set_ylabel(f'{y_col} (dB)', fontsize=14, fontweight='bold', labelpad=15)
    ax.set_zlabel('Soil Moisture SM1 (%)', fontsize=14, fontweight='bold', labelpad=15)
    ax.set_title(f'{x_col} vs {y_col} vs SM1 (%)', 
                fontsize=16, fontweight='bold', pad=20)

    # Colorbar with better styling
    cbar = fig.colorbar(scatter, ax=ax, pad=0.1, shrink=0.8)
    cbar.set_label('Soil Moisture (%)', fontsize=12, fontweight='bold')
    cbar.ax.tick_params(labelsize=10)

    # Set initial viewing angle
    elev_init = 25
    azim_init = 45
    zoom_init = 1.0  # 1.0 means no zoom (full data range)
    ax.view_init(elev=elev_init, azim=azim_init)

    # Add grid with subtle styling
    ax.grid(True, alpha=0.2, linestyle='--')

    # Improve tick labels
    ax.tick_params(axis='both', which='major', labelsize=10)

    # Add a background pane for better depth perception
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor('w')
    ax.yaxis.pane.set_edgecolor('w')
    ax.zaxis.pane.set_edgecolor('w')
    
    # Create sliders for elevation, azimuth, and zoom
    ax_elev = plt.axes([0.25, 0.2, 0.65, 0.03])
    ax_azim = plt.axes([0.25, 0.15, 0.65, 0.03])
    ax_zoom = plt.axes([0.25, 0.1, 0.65, 0.03])
    
    elev_slider = Slider(
        ax=ax_elev,
        label='Elevation',
        valmin=0,
        valmax=90,
        valinit=elev_init,
    )
    
    azim_slider = Slider(
        ax=ax_azim,
        label='Azimuth',
        valmin=0,
        valmax=360,
        valinit=azim_init,
    )
    
    zoom_slider = Slider(
        ax=ax_zoom,
        label='Zoom',
        valmin=0.1,
        valmax=2.0,
        valinit=zoom_init,
        valfmt='%1.2f',
    )
    
    # Update function for sliders
    def update(val):
        ax.view_init(elev=elev_slider.val, azim=azim_slider.val)
        
        # Update zoom by adjusting axis limits
        zoom_factor = zoom_slider.val
        ax.set_xlim3d(x_mid - x_range/2/zoom_factor, x_mid + x_range/2/zoom_factor)
        ax.set_ylim3d(y_mid - y_range/2/zoom_factor, y_mid + y_range/2/zoom_factor)
        ax.set_zlim3d(z_mid - z_range/2/zoom_factor, z_mid + z_range/2/zoom_factor)
        
        fig.canvas.draw_idle()
    
    # Register the update function with each slider
    elev_slider.on_changed(update)
    azim_slider.on_changed(update)
    zoom_slider.on_changed(update)
    
    # Add a button to print the current view parameters
    ax_button = plt.axes([0.8, 0.05, 0.1, 0.04])
    button = Button(ax_button, 'Print Parameters')
    
    def print_parameters(event):
        print(f"Current view parameters: elev={elev_slider.val:.1f}, azim={azim_slider.val:.1f}, zoom={zoom_slider.val:.2f}")
    
    button.on_clicked(print_parameters)
    
    # Add a reset button
    ax_reset = plt.axes([0.65, 0.05, 0.1, 0.04])
    reset_button = Button(ax_reset, 'Reset View')
    
    def reset_view(event):
        elev_slider.reset()
        azim_slider.reset()
        zoom_slider.reset()
    
    reset_button.on_clicked(reset_view)
    
    # Initialize the view
    update(None)
    
    plt.show()
    
    return elev_slider, azim_slider, zoom_slider

# Load your data
DATA_PATH = Path("/home/kshipra/work/major/ml experiments/data")
eos = pd.read_excel(DATA_PATH / "eos_processed.xlsx")
sentinel = pd.read_excel(DATA_PATH / "sentinel_processed.xlsx")

# Create the interactive plot
elev_slider, azim_slider, zoom_slider = make_interactive_3d_plot(sentinel, 'VH-pol', 'VV-pol', 'SM1 (%)')