import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------------------
# Create Sequential Colors
# ---------------------------------------------------------------------------------------
def create_sequential_colors(num_colors, colormap_name):
    """
    create_sequential_colors

    This function creates a sequence of colors extracted from a specified colormap.

    num_colors: int - number of colors we want to extract
    colormap_name: string - colormap name we want to use

    """
    colormap = plt.cm.get_cmap(colormap_name)
    colormap_values = np.linspace(0, 1, num_colors)
    colors = [colormap(value) for value in colormap_values]
    return colors