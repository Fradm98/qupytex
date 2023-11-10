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

# ---------------------------------------------------------------------------------------
# Open rdms
# ---------------------------------------------------------------------------------------
def open_rdms(file_path: str):
    """
    open_rdms

    This function loads the saved rmds (1 qubit rdm 2x2) as
    numpy arrays.

    file_path: str - path and name of the file to open
    
    """
    with open(file_path, 'r') as file:
        lines = file.readlines()

    X = []
    for line in lines:
        line1 = line.split(" ")
        op = '['
        cl = ']'
        line2 = []
        for elem in line1:
            if len(elem) > 1 and '\n' not in elem:
                if op in elem:
                    elem = elem.replace(op,'')
                elif cl in elem:  
                    elem = elem.replace(cl,'')

                line2.append(float(elem))
            elif len(elem) > 3 and ']\n' in elem:  
                elem = elem.replace(']\n','')
                line2.append(float(elem))
                    
        rdm = np.array(line2).reshape(2,2)
        X.append(rdm)
    return X
