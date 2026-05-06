from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from qupytex.uhlmann_kernel import uhlmann_fidelity_1q
from qs_mps.utils import get_precision, load_list_of_lists, create_sequential_colors
from qupytex.utils import open_rdms
from scipy.interpolate import UnivariateSpline
import argparse
import numpy as np
import matplotlib.pyplot as plt


# default parameters of the plot layout
plt.rcParams["text.usetex"] = True # use latex
plt.rcParams["font.size"] = 13
plt.rcParams["figure.dpi"] = 300
plt.rcParams["figure.constrained_layout.use"] = True

# Purity check of local density matrices
# parameters
parser = argparse.ArgumentParser(prog="Purity Check")
parser.add_argument("L", help="Spin chain length", type=int)
parser.add_argument(
    "npoints",
    help="Number of points in an interval of transverse field values",
    type=int,
)
parser.add_argument(
    "h_i", help="Starting value of h (external transverse field)", type=float
)
parser.add_argument(
    "h_f", help="Final value of h (external transverse field)", type=float
)
parser.add_argument(
    "path", help="Path to the drive depending on the device used. Available are 'pc', 'mac', 'marcos'", type=str
)
parser.add_argument(
    "-m", "--model", help="Model to simulate", default="Ising", type=str
)

args = parser.parse_args()

# take the path and precision to save files
if args.path == 'pc':
    path_drive = "G:/My Drive/projects/0_ISING"
elif args.path == 'mac':
    path_drive = "/Users/fradm98/Google Drive/My Drive/projects/0_ISING"
elif args.path == 'marcos':
    path_drive = "/Users/fradm/Google Drive/My Drive/projects/0_ISING"
else:
    raise SyntaxError("Path not valid. Choose among 'pc', 'mac', 'marcos'")

interval = np.linspace(args.h_i, args.h_f, args.npoints).tolist()
precision = get_precision((args.h_f - args.h_i)/args.npoints)

file_path = f"{path_drive}/results/dataset/X_1-rdms_h_{args.h_i}-{args.h_f}_delta_{args.npoints}"
X = open_rdms(file_path=file_path)
Y = np.loadtxt(f"{path_drive}/results/dataset/Y_1-rdms_h_{args.h_i}-{args.h_f}_delta_{args.npoints}")

# check purity and entropy
flag = 0
purities = []
entropies = []
for i, rdm in enumerate(X):
    purity = np.trace(rdm @ rdm)
    purities.append(purity)
    entropies.append(-np.trace(rdm @ np.log2(rdm)))
    print(f"Purity: {purity:.5f} for h: {interval[i]:.{precision}f}")

    if purity == 1:
        flag += 1
        print(f"Pure for h: {interval[i]:.{precision}f}")

print(f"there are {flag} pure rdms in the loaded dataset")

# try the k_2 
def k_2(purity: list, delta: int):
    """
    k_2

    This function computes the second term in the 1-rdm uhlmann fidelity
    by taking the purity array of all the computed 1-rdms, and a
    step of delta.

    purity: list - a list of purities for the rdms in the specific interval of the parameter space
    delta: int - It is the step in the "index space" of the purity array. 
            Minimum value is 1

    """
    k = []
    for i in range(len(purity)-delta):
        k.append(np.sqrt((1-purity[i])*(1-purity[i+delta])))
    return k

chi = 64
fname_what = f"{path_drive}/results/entropy/{args.L//2}_bond_entropy"
entropies = np.loadtxt(f"{fname_what}_{args.model}_L_{args.L}_h_{args.h_i}-{args.h_f}_delta_{args.npoints}_chi_{chi}")
norm_entropies = entropies/np.max(entropies)

interval = interval[100:]
norm_entropies = norm_entropies[100:]
purities = purities[100:]

k2 = False
deltas = [1,2,5,10,100,500]
colors = create_sequential_colors(num_colors=len(deltas), colormap_name='winter')
title = "entropy_vs_purity"
plt.title(f"Entropy $vs$ Purity of rdms for $L={args.L}$")
plt.plot(interval, purities, label='purity')
plt.plot(interval, norm_entropies, label='normalized entropy')
if k2:
    for i, delta in enumerate(deltas):
        second_term = k_2(purity=purities, delta=delta)
        pr = 10**(-precision)
        plt.plot(interval[:-1-delta+1], second_term, color=colors[i], alpha=0.5, linestyle=':', label=f'K_2 Uhlmann: $\\Delta h ={(delta*pr)}$')
plt.hlines(y=1/2, xmin=interval[0], xmax=interval[-1], colors='red', linestyles='--', linewidth=1, label='maximally entangled states')
plt.xlabel("external field (h)")
plt.ylabel("$S_{\\chi}(h)$ vs Tr$\\left[(\\rho^2(h))\\right]$")
plt.legend(fontsize=10)
plt.savefig(f"{path_drive}/figures/purity/{title}_{args.model}_L_{args.L}_h_{args.h_i}-{args.h_f}_delta_{args.npoints}_k2_{k2}.png")
# plt.show()


# let us find the second derivative
y_spl_entr = UnivariateSpline(interval,norm_entropies,s=0,k=4)
y_spl_purity = UnivariateSpline(interval,purities,s=0,k=4)

y_spl_1d_entr = y_spl_entr.derivative(n=2)
y_spl_2d_purity = y_spl_purity.derivative(n=2)

derivative_eval_entr = y_spl_1d_entr(interval)
derivative_eval_purity = y_spl_2d_purity(interval)

zero_crossings_entr = np.where(np.diff(np.sign(derivative_eval_entr)))[0]
zero_crossings_purity = np.where(np.diff(np.sign(derivative_eval_purity)))[0]

y_entr = np.array([np.nan for _ in range(len(interval))])
for zero_entr in zero_crossings_entr:
    y_entr[zero_entr] = 0    
y_purity = np.array([np.nan for _ in range(len(interval))])
for zero_purity in zero_crossings_purity:
    y_purity[zero_purity] = 0

title = 'second_derivative_entropy_vs_second_derivative_purity'
plt.title("$\dot{d}$ of Entropy $vs$ $\ddot{d}$ of Purity of rdms for " + f"$L={args.L}$")
plt.plot(interval,derivative_eval_entr, color='blue', label='second derivative entropy')
plt.plot(interval,derivative_eval_purity, color='darkorange', label='second derivative purity')
plt.scatter(interval, y_entr, marker='o', edgecolor='blue', facecolor='none', alpha=0.6, label='zeros entropy')
plt.scatter(interval, y_purity, marker='o', edgecolor='darkorange', facecolor='none', alpha=0.6, label='zeros purity')
plt.hlines(y=0, xmin=interval[0], xmax=interval[-1], colors='red', linestyles='--', linewidth=1)
plt.xlabel("external field (h)")
plt.ylabel("$d^2/dh^2$ $S_{\\chi}(h)$ vs $d^2/dh^2$ Tr$\\left[(\\rho(h))\\right]$")
plt.legend()
plt.savefig(f"{path_drive}/figures/purity/{title}_{args.model}_L_{args.L}_h_{args.h_i}-{args.h_f}_delta_{args.npoints}.png")
# plt.show()

# error
interval_entr = interval[zero_crossings_entr[-1]]
interval_purity = interval[zero_crossings_purity[-1]]
print(f"zero crossing entropy at h: {interval_entr}")
print(f"zero crossing purity at h: {interval_purity}")
print(f"Error entropy vs purity: {np.abs(interval_purity-interval_entr):1e}")
print(f"Remember, precision of parameter space: 1e-{precision}")
print(f"Hence, mean value of the phase transition is h_c: {np.abs(interval_purity+interval_entr)/2:.{precision}f}")
