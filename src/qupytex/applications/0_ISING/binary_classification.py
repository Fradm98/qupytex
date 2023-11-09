from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from qupytex.uhlmann_kernel import uhlmann_fidelity_1q
from qs_mps.utils import get_precision, load_list_of_lists
from scipy.interpolate import UnivariateSpline
import argparse
import numpy as np
import matplotlib.pyplot as plt


# default parameters of the plot layout
plt.rcParams["text.usetex"] = True # use latex
plt.rcParams["font.size"] = 13
plt.rcParams["figure.dpi"] = 300
plt.rcParams["figure.constrained_layout.use"] = True

# Decision boundary with Uhlmann Distance
# parameters

parser = argparse.ArgumentParser(prog="Binary Classification")
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

# X = load_list_of_lists(f"{path_drive}/results/dataset/X_1-rdms_h_{args.h_i}-{args.h_f}_delta_{args.npoints}")
Y = np.loadtxt(f"{path_drive}/results/dataset/Y_1-rdms_h_{args.h_i}-{args.h_f}_delta_{args.npoints}")

file_path = f"{path_drive}/results/dataset/X_1-rdms_h_{args.h_i}-{args.h_f}_delta_{args.npoints}"
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

# check purity
flag = 0
purities = []
for i, rdm in enumerate(X):
    purity = np.trace(rdm @ rdm)
    purities.append(purity)
    print(f"Purity: {purity:.5f} for h: {interval[i]:.{precision}f}")

    if purity == 1:
        flag += 1
        print(f"Pure for h: {interval[i]:.{precision}f}")

print(f"there are {flag} pure rdms in the loaded dataset")

# try the k_2 
def k_2(purity,delta):
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
        k.append(np.sqrt((1-purity[i])(1-purity[i+delta])))
    return k

delta = 1
second_term = k_2(purities, delta)
chi = 32
fname_what = f"{path_drive}/results/entropy/{args.L//2}_bond_entropy"
entropies = np.loadtxt(f"{fname_what}_{args.model}_L_{args.L}_h_{args.h_i}-{args.h_f}_delta_{args.npoints}_chi_{chi}")
norm_entropies = entropies/np.max(entropies)
title = "entropy_vs_purity"
plt.title(f"Entropy $vs$ Purity of rdms for $L={args.L}$")
plt.plot(interval, purities, label='purity')
plt.plot(interval, norm_entropies, label='normalized entropy')
plt.plot(interval, norm_entropies, label=f'K_2 Uhlmann: $\Delta={delta*precision}$')
plt.hlines(y=1/2, xmin=interval[0], xmax=interval[-1], colors='red', linestyles='--', linewidth=1, label='maximally entangled states')
plt.xlabel("external field (h)")
plt.ylabel("$S_{\\chi}(h)$ vs Tr$\\left[(\\rho(h))\\right]$")
plt.legend()
plt.savefig(f"{path_drive}/figures/purity/{title}_{args.model}_L_{args.L}_h_{args.h_i}-{args.h_f}_delta_{args.npoints}.png")
plt.show()


# let us find the second derivative
interval = interval[100:]
norm_entropies = norm_entropies[100:]
purities = purities[100:]
y_spl_entr = UnivariateSpline(interval,norm_entropies,s=0,k=4)
y_spl_purity = UnivariateSpline(interval,purities,s=0,k=4)

y_spl_2d_entr = y_spl_entr.derivative(n=2)
y_spl_2d_purity = y_spl_purity.derivative(n=2)

derivative_eval_entr = y_spl_2d_entr(interval)
derivative_eval_purity = y_spl_2d_purity(interval)

zero_crossings_entr = np.where(np.diff(np.sign(derivative_eval_entr)))[0]
zero_crossings_purity = np.where(np.diff(np.sign(derivative_eval_purity)))[0]

y_entr = np.array([np.nan for _ in range(len(interval))])
for zero_entr in zero_crossings_entr:
    y_entr[zero_entr] = 0    
y_purity = np.array([np.nan for _ in range(len(interval))])
for zero_purity in zero_crossings_purity:
    y_purity[zero_purity] = 0

title = 'second_derivative_entropy_vs_purity'
plt.title(f"Second derivative of Entropy $vs$ Purity of rdms for $L={args.L}$")
plt.plot(interval,derivative_eval_entr, color='blue', label='second derivative entropy')
plt.plot(interval,derivative_eval_purity, color='darkorange', label='second derivative purity')
plt.scatter(interval, y_entr, marker='o', edgecolor='blue', facecolor='none', alpha=0.6, label='zeros entropy')
plt.scatter(interval, y_purity, marker='o', edgecolor='darkorange', facecolor='none', alpha=0.6, label='zeros purity')
plt.hlines(y=0, xmin=interval[0], xmax=interval[-1], colors='red', linestyles='--', linewidth=1)
plt.xlabel("external field (h)")
plt.ylabel("$d^2/dh^2$ $S_{\\chi}(h)$ vs $d^2/dh^2$ Tr$\\left[(\\rho(h))\\right]$")
plt.legend()
plt.savefig(f"{path_drive}/figures/purity/{title}_{args.model}_L_{args.L}_h_{args.h_i}-{args.h_f}_delta_{args.npoints}.png")
plt.show()

# error
interval_entr = interval[zero_crossings_entr[-1]]
interval_purity = interval[zero_crossings_purity[-1]]
print(f"zero crossing entropy at h: {interval_entr}")
print(f"zero crossing purity at h: {interval_purity}")
print(f"Error entropy vs purity: {np.abs(interval_purity-interval_entr):1e}")
print(f"Remember, precision of parameter space: 1e-{precision}")
print(f"Hence, mean value of the phase transition is h_c: {np.abs(interval_purity+interval_entr)/2:.{precision}f}")
