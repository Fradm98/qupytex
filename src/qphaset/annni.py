# TenPy related helpers for the ANNNI model.

from tenpy.models.spins_nnn import SpinChainNNN
from qs_mps.mps_class import MPS as mps
import numpy as np
from qiskit import QuantumCircuit



class SpinChainNNNExt(SpinChainNNN):
    def init_terms(self, model_params):
        super().init_terms(model_params)
        c1 = model_params.get('c1', 0)
        self.add_local_term(c1, [('Sx1', (-1, 0))])


def model_annni_ext(k, h, *, l, j1=1, c1=0, params={}) -> SpinChainNNN:
    """Prepare the ANNNI model with parameters k and h.
    The half-length of the chain is given in l. The parameter c1 control
    an additional term that helps reducing the numerical issues with DMRG."""
    # -k * j1 = j2
    # h * j1 = b
    model_params = dict(L=l,    # Length
                        Jy=0, Jz=0, Jyp=0, Jzp=0,
                        hx=0, hy=0,
                        bc_MPS='finite',  # Boundary condition
                        Jx=-j1 * 2,
                        Jxp=k * j1 * 2,     # J_2
                        hz=-(h * j1),    # B
                        c1=c1, conserve=None)
    model_params.update(params)
    return SpinChainNNNExt(model_params)

def model_annni_qs_mps(k, h, chi, l, j1=1, c1=0, d=2) -> mps:
    """Prepare the ANNNI model with parameters k and h.
    The length of the chain is given in l. The parameter c1 control
    an additional term that helps reducing the numerical issues with DMRG."""
    # -k * j1 = j2
    # h * j1 = b
    return mps(L=l, d=d, model="ANNNI", chi=chi, h=h, k=k, J=j1, eps=c1)

def annni_lowfru_eig(n, en=0):
    """Circuit for ANNNI ground state corresponding to perturbative h and low frustration.
    Parameter 'en' in {0, 1} determines the eigenvector, 0 ground state,
    1 first excited state."""
    assert n % 2 == 0 and n >= 4
    assert en in {0, 1}
    qc = QuantumCircuit(n)
    qc.h(range(n - 1, 0, -1))
    for i in range(n - 1, 1, -1):
        qc.cx(i, i - 1)
    for i in range(1, n):
        qc.cx(i, i - 1)
    if en == 1:
        qc.x(0)
    return qc


def PT_tran(k):
    k = np.array(list(filter(lambda x: x>=0.5, k)))
    tran_name="PT"
    return k, ((1.05)*(k-0.5)), tran_name


def BKT_tran(k):
    k = np.array(list(filter(lambda x: x>=0.5, k)))
    tran_name="BKT"
    return k, ((1.05)*(np.sqrt((k-0.5)*(k-0.1)))), tran_name

def I_tran(k):
    k = np.array(list(filter(lambda x: x<=0.5, k)))
    tran_name="I"
    return k, (((1-k)/k)*(1-np.sqrt((1-3*k+4*(k**2))/(1-k)))), tran_name

def th_lines(line, ax, extent, color, linestyle):
    data = np.linspace(extent[-2],extent[-1],50)
    data, line_data, name = line(data)
    ax.plot(data, line_data, linewidth=1, color=color, linestyle=linestyle, label=name)
    ax.set_xlim((extent[0],extent[1]))
    ax.set_ylim((extent[2],extent[3]))