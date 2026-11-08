from qs_mps.mps_class import MPS as mps

def model_rydberg_qs_mps(delta_omega, blockade_radius, chi, l, c1=0, d=2, n_neighbours=3) -> mps:
    """Prepare the rydberg model with parameters (delta/omega) and (blockade radius/a).
    The length of the chain is given in l. The parameter c1 control
    an additional term that helps reducing the numerical issues with DMRG
    (not used by the qs-mps Rydberg MPO).
    n_neighbours: range of the truncated 1/r^6 tail (the paper truncates at the fourth
    neighbour; 3 keeps the original MPO).
    """
    m = mps(L=l, d=d, model="Rydberg", chi=chi, h=delta_omega, J=blockade_radius**6, eps=c1)
    m.rydberg_range = n_neighbours
    return m
