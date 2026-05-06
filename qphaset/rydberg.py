from qs_mps.mps_class import MPS as mps

def model_rydberg_qs_mps(delta_omega, blockade_radius, chi, l, c1=0, d=2) -> mps:
    """Prepare the rydberg model with parameters (delta/omega) and (blockade radius/a).
    The length of the chain is given in l. The parameter c1 control
    an additional term that helps reducing the numerical issues with DMRG.
    
    """
    return mps(L=l, d=d, model="Rydberg", chi=chi, h=delta_omega, J=blockade_radius, eps=c1)