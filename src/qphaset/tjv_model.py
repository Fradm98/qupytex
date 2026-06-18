from qs_mps.mps_class import MPS as mps

def model_tjv_qs_mps(V, t, chi, l, c1=0, d=3, Jz=10) -> mps:
    """Prepare the tjv model with parameters k and h.
    The length of the chain is given in l. The parameter c1 control
    an additional term that helps reducing the numerical issues with DMRG."""
    # -k * j1 = j2
    # h * j1 = b
    return mps(L=l, d=d, model="tj", chi=chi, h=V, J=Jz, k=(t,t), eps=c1)