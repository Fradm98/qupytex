from qs_mps.mps_class import MPS as mps

def model_cluster_qs_mps(K, h, chi, l, c1=0, d=2) -> mps:
    """Prepare the cluster model with parameters k and h.
    The length of the chain is given in l. The parameter c1 control
    an additional term that helps reducing the numerical issues with DMRG."""
    # -k * j1 = j2
    # h * j1 = b
    return mps(L=l, d=d, model="Cluster", chi=chi, h=h, J=K, eps=c1)