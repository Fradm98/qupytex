from tenpy.linalg import np_conserved as npc


def model2ham(model):
    """Code extracted from Tenpy to convert the model MPO to a Hamiltonian matrix."""
    _labels_p = ['p' + str(i) for i in range(model.lat.N_sites)]
    _labels_pconj = [l + '*' for l in _labels_p]

    _sites = model.lat.mps_sites()
    legs = [s.leg for s in _sites]
    _pipe = npc.LegPipe(legs, qconj=1, sort=True, bunch=True)
    _pipe_conj = _pipe.conj()

    mpo = model.H_MPO
    full_H = mpo.get_W(0).take_slice(mpo.get_IdL(0), 'wL')
    full_H.ireplace_labels(['p', 'p*'], [_labels_p[0], _labels_pconj[0]])
    for i in range(1, mpo.L):
        W = mpo.get_W(i, copy=True)
        W.ireplace_labels(['p', 'p*'], [_labels_p[i], _labels_pconj[i]])
        if i == mpo.L - 1:
            W = W.take_slice(mpo.get_IdR(mpo.L - 1), 'wR')
        full_H = npc.tensordot(full_H, W, axes=['wR', 'wL'])
    full_H = full_H.combine_legs([_labels_p, _labels_pconj],
                                    new_axes=[0, 1],
                                    pipes=[_pipe, _pipe_conj])
    if mpo.explicit_plus_hc:
        full_H = full_H + full_H.conj().itranspose(full_H.get_leg_labels())
    return full_H
