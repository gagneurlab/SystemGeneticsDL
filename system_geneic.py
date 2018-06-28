import numpy as np

bases = ['A', 'C', 'G', 'T']

def onehot(seq):
    X = np.zeros((len(seq), len(bases)))
    for i, char in enumerate(seq):
        if char == "N":
            pass
        else:
            X[i, bases.index(char.upper())] = 1
    return X

# A naiv version
def pwm_scan(sequence, pwm, pad=0, stride=1):
    """ sequence: (N, L, 4)
    output length given by L' = 1 + (L + 2*P - F) / stride
    """
    assert len(sequence.shape) == 3
    assert pwm.shape[1] == 4
    N, L, _ = sequence.shape
    F, _ = pwm.shape
    S = stride
    assert (L + 2 * pad - F) % S == 0, "Size not fit."
    L_out = int(1 + (L + 2 * pad - F) / S)
    out = np.zeros((N, L_out))

    x_pad = np.pad(sequence, ((0, 0), (1, 1), (0,0)), mode='constant')

    for i in range(N):
        seq = sequence[i]
        for j in range(L_out):
            seq_patch = seq[(j * S):(j * S + F)]
            out[i, j] = np.dot(seq_patch.flatten(), pwm.flatten())

    return out