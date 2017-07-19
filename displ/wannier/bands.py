import numpy as np
import numpy.matlib as matlib
from numpy.linalg import inv

def Hk(k, Hr, latVecs):
    '''Return the Hamiltonian H(k) in the Wannier basis as determined from
    the real-space Hamiltonian Hr. 
    
    k is a 3-element tuple representing a vector in the Cartesian basis
    (i.e. k = (kx, ky, kz)).
    Hr should have the format returned by extractHr().
    latVecs is a list of Cartesian representations of the lattice vectors
    (i.e. latVecs[0] = a, latVecs[1] = b, latVecs[2] = c).

    The returned matrix H(k) is a numpy matrix of shape (nb, nb) with data
    type complex128, where nb is the number of bands in the Wannier basis.
    '''
    result = None
    for key, value in Hr.items():
        Hr_r = value[0]
        degen = value[1]
        ra, rb, rc = key[0], key[1], key[2]
        # Initialize result matrix if necessary.
        if result is None:
            nb = Hr_r.shape[0]
            result = matlib.zeros((nb, nb), dtype=np.complex128)

        # Add contribution from H_{r} to H(k):
        #   weight * exp(i k dot (r_i - r_j)) * H_{ij}
        weight = 1.0 / float(degen)
        kr = _k_dot_eta(k, [ra, rb, rc], latVecs)
        result += weight * np.exp(1j * kr) * Hr_r

    return result

def _k_dot_eta(k, eta, latVecs):
    '''Compute k dot eta, where k is a vector in the Cartesian basis and eta
    is a vector in the lattice basis.

    latVecs is defined as in Hk().
    '''
    rtrans = np.array([[eta[0], eta[1], eta[2]]])
    kcart = np.array([[k[0], k[1], k[2]]]).T
    ret = np.dot(rtrans, np.dot(np.array(latVecs), kcart))[0, 0]
    return ret

def Hk_recip(k, Hr):
    '''Return the Hamiltonian H(k) in the Wannier basis as determined from
    the real-space Hamiltonian Hr.

    k is a 3-element tuple representing a vector in the reciprocal lattice
    basis (i.e. k = (k1, k2, k3)).

    Hr should have the format returned by extractHr().

    The returned matrix H(k) is a numpy matrix of shape (nb, nb) with data
    type complex128, where nb is the number of bands in the Wannier basis.
    '''
    result = None
    for key, value in Hr.items():
        Hr_r = value[0]
        degen = value[1]
        ra, rb, rc = key[0], key[1], key[2]
        # Initialize result matrix if necessary.
        if result is None:
            nb = Hr_r.shape[0]
            result = matlib.zeros((nb, nb), dtype=np.complex128)

        # Add contribution from H_{r} to H(k):
        #   weight * exp(i k dot (r_i - r_j)) * H_{ij}
        weight = 1.0 / float(degen)
        kr = 2*np.pi*(k[0]*ra + k[1]*rb + k[2]*rc)
        result += weight * np.exp(1j * kr) * Hr_r

    return result

def dHk_dk(k, Hr, latVecs):
    '''Return dH(k)/dk, the gradient of the Hamiltonian H(k) in the Wannier
    basis as determined from the real-space Hamiltonian Hr. The gradient
    is evaluated at the input point k, which is a 3-element tuple in the
    Cartesian basis. The returned gradient is in the Cartesian basis of k.
    
    Hr should have the format returned by extractHr(). The returned value 
    dH(k)/dk is a tuple of numpy matrices of shape (nb, nb) with data type
    complex128, where nb is the number of bands in the Wannier basis.
    The components of the returned tuple correspond to the (x, y, z)
    components of the gradient.

    latVecs is defined as in Hk().
    '''
    result = None
    for key, value in Hr.items():
        Hr_r = value[0]
        degen = value[1]
        ra, rb, rc = key[0], key[1], key[2]
        # Initialize result matrix if necessary.
        if result is None:
            nb = Hr_r.shape[0]
            result = []
            for cartIndex in range(3):
                result.append(matlib.zeros((nb, nb), dtype=np.complex128))

        # Add contibution from H_{r} to dH(k)/dk:
        #   i/degen * (d/dk)(k dot (r_i - r_j)) * exp(i k dot (r_i - r_j)) * H_{ij}
        weight = 1.0 / float(degen)
        kr = _k_dot_eta(k, [ra, rb, rc], latVecs)
        coeff = 1j * weight * np.exp(1j * kr) * Hr_r
        for cartIndex in range(3):
            derivdot = (ra * latVecs[0, cartIndex]
                        + rb * latVecs[1, cartIndex]
                        + rc * latVecs[2, cartIndex])
            result[cartIndex] += coeff * derivdot

    return result

def d2Hk_dk(k, Hr, latVecs):
    '''Return a list of values giving d^2 H(k)/dk^2, the second derivative of
    the Hamiltonian H(k) in the Wannier basis as determined from the real-space
    Hamiltonian Hr, for each combination of Cartesian k directions (dk_x dk_x,
    dk_x dk_y, dk_x dk_z, dk_y dk_x, ...). The second derivative is evaluated
    at the input point k, which is a 3-element tuple in the Cartesian basis.

    Hr should have the format returned by extractHr(). The returned value
    dH(k)/dk is a dictionary with keys (cp, c) where cp = (0, 1, 2) for (x, y,
    z) (and similarly for c) and values which are numpy matrices of shape (nb,
    nb) with data type complex128, where nb is the number of bands in the
    Wannier basis.

    latVecs is defined as in Hk().
    '''
    result = None
    for key, value in Hr.items():
        Hr_r = value[0]
        degen = value[1]
        ra, rb, rc = key[0], key[1], key[2]
        # Initialize result matrix if necessary.
        if result is None:
            nb = Hr_r.shape[0]
            result = {}
            for cp in range(3):
                for c in range(3):
                    result[(cp, c)] = matlib.zeros((nb, nb), dtype=np.complex128)

        # Add contibution from H_{r} to d2H(k)/dk2:
        #   1/degen * (d2/dk2)(exp(i k dot (r_i - r_j))) * H_{ij}
        weight = 1.0 / float(degen)
        kr = _k_dot_eta(k, [ra, rb, rc], latVecs)
        coeff = -weight * np.exp(1j * kr) * Hr_r
        for cp in range(3):
            eta_cp = (ra * latVecs[0, cp]
                    + rb * latVecs[1, cp]
                    + rc * latVecs[2, cp])
            for c in range(3):
                eta_c = (ra * latVecs[0, c]
                       + rb * latVecs[1, c]
                       + rc * latVecs[2, c])

                result[(cp, c)] += coeff * eta_cp * eta_c

    return result
