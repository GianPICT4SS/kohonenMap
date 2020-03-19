import numpy as np


class Kohonen():

    """Two-dimensional Self-Organizing Map"""

    def __init__(self, lattice, tau_1, tau_2, eta_0, sigma_0):

        self.lattice = lattice  # matrix of dimension mxn
        self.m = lattice.shape[0]
        self.n = lattice.shape[1]
        self.tau_sig = float(tau_1)  # constant time in sigma(n)
        self.tau_eta = float(tau_2)  # constant time in eta(n)
        self.eta_0 = float(eta_0)
        self.sigma_0 = float(sigma_0)

    def init_weights(self):

        self.lattice = np.random.randn(self.m, self.n)
        return self

    def eta(self, n):


        eta = self.eta_0*np.exp(-1*(n /self.tau_eta))
        return eta

    def lateral_distance(self, winner_vec, act_vec):

        r_i = self.lattice[winner_vec]

        d = np.linalg.norm(self.lattice[])
    def sigma(self, n):
        sigma = self.sigma_0*np.exp(-1*())