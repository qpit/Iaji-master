"""
This module describes a mixed quantum state or a quantum harmonic oscillator
"""
class QuantumHarmonicOscillatorState:
    """
    This module defines a mixed quantum state.
    It consists of:
        - A Hilbert space
        - A density operator in the Fock basis
    """

    def __init__(self, density_matrix):
        """
        :param density_matrix:
            The density matrix of the quantum state.
        """
        