"""
SemiCART: Semi-Supervised Classification and Regression Tree algorithm.

This package provides an implementation of the Semi-CART algorithm, which enhances
traditional CART by incorporating a distance-based weighting method that assigns
weights to training instances based on their proximity to test instances.
"""

from semicart.core.semicart import SemiCART

__version__ = "1.0.0"
__author__ = "Aydin Abedinia and Vahid Seydi"
__license__ = "MIT"

__all__ = ["SemiCART"] 