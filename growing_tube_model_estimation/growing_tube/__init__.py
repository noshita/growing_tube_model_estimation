"""
The :mod:`growing_tube` module implements
a simulator and an estimator of growing tube model .
"""

from ._growting_tube_estimation import (
    estimate_gt_e_and_r0,
    fit_bspline_curve,
    gen_knots,
)

__all__ = ["estimate_gt_e_and_r0", "fit_bspline_curve", "gen_knots"]
