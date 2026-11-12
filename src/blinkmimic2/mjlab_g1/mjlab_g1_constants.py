"""Unitree G1 as mjlab ships it (the "standard mjlab G1" that realmimic trains and evaluates on).

Same motors, gains, action scale, default pose, joint limits and inertias as ``g1`` (checked in audit B);
the difference is contact: mjlab's capsule collision geometry (7 capsules per foot) with self-collisions
on (``FULL_COLLISION``: feet condim 3 and priority 1, the rest condim 1), and mjlab's solver budget
(10 Newton / 20 line-search iterations, tolerance 1e-8, 50 CCD iterations).
"""

from __future__ import annotations

import dataclasses

from mjlab.asset_zoo.robots.unitree_g1.g1_constants import G1_XML, get_g1_robot_cfg

from blinkmimic2.g1.g1_constants import SPEC as G1_SPEC

SPEC = dataclasses.replace(
    G1_SPEC,
    name="mjlab_g1",
    mjcf_path=str(G1_XML),
    robot_cfg=get_g1_robot_cfg,
    solver=(10, 20, 1e-8, 50),
)
