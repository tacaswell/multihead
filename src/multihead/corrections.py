from typing import NamedTuple, Self

import numpy as np
from numpy.typing import NDArray

from multihead.config import AnalyzerConfig


class TrigAngle(NamedTuple):
    angle: float
    sin: float
    cos: float

    @classmethod
    def from_deg(cls, angle: float) -> Self:
        return cls.from_rad(np.deg2rad(angle))

    @classmethod
    def from_rad(cls, angle: float) -> Self:
        return cls(angle, np.sin(angle), np.cos(angle))


def _arm_from_phi_tth(
    crystal_roll: TrigAngle,
    crystal_yaw: TrigAngle,
    tth: TrigAngle,
    phi: TrigAngle,
    theta_i: TrigAngle,
) -> TrigAngle:
    """
    Helper for equation 20.  All angles are in rads

    Returns 2ϴ - θi
    """

    def inner(
        crystal_roll: TrigAngle,
        crystal_yaw: TrigAngle,
        tth: TrigAngle,
        phi: TrigAngle,
        theta_i: TrigAngle,
        delta: float,
    ) -> float:
        phi_d = phi.angle + delta
        X = (
            crystal_yaw.sin * crystal_roll.sin * tth.cos
            - crystal_yaw.cos * tth.sin * np.cos(phi_d)
        )
        Y = (
            crystal_roll.cos * tth.cos
            + crystal_roll.sin * crystal_yaw.sin * tth.sin * np.cos(phi_d)
        )
        Z = crystal_roll.sin * crystal_yaw.cos * tth.sin * np.sin(phi_d) - theta_i.sin
        return np.arccos(
            (2 * X * Z + np.sqrt(4 * Z**2 * X**2 - 4 * (X**2 + Y**2) * (Z**2 - Y**2)))
            / (2 * (X**2 + Y**2))
        )

    # when 2ϴ - θi goes negative, we need to take the negative of the arccos
    # as cos(x) == cos(-x) but arccos always returns a positive number
    # Per the SI, the solution is to take the numerical second derivative and
    # ensure it is negative

    # paper says 10**-10, but that gives numerical instabilities
    delta = 5e-7
    fm = inner(crystal_roll, crystal_yaw, tth, phi, theta_i, -delta)
    f0 = inner(crystal_roll, crystal_yaw, tth, phi, theta_i, 0)
    fp = inner(crystal_roll, crystal_yaw, tth, phi, theta_i, delta)
    sign = -np.sign((fp - 2 * f0 + fm) / (delta**2))
    return TrigAngle.from_rad(sign * f0)


def arm_from_z(z: NDArray[float], scatter_tth: float, config: AnalyzerConfig):
    """
    Given a range of z and a scattering angle, compute the arm 2ϴ where scatter
    will be seen

    Parameters
    ----------
    z : Array[float]
        The position along the detector face in mm

    scatter_tth : float
        The angle of the scatter of interest In deg.

    config : AnalyzerConfig
        All of the calibration / alignment values
    """

    if config.detector_roll != 0:
        z *= np.cos(np.deg2rad(config.detector_roll))

    # pull out and convert all angles to radians
    det_yaw = TrigAngle.from_deg(config.detector_yaw)
    crystal_yaw = TrigAngle.from_deg(config.crystal_yaw)
    crystal_roll = TrigAngle.from_deg(config.crystal_roll)

    theta_i = TrigAngle.from_deg(config.theta_i)
    theta_d = TrigAngle.from_deg(config.theta_d)

    tth = TrigAngle.from_deg(scatter_tth)

    Rp = (
        config.R
        * (
            theta_i.cos * crystal_roll.sin * crystal_yaw.sin
            - theta_i.sin * crystal_roll.cos
        )
        / (-theta_i.sin)
    )

    # step 1: estimate phi
    # step 2: plug into eq 20 to get arm_th - theta_i
    # step 3: plug in to modified eq 13 to get L3
    # step 4: plug into eq 15 to get updated phi
    # step 6: repeat from 2
    # step 7: when stable put arm tth from 2 in output

    arm_tth_out = np.zeros_like(z).astype(float)

    for i, zd in enumerate(z):
        # step 1
        phi = TrigAngle.from_rad(np.arctan(zd / ((config.R + config.Rd) * tth.sin)))

        for _ in range(5):
            # step 2
            arm_tth_i = _arm_from_phi_tth(crystal_roll, crystal_yaw, tth, phi, theta_i)
            arm_tth_d = TrigAngle.from_rad(
                arm_tth_i.angle - theta_d.angle + theta_i.angle
            )
            # step 3
            numerator = (
                -config.R * theta_d.cos
                - config.Rd
                + Rp
                * (
                    arm_tth_d.cos * tth.cos
                    + arm_tth_d.sin * tth.sin * phi.cos
                    + np.tan(det_yaw.angle) * tth.sin * phi.sin
                )
            )
            denominator = (
                -arm_tth_d.cos
                * (
                    tth.cos
                    + 2
                    * theta_i.sin
                    * (
                        arm_tth_i.sin * crystal_roll.cos
                        + arm_tth_i.cos * crystal_roll.sin * crystal_yaw.sin
                    )
                )
                - arm_tth_d.sin
                * (
                    tth.sin * phi.cos
                    + 2
                    * theta_i.sin
                    * (
                        arm_tth_i.sin * crystal_roll.sin * crystal_yaw.sin
                        - arm_tth_i.cos * crystal_roll.cos
                    )
                )
                - np.tan(det_yaw.angle)
                * (
                    tth.sin * phi.sin
                    - 2 * theta_i.sin * crystal_roll.sin * crystal_yaw.sin
                )
            )
            L3 = (numerator) / denominator

            new_phi = TrigAngle.from_rad(
                np.arcsin(
                    (zd + 2 * L3 * theta_i.sin * crystal_roll.sin * crystal_yaw.cos)
                    / ((Rp + L3) * tth.sin)
                )
            )

            phi = new_phi
        arm_tth_out[i] = arm_tth_d.angle + theta_d.angle
    return np.rad2deg(arm_tth_out)
