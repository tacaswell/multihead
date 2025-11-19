from typing import NamedTuple, Self

import numpy as np
from numpy.typing import NDArray, ArrayLike

from multihead.config import AnalyzerConfig


class TrigAngle(NamedTuple):
    angle: NDArray[np.float64]
    sin: NDArray[np.float64]
    cos: NDArray[np.float64]

    @classmethod
    def from_deg(cls, angle: ArrayLike[np.float64]) -> Self:
        return cls.from_rad(np.deg2rad(angle))

    @classmethod
    def from_rad(cls, angle: ArrayLike[np.float64]) -> Self:
        np_angle = np.asarray(angle)
        return cls(np_angle, np.sin(np_angle), np.cos(np_angle))


def _arm_from_phi_tth_eq20(
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
    ) -> NDArray[np.float64]:
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


def _compute_new_phi_eq15(
    zd: NDArray[np.float64],
    L3: NDArray[np.float64],
    theta_i: TrigAngle,
    crystal_roll: TrigAngle,
    crystal_yaw: TrigAngle,
    Rp: NDArray[np.float64],
    tth: TrigAngle,
) -> TrigAngle:
    """
    Implementation of Equation 15 to compute the updated phi angle.

    Parameters
    ----------
    zd : float
        The z position on the detector
    L3 : float
        The L3 value from Equation 13
    theta_i : TrigAngle
        Theta i angle
    crystal_roll : TrigAngle
        Crystal roll angle
    crystal_yaw : TrigAngle
        Crystal yaw angle
    Rp : float
        Corrected length (L' in the paper)
    tth : TrigAngle
        Scattering 2theta angle

    Returns
    -------
    TrigAngle
        The updated phi angle
    """
    return TrigAngle.from_rad(
        np.arcsin(
            (zd + 2 * L3 * theta_i.sin * crystal_roll.sin * crystal_yaw.cos)
            / ((Rp + L3) * tth.sin)
        )
    )


def _compute_L3_eq13(
    config: AnalyzerConfig,
    theta_d: TrigAngle,
    theta_i: TrigAngle,
    arm_tth_i: TrigAngle,
    arm_tth_d: TrigAngle,
    tth: TrigAngle,
    phi: TrigAngle,
    det_yaw: TrigAngle,
    crystal_roll: TrigAngle,
    crystal_yaw: TrigAngle,
    Rp: NDArray[np.float64],
) -> NDArray[np.float64]:
    """
    Implementation of Equation 13 to compute L3.

    Parameters
    ----------
    config : AnalyzerConfig
        Configuration containing R and Rd values
    theta_d : TrigAngle
        Theta d angle
    theta_i : TrigAngle
        Theta i angle
    arm_tth_i : TrigAngle
        Arm 2theta i angle
    arm_tth_d : TrigAngle
        Arm 2theta d angle
    tth : TrigAngle
        Scattering 2theta angle
    phi : TrigAngle
        Phi angle
    det_yaw : TrigAngle
        Detector yaw angle
    crystal_roll : TrigAngle
        Crystal roll angle
    crystal_yaw : TrigAngle
        Crystal yaw angle
    Rp : float
        Corrected length (L' in the paper)

    Returns
    -------
    float
        The computed L3 value
    """
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
        * (tth.sin * phi.sin - 2 * theta_i.sin * crystal_roll.sin * crystal_yaw.sin)
    )
    return numerator / denominator


def arm_from_z(
    z: ArrayLike[np.float64], scatter_tth: ArrayLike[np.float64], config: AnalyzerConfig
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Given a range of z and a scattering angle, compute the arm 2ϴ where scatter
    will be seen

    Parameters
    ----------
    z : Array[np.float]
        The position along the detector face in mm

    scatter_tth : float
        The angle of the scatter of interest In deg.

    config : AnalyzerConfig
        All of the calibration / alignment values

    Returns
    -------
    arm_tth, phi
    """
    z = np.asarray(z)
    if config.detector_roll != 0:
        z *= np.cos(np.deg2rad(config.detector_roll))

    # pull out and convert all angles to radians
    det_yaw = TrigAngle.from_deg(config.detector_yaw)
    crystal_yaw = TrigAngle.from_deg(config.crystal_yaw)
    crystal_roll = TrigAngle.from_deg(config.crystal_roll)

    theta_i = TrigAngle.from_deg(config.theta_i)
    theta_d = TrigAngle.from_deg(config.theta_d)

    tth = TrigAngle.from_deg(scatter_tth)

    # corrected sample to crystal distance, L' in paper
    Rp = (
        config.R
        * (
            theta_i.cos * crystal_roll.sin * crystal_yaw.sin
            - theta_i.sin * crystal_roll.cos
        )
        / (-theta_i.sin)
    )

    # step 1: estimate phi
    #    step 2: plug into eq 20 to get arm_th - theta_i and arm_th - theta_d
    #    step 3: plug in to modified eq 13 to get L3
    #    step 4: plug into eq 15 to get updated phi
    #    step 5: repeat from 2 N times
    # step 6: when stable put arm tth from 2 in output

    # step 1
    phi = TrigAngle.from_rad(np.arctan(z / ((config.R + config.Rd) * tth.sin)))

    for _ in range(9):
        # step 2
        arm_tth_i = _arm_from_phi_tth_eq20(crystal_roll, crystal_yaw, tth, phi, theta_i)
        arm_tth_d = TrigAngle.from_rad(arm_tth_i.angle - theta_d.angle + theta_i.angle)
        # step 3
        L3 = _compute_L3_eq13(
            config,
            theta_d,
            theta_i,
            arm_tth_i,
            arm_tth_d,
            tth,
            phi,
            det_yaw,
            crystal_roll,
            crystal_yaw,
            Rp,
        )

        # step 4
        new_phi = _compute_new_phi_eq15(
            z, L3, theta_i, crystal_roll, crystal_yaw, Rp, tth
        )

        # step 5
        phi = new_phi

    # step 6
    arm_tth_i = _arm_from_phi_tth_eq20(crystal_roll, crystal_yaw, tth, phi, theta_i)
    return np.rad2deg(np.array(arm_tth_i.angle + theta_i.angle)), np.rad2deg(
        np.array(phi.angle)
    )
