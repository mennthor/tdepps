# coding: utf-8

"""
Collection of coordinate related helper methods.
"""

from __future__ import division, absolute_import
from builtins import map
from future import standard_library
standard_library.install_aliases()

import numpy as np


def ThetaPhiToDecRa(theta, phi):
    """
    Convert healpy theta, phi coordinates to equatorial ra, dec coordinates.
    ``phi`` and right-ascension are assumed to be the same value, declination is
    converted with ``dec = pi/2 - theta``.

    Parameters
    ----------
    theta, phi : array-like
        Healpy cooridnates in radian. ``theta ``is i n range ``[0, pi]``,
        ``phi`` in ``[0, 2pi]``.

    Returns
    -------
    dec, ra : array-like
        Equtorial coordinates declination in ``[-pi/2, pi/2]`` and
        right-ascension equal to ``phi`` in ``[0, 2pi]``.
    """
    theta, phi = map(np.atleast_1d, [theta, phi])
    return np.pi / 2. - theta, phi


def cos_angdist(ra1, dec1, ra2, dec2):
    """
    Great circle angular distance in equatorial coordinates.

    Parameters
    ----------
    ra1, dec1 : array-like
        Position(s) of the source point(s)
    ra2, dec2 : array-like
        Position(s) of the target point(s)

    Returns
    -------
    cos_dist : array-like
        Cosine of angular distances from point(s) (ra1, dec1) to (ra2, dec2).
        Shape depends onn input shapes. If both arrays are 1D they must have the
        same length and output is 1D too. General broadcasting rules up to 2D
        should apply.
    """
    cos_dist = (np.cos(ra1 - ra2) * np.cos(dec1) * np.cos(dec2) +
                np.sin(dec1) * np.sin(dec2))
    return np.clip(cos_dist, -1., 1.)


def rotator(ra1, dec1, ra2, dec2, ra3, dec3):
    """
    Rotate vectors pointing to directions given by pairs `(ra3, dec3)` by the
    rotation defined by going from `(ra1, dec1)` to `(ra2, dec2)`.

    `ra`, `dec` are per event right-ascension in :math:`[0, 2\pi]` and
    declination in :math:`[-\pi/2, \pi/2]`, both in radians.

    Parameters
    ----------
    ra1, dec1 : array-like, shape (nevts)
        The points we start the rotation at.
    ra2, dec2 : array-like, shape (nevts)
        The points we end the rotation at.
    ra3, dec3 : array-like, shape (nevts)
        The points we actually rotate around the axis defined by the directions
        above.

    Returns
    -------
    ra3t, dec3t : array-like, shape (nevts)
        The rotated directions `(ra3, dec3) -> (ra3t, dec3t)`.

    Notes
    -----
    Using quaternion rotation from [1]_. Was a good way to recap this stuff.
    If you are keen, you can show that this is the same as the rotation
    matrix formalism used in skylabs rotator.

    Alternative ways to express the quaternion conjugate product:

    .. code-block::
       A) ((q0**2 - np.sum(qv * qv, axis=1).reshape(qv.shape[0], 1)) * rv +
            2 * q0 * np.cross(qv, rv) +
            2 * np.sum(qv * rv, axis=1).reshape(len(qv), 1) * qv)

       B) rv + 2 * q0 * np.cross(qv, rv) + 2 * np.cross(qv, np.cross(qv, rv))


    .. [1] http://people.csail.mit.edu/bkph/articles/Quaternions.pdf
    """
    def ra_dec_to_quat(ra, dec):
        """
        Convert equatorial coordinates to quaternion representation.

        Parameters
        ----------
        ra, dec : array-like, shape (nevts)
            Per event right-ascension in :math:`[0, 2\pi]` and declination in
            :math:`[-\pi/2, \pi/2]`, both in radians.

        Returns
        -------
        out : array-like, shape (nevts, 4)
            One quaternion per row from each (ra, dec) pair.
        """
        x = np.cos(dec) * np.cos(ra)
        y = np.cos(dec) * np.sin(ra)
        z = np.sin(dec)
        return np.vstack((np.zeros_like(x), x, y, z)).T

    def quat_to_ra_dec(q):
        """
        Convert quaternions back to quatorial coordinates.

        Parameters
        ----------
        q : array-like, shape (nevts, 4)
            One quaternion per row to convert to a (ra, dec) pair each.

        Returns
        -------
        ra, dec : array-like, shape (nevts)
            Per event right-ascension in :math:`[0, 2\pi]` and declination in
            :math:`[-\pi/2, \pi/2]`, both in radians.
        """
        nv = norm(q[:, 1:])
        x, y, z = nv[:, 0], nv[:, 1], nv[:, 2]
        dec = np.arcsin(z)
        ra = np.arctan2(y, x)
        ra[ra < 0] += 2. * np.pi
        return ra, dec

    def norm(v):
        """
        Normalize a vector, so that the sum over the squared elements is one.

        Also valid for quaternions.

        Parameters
        ----------
        v : array-like, shape (nevts, ndim)
            One vector per row to normalize

        Returns
        -------
        nv : array-like, shape (nevts, ndim)
            Normed vectors per row.
        """
        norm = np.sqrt(np.sum(v**2, axis=1))
        m = (norm == 0.)
        norm[m] = 1.
        vn = v / norm.reshape(v.shape[0], 1)
        assert np.allclose(np.sum(vn[~m]**2, axis=1), 1.)
        return vn

    def quat_mult(p, q):
        """
        Multiply p * q in exactly this order.

        Parameters
        ----------
        p, q : array-like, shape (nevts, 4)
            Quaternions in each row to multiply.

        Returns
        -------
        out : array-like, shape (nevts, 4)
            Result of the quaternion multiplication. One quaternion per row.
        """
        p0, p1, p2, p3 = p[:, [0]], p[:, [1]], p[:, [2]], p[:, [3]]
        q0, q1, q2, q3 = q[:, [0]], q[:, [1]], q[:, [2]], q[:, [3]]
        # This algebra reflects the similarity to the rotation matrices
        a = q0 * p0 - q1 * p1 - q2 * p2 - q3 * p3
        x = q0 * p1 + q1 * p0 - q2 * p3 + q3 * p2
        y = q0 * p2 + q1 * p3 + q2 * p0 - q3 * p1
        z = q0 * p3 - q1 * p2 + q2 * p1 + q3 * p0

        return np.hstack((a, x, y, z))

    def quat_conj(q):
        """
        Get the conjugate quaternion. This means switched signs of the
        imagenary parts `(i,j,k) -> (-i,-j,-k)`.

        Parameters
        ----------
        q : array-like, shape (nevts, 4)
            One quaternion per row to conjugate.

        Returns
        -------
        out : array-like, shape (nevts, 4)
            Conjugated quaternions. One quaternion per row.
        """
        return np.hstack((q[:, [0]], -q[:, [1]], -q[:, [2]], -q[:, [3]]))

    def get_rot_quat_from_ra_dec(ra1, dec1, ra2, dec2):
        """
        Construct quaternion which defines the rotation from a vector
        pointing to `(ra1, dec1)` to another one pointing to `(ra2, dec2)`.

        The rotation quaternion has the rotation angle in it's first
        component and the axis around which is rotated in the last three
        components. The quaternion must be normed :math:`\sum(q_i^2)=1`.

        Parameters
        ----------
        ra1, dec1 : array-like, shape (nevts)
            The points we start the rotation at.
        ra2, dec2 : array-like, shape (nevts)
            The points we end the rotation at.

        Returns
        -------
        out : array-like, shape (nevts, 4)
            One quaternion per row defining the rotation axis and angle for
            each given pair of `(ra1, dec1)`, `(ra2, dec2)`.
        """
        p0 = ra_dec_to_quat(ra1, dec1)
        p1 = ra_dec_to_quat(ra2, dec2)
        # Norm rotation axis for proper quaternion normalization
        ax = norm(np.cross(p0[:, 1:], p1[:, 1:]))

        cos_dist = np.clip((np.cos(ra1 - ra2) * np.cos(dec1) * np.cos(dec2) +
                           np.sin(dec1) * np.sin(dec2)), -1., 1.)

        ang = np.arccos(cos_dist).reshape(cos_dist.shape[0], 1)
        ang /= 2.
        a = np.cos(ang)
        ax = ax * np.sin(ang)
        # Normed because: sin^2 + cos^2 * vec(ax)^2 = sin^2 + cos^2 = 1
        return np.hstack((a, ax[:, [0]], ax[:, [1]], ax[:, [2]]))

    ra1, dec1, ra2, dec2, ra3, dec3 = map(np.atleast_1d,
                                          [ra1, dec1, ra2, dec2, ra3, dec3])
    assert(len(ra1) == len(dec1) == len(ra2) == len(dec2) ==
           len(ra3) == len(dec3))

    # Convert (ra3, dec3) to imaginary quaternion -> (0, vec(ra, dec))
    q3 = ra_dec_to_quat(ra3, dec3)

    # Make rotation quaternion: (angle, vec(rot_axis)
    q_rot = get_rot_quat_from_ra_dec(ra1, dec1, ra2, dec2)

    # Rotate by multiplying q3' = q_rot * q3 * q_rot_conj
    q3t = quat_mult(q_rot, quat_mult(q3, quat_conj(q_rot)))
    # Rotations preserves vectors, so imaganery part stays zero
    assert np.allclose(q3t[:, 0], 0.)

    # And transform back to (ra3', dec3')
    return quat_to_ra_dec(q3t)
