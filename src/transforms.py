"""Petites opérations SE(3), quaternions dans la convention MuJoCo [w,x,y,z]."""
from __future__ import annotations
import numpy as np

def normalize(q):
    q = np.asarray(q, dtype=float); return q / np.linalg.norm(q)
def mul(a, b):
    aw, ax, ay, az = a; bw, bx, by, bz = b
    return np.array([aw*bw-ax*bx-ay*by-az*bz, aw*bx+ax*bw+ay*bz-az*by, aw*by-ax*bz+ay*bw+az*bx, aw*bz+ax*by-ay*bx+az*bw])
def inv(q): return np.array([q[0], -q[1], -q[2], -q[3]]) / np.dot(q, q)
def rotate(q, v): return mul(mul(q, np.r_[0., v]), inv(q))[1:]
def compose(a, b):
    """T_ac = T_ab @ T_bc, poses as (position, quaternion)."""
    return a[0] + rotate(a[1], b[0]), normalize(mul(a[1], b[1]))
def inverse(p): return -rotate(inv(p[1]), p[0]), inv(p[1])
def relative(reference, object_pose): return compose(inverse(reference), object_pose)
def rotvec_to_quat(v):
    angle = np.linalg.norm(v)
    if angle < 1e-12: return np.array([1., 0., 0., 0.])
    return np.r_[np.cos(angle/2), np.sin(angle/2)*np.asarray(v)/angle]
def quat_to_rotvec(q):
    q = normalize(q)
    if q[0] < 0: q = -q
    angle = 2*np.arctan2(np.linalg.norm(q[1:]), q[0])
    return np.zeros(3) if angle < 1e-12 else q[1:] * angle / np.linalg.norm(q[1:])
def euler_xyz_to_quat(angles):
    q = np.array([1., 0., 0., 0.])
    for axis, angle in zip(np.eye(3), angles): q = mul(q, rotvec_to_quat(axis*angle))
    return normalize(q)
