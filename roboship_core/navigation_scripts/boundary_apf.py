"""
boundary_apf.py - Artificial Potential Field boundary for Roboship USV

Drop-in replacement for the BoundaryPolygon class used in los_guidance.py,
ilos_guidance.py, and pure_pursuit.py. Instead of linearly scaling surge
speed to zero near the boundary (which cannot prevent drift-out if the
guidance law points toward the wall), this class generates an active
repulsive velocity vector that pushes the vessel away from the boundary
AND turns it back toward the polygon interior.

The repulsive potential follows Khatib (1986) [1]:

    U_rep(d) = 0.5 * eta * (1/d - 1/d0)^2     if d < d0
    U_rep(d) = 0                                 if d >= d0

The repulsive force (negative gradient of U_rep) is:

    F_rep = eta * (1/d - 1/d0) * (1/d^2) * n_hat

where d is the signed distance to the nearest edge, d0 is the influence
distance (margin), and n_hat is the inward unit normal of the nearest edge.

Integration into guidance nodes
-------------------------------
`apply_to_command` decomposes the world-frame repulsive force into:
    1. Surge correction along the heading direction.
    2. Lateral correction perpendicular to the heading, converted via
       a gain into a yaw rate correction.

The result is added to the guidance law's (U_cmd, r_cmd) when inside the
margin band. When OUTSIDE the polygon (sd <= 0), the APF takes over
entirely: it overrides r_cmd to steer the vessel toward the inward
normal and only commands forward surge once the heading is roughly
aligned with that direction.

References
----------
[1] Khatib, O. (1986). "Real-Time Obstacle Avoidance for Manipulators and
    Mobile Robots." International Journal of Robotics Research, 5(1), 90-98.

Tuning parameters
-----------------
eta     : Repulsive gain. Higher = stronger push-back inside the margin.
          0.3 is a reasonable default for the mocap volume.
margin  : Influence distance d0 in metres. The repulsive field activates
          when the vessel is closer than this to any edge.
          0.3-0.5 m is typical for the mocap volume.
"""

import math


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _wrap(a: float) -> float:
    return math.atan2(math.sin(a), math.cos(a))


def dist_point_to_segment(px, py, x1, y1, x2, y2):
    """Shortest distance from point (px, py) to line segment (x1,y1)-(x2,y2).
    Returns (distance, nearest_x, nearest_y)."""
    dx, dy = x2 - x1, y2 - y1
    len_sq = dx * dx + dy * dy
    if len_sq < 1e-12:
        return math.hypot(px - x1, py - y1), x1, y1
    t = max(0.0, min(1.0, ((px - x1) * dx + (py - y1) * dy) / len_sq))
    cx = x1 + t * dx
    cy = y1 + t * dy
    return math.hypot(px - cx, py - cy), cx, cy


# ---------------------------------------------------------------------------
# Boundary polygon with APF
# ---------------------------------------------------------------------------

class BoundaryPolygon:
    """Polygon boundary with artificial potential field repulsion.

    Parameters
    ----------
    vertices : list of (x, y) tuples
        Polygon vertices, ordered CW or CCW.
    margin : float
        Influence distance d0 in metres. Repulsive field activates
        when the vessel is closer than this to any edge.
    eta : float
        Repulsive gain. Controls the magnitude of the push-back
        velocity.
    f_max : float
        Hard cap on the repulsive force magnitude [m/s]. Prevents the
        APF from commanding physically silly velocities while still
        allowing it to dominate the guidance command.
    """

    def __init__(self, vertices: list, margin: float,
                 eta: float = 0.3, f_max: float = 1.0):
        self.verts = list(vertices)
        self.n = len(self.verts)
        self.margin = margin
        self.eta = eta
        self.f_max = f_max

        if self.n < 3:
            raise ValueError(f'Polygon needs >= 3 vertices, got {self.n}')

        # Detect winding via signed area (positive = CCW, negative = CW)
        area2 = 0.0
        for i in range(self.n):
            x1, y1 = self.verts[i]
            x2, y2 = self.verts[(i + 1) % self.n]
            area2 += x1 * y2 - x2 * y1
        sign = 1.0 if area2 > 0.0 else -1.0

        # Precompute edges with inward unit normals
        self.edges = []  # (x1, y1, x2, y2, nx, ny)
        for i in range(self.n):
            x1, y1 = self.verts[i]
            x2, y2 = self.verts[(i + 1) % self.n]
            edx, edy = x2 - x1, y2 - y1
            length = math.sqrt(edx * edx + edy * edy)
            if length < 1e-9:
                continue
            nx = sign * (-edy) / length
            ny = sign * (edx) / length
            self.edges.append((x1, y1, x2, y2, nx, ny))

    # ------------------------------------------------------------------
    # Geometry queries
    # ------------------------------------------------------------------

    def point_inside(self, px: float, py: float) -> bool:
        """Ray-casting point-in-polygon test."""
        inside = False
        j = self.n - 1
        for i in range(self.n):
            xi, yi = self.verts[i]
            xj, yj = self.verts[j]
            if ((yi > py) != (yj > py)) and \
               (px < (xj - xi) * (py - yi) / (yj - yi) + xi):
                inside = not inside
            j = i
        return inside

    def signed_distance(self, px: float, py: float):
        """Signed min distance from (px, py) to polygon boundary.
        Positive = inside, negative = outside.
        Returns (signed_dist, nearest_edge_index)."""
        min_dist = float('inf')
        nearest_idx = 0
        for i, (x1, y1, x2, y2, _nx, _ny) in enumerate(self.edges):
            d, _, _ = dist_point_to_segment(px, py, x1, y1, x2, y2)
            if d < min_dist:
                min_dist = d
                nearest_idx = i
        if not self.point_inside(px, py):
            min_dist = -min_dist
        return min_dist, nearest_idx

    # ------------------------------------------------------------------
    # Repulsive field
    # ------------------------------------------------------------------

    def repulsive_force(self, px: float, py: float):
        """Compute APF repulsive velocity vector (world frame).

        Returns (fx, fy, signed_dist, nearest_edge_index).

            INSIDE polygon, sd < margin    : Khatib (1986) form
            OUTSIDE polygon, sd <= 0       : constant + linear ramp inward
                                              with magnitude growing with
                                              depth outside.
            INSIDE polygon, sd >= margin   : zero (free region)
        """
        sd, nearest_idx = self.signed_distance(px, py)
        nx, ny = self.edges[nearest_idx][4], self.edges[nearest_idx][5]

        if sd >= self.margin:
            # Outside influence zone: no repulsion
            return 0.0, 0.0, sd, nearest_idx

        if sd <= 0.0:
            # OUTSIDE polygon. Strong inward push that grows linearly
            # with how far out the vessel is, so deep excursions are
            # recovered faster.
            depth_out = -sd
            magnitude = self.eta * (10.0 + 5.0 * depth_out)
            magnitude = min(magnitude, self.f_max)
            return magnitude * nx, magnitude * ny, sd, nearest_idx

        # Inside polygon, within margin: classical Khatib form.
        # F = eta * (1/d - 1/d0) * (1/d^2) * n_hat
        d = max(sd, 0.05)         # clamp to avoid 1/d singularity
        d0 = self.margin
        magnitude = self.eta * (1.0 / d - 1.0 / d0) * (1.0 / (d * d))
        magnitude = min(magnitude, self.f_max)
        return magnitude * nx, magnitude * ny, sd, nearest_idx

    # ------------------------------------------------------------------
    # Command modifier
    # ------------------------------------------------------------------

    def apply_to_command(self, px, py, psi, U_cmd, r_cmd,
                         K_repulse_yaw: float = 2.0,
                         U_recover: float = 0.15):
        """Modify a (U_cmd, r_cmd) pair to respect the boundary.

        Inside the margin band, the repulsive force is decomposed into a
        body-frame surge correction (added to U_cmd) and a body-frame
        lateral component (converted to a yaw rate correction via
        K_repulse_yaw).

        Outside the polygon, the guidance command is completely
        overridden: the boat is steered toward the inward normal at
        K_repulse_yaw gain, and drives forward at U_recover only when
        roughly aligned with that direction.

        Parameters
        ----------
        px, py         : vessel position in world frame [m]
        psi            : vessel heading [rad]
        U_cmd, r_cmd   : commanded surge speed [m/s] and yaw rate [rad/s]
                         from the guidance law.
        K_repulse_yaw  : gain converting body-frame lateral repulsion
                         to a yaw rate correction [rad/s per m/s].
        U_recover      : forward speed used during outside-boundary
                         recovery once roughly facing inward.

        Returns
        -------
        U_out, r_out, signed_dist, fx, fy
        """
        fx, fy, sd, _ = self.repulsive_force(px, py)

        if abs(fx) < 1e-9 and abs(fy) < 1e-9:
            # Free region: no modification
            return U_cmd, r_cmd, sd, 0.0, 0.0

        # Body-frame decomposition of the repulsive force.
        c = math.cos(psi)
        s = math.sin(psi)
        f_surge   =  c * fx + s * fy     # + = along heading
        f_lateral = -s * fx + c * fy     # + = to port (left of heading)

        if sd <= 0.0:
            # OUTSIDE polygon: recovery mode. Ignore guidance, steer
            # toward inward normal, drive forward only when aligned.
            psi_inward = math.atan2(fy, fx)
            err = _wrap(psi_inward - psi)
            r_out = K_repulse_yaw * err
            if abs(err) < math.pi / 3:        # within 60 deg of inward
                U_out = U_recover * math.cos(err)
            else:
                U_out = 0.0
            return U_out, r_out, sd, fx, fy

        # INSIDE margin band: add repulsion to guidance command.
        U_out = max(0.0, U_cmd + f_surge)
        r_out = r_cmd + K_repulse_yaw * f_lateral
        return U_out, r_out, sd, fx, fy

    # ------------------------------------------------------------------
    # Legacy interface (for backward compatibility)
    # ------------------------------------------------------------------

    def speed_factor(self, px, py, heading_rad):
        """Legacy interface. Returns (factor, signed_dist, nearest_idx).
        Kept so existing code doesn't break; prefer apply_to_command()."""
        sd, nearest_idx = self.signed_distance(px, py)
        if sd <= 0.0:
            return 0.0, sd, nearest_idx
        if sd >= self.margin:
            return 1.0, sd, nearest_idx
        factor = sd / self.margin
        _, _, _, _, nx, ny = self.edges[nearest_idx]
        hx = math.cos(heading_rad)
        hy = math.sin(heading_rad)
        dot = hx * nx + hy * ny
        if dot < 0.0:
            factor *= (1.0 + dot)
        return max(factor, 0.0), sd, nearest_idx