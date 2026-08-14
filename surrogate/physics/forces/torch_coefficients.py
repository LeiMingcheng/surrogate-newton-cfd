"""
Torch-differentiable force coefficient computation on O-grid.

Provides:
- compute_cdp_torch: Single-sample CDp (legacy, for adjoint diagnostics)
- compute_force_components_ogrid_torch: Batched ADFLOW-aligned total/components
- compute_force_coefficients_ogrid_torch: Batch CL/CD/Cm with full autograd support
  Used by Newton AoA solver for dCL/dAoA gradient computation.

Mirrors the logic in force_coefficients.py but uses pure torch operations.
"""

import math
import torch

from surrogate.physics.forces.conventions import (
    STANDARD_MOMENT_REFERENCE,
    right_hand_cmz_to_standard_cm,
)
from surrogate.physics.pde.surface_forces import (
    compute_viscous_wall_force_adflow_like_torch,
    prepare_surface_force_geometry,
)


def extract_wall_cp_ogrid_torch(
    fields: torch.Tensor,
    coords_vertex: torch.Tensor,
    flow_conditions: torch.Tensor,
    gamma: float = 1.4,
) -> tuple:
    """
    Extract differentiable wall Cp and arc-length weights on O-grid/C-grid wall line.

    Args:
        fields: (B, C, H, W) or (C, H, W) physical-space flow field. Channel 3 is pressure.
        coords_vertex: (B, 2, H+1, W+1) or (2, H+1, W+1) vertex coordinates.
        flow_conditions: (B, 3) or (3,) [Ma, AoA_deg, Re].
        gamma: Ratio of specific heats.

    Returns:
        cp_wall: (B, W) or (W,) wall Cp distribution.
        arc_weights: (B, W) or (W,) valid wall arc-length weights with wake-cut masking applied.
    """
    single_sample = fields.dim() == 3
    if single_sample:
        fields = fields.unsqueeze(0)
    if flow_conditions.dim() == 1:
        flow_conditions = flow_conditions.unsqueeze(0)

    batch_size = fields.shape[0]

    if coords_vertex.dim() == 3:
        coords_vertex = coords_vertex.unsqueeze(0).expand(batch_size, -1, -1, -1)

    p_wall = fields[:, 3, 0, :]  # (B, W)
    x_wall_v = coords_vertex[:, 0, 0, :]  # (B, W+1)
    y_wall_v = coords_vertex[:, 1, 0, :]  # (B, W+1)

    dx = x_wall_v[:, 1:] - x_wall_v[:, :-1]  # (B, W)
    dy = y_wall_v[:, 1:] - y_wall_v[:, :-1]
    ds = torch.sqrt(dx ** 2 + dy ** 2)

    with torch.no_grad():
        seg_mask = _wall_segment_mask_torch(
            x_wall_v,
            y_wall_v,
            dtype=fields.dtype,
        )

    mach = flow_conditions[:, 0].clamp_min(0.01)
    q_nondim = 0.5 * gamma * mach ** 2
    cp_wall = (p_wall - 1.0) / (q_nondim.unsqueeze(1) + 1e-12)
    arc_weights = ds.to(dtype=fields.dtype) * seg_mask

    if single_sample:
        return cp_wall[0], arc_weights[0]
    return cp_wall, arc_weights


def compute_cdp_torch(
    fields: torch.Tensor,
    coords_vertex: torch.Tensor,
    flow_conditions: torch.Tensor,
    gamma: float = 1.4,
    chord_ref: float = 1.0,
) -> torch.Tensor:
    """
    Differentiable CDp (pressure drag) computation for a single sample.

    Args:
        fields: (C, H, W) flow field in physical (nondimensional) space.
                Channel 3 is pressure.
        coords_vertex: (2, H+1, W+1) vertex coordinates.
        flow_conditions: (3,) tensor [Ma, AoA_deg, Re].
        gamma: Ratio of specific heats.
        chord_ref: Reference chord length.

    Returns:
        CDp scalar tensor with gradient support.
    """
    Ma = flow_conditions[0]
    AoA_deg = flow_conditions[1]
    AoA_rad = AoA_deg * (math.pi / 180.0)

    # Wall pressure (j=0 row)
    p_wall = fields[3, 0, :]  # (W,)

    # Wall vertex coordinates
    x_wall_v = coords_vertex[0, 0, :]  # (W+1,)
    y_wall_v = coords_vertex[1, 0, :]  # (W+1,)

    # Segment vectors and lengths
    dx = x_wall_v[1:] - x_wall_v[:-1]  # (W,)
    dy = y_wall_v[1:] - y_wall_v[:-1]  # (W,)
    ds = torch.sqrt(dx ** 2 + dy ** 2)  # (W,)

    # Outward normal direction (orientation from signed area)
    signed_area = 0.5 * torch.sum(
        x_wall_v[:-1] * y_wall_v[1:] - x_wall_v[1:] * y_wall_v[:-1]
    )
    orient = torch.sign(signed_area)  # +1 for CCW, -1 for CW
    # If exactly zero (degenerate), default to +1
    orient = torch.where(orient == 0, torch.ones_like(orient), orient)

    nx = orient * dy / (ds + 1e-12)
    ny = orient * (-dx) / (ds + 1e-12)

    # C-grid wake cut mask (depends only on coords, no gradient needed)
    with torch.no_grad():
        y_eps = 1e-6
        surface_vmask = torch.abs(y_wall_v) > y_eps
        seg_mask = torch.ones_like(ds, dtype=torch.bool)
        if surface_vmask.any():
            x_max_surface = x_wall_v[surface_vmask].max()
            x_min_surface = x_wall_v[surface_vmask].min()
            chord_like = torch.clamp(x_max_surface - x_min_surface, min=1e-6)
            x_margin = torch.clamp(0.01 * chord_like, min=1e-3)
            x_thresh = x_max_surface + x_margin
            has_cut = ((x_wall_v > x_thresh) & (torch.abs(y_wall_v) <= y_eps)).any()
            if has_cut:
                seg_mask = (x_wall_v[:-1] <= x_thresh) & (x_wall_v[1:] <= x_thresh)

    # Pressure coefficient
    q_nondim = 0.5 * gamma * Ma ** 2
    Cp = (p_wall - 1.0) / (q_nondim + 1e-12)

    # Masked integration (use seg_mask as float multiplier for differentiability)
    mask_f = seg_mask.float()
    Fx_body = -torch.sum(Cp * nx * ds * mask_f)
    Fy_body = -torch.sum(Cp * ny * ds * mask_f)

    cos_a = torch.cos(AoA_rad)
    sin_a = torch.sin(AoA_rad)

    CDp = (Fx_body * cos_a + Fy_body * sin_a) / chord_ref

    return CDp


def _wall_segment_mask_torch(
    x_wall_v: torch.Tensor,
    y_wall_v: torch.Tensor,
    *,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Return the batched airfoil-wall mask without host-side batch loops."""
    y_eps = 1.0e-6
    surface_vertices = torch.abs(y_wall_v) > y_eps
    has_surface = surface_vertices.any(dim=1)
    positive_inf = torch.full_like(x_wall_v, torch.inf)
    negative_inf = torch.full_like(x_wall_v, -torch.inf)
    x_min = torch.where(surface_vertices, x_wall_v, positive_inf).amin(dim=1)
    x_max = torch.where(surface_vertices, x_wall_v, negative_inf).amax(dim=1)
    x_min = torch.where(has_surface, x_min, x_wall_v.amin(dim=1))
    x_max = torch.where(has_surface, x_max, x_wall_v.amax(dim=1))
    chord_like = torch.clamp(x_max - x_min, min=1.0e-6)
    x_threshold = x_max + torch.clamp(0.01 * chord_like, min=1.0e-3)
    has_cut = (
        (x_wall_v > x_threshold[:, None])
        & (torch.abs(y_wall_v) <= y_eps)
    ).any(dim=1)
    surface_segments = (
        (x_wall_v[:, :-1] <= x_threshold[:, None])
        & (x_wall_v[:, 1:] <= x_threshold[:, None])
    )
    return torch.where(
        has_cut[:, None],
        surface_segments,
        torch.ones_like(surface_segments),
    ).to(dtype=dtype)


def _batch_scalar(
    value,
    *,
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
    name: str,
) -> torch.Tensor:
    tensor = torch.as_tensor(value, device=device, dtype=dtype)
    if tensor.numel() == 1:
        return tensor.reshape(1).expand(batch_size)
    if tensor.numel() == batch_size:
        return tensor.reshape(batch_size)
    raise ValueError(f"{name} must be scalar or contain one value per batch sample")


def _batch_moment_center(
    value,
    *,
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    tensor = torch.as_tensor(value, device=device, dtype=dtype)
    if tensor.shape == (2,):
        return tensor.unsqueeze(0).expand(batch_size, -1)
    if tensor.shape == (batch_size, 2):
        return tensor
    raise ValueError(
        "moment_center must have shape (2,) or (B, 2) and uses the same "
        "absolute coordinate system as ADFLOW xRef/yRef"
    )


def compute_force_components_ogrid_torch(
    fields: torch.Tensor,
    coords_vertex: torch.Tensor,
    flow_conditions: torch.Tensor,
    gamma: float = 1.4,
    chord_ref: float = 1.0,
    area_ref=None,
    moment_center: tuple = STANDARD_MOMENT_REFERENCE,
    compute_viscous: bool = True,
    T_inf: float = 300.0,
    prepared_geometry=None,
) -> dict[str, torch.Tensor]:
    """
    Compute ADFLOW-aligned force components for an O-grid batch.

    Pressure integration and the ADFLOW-like viscous wall-traction path are
    fully batched on the input device. Reuse ``prepared_geometry`` for repeated
    evaluations on the same mesh to avoid rebuilding metric and halo geometry.

    Args:
        fields: (B, C, H, W) flow field [rho, u, v, p, ...]. C >= 4.
        coords_vertex: (B, 2, H+1, W+1) or (2, H+1, W+1) vertex coords.
        flow_conditions: (B, 3) [Ma, AoA_deg, Re].
        gamma: Ratio of specific heats.
        chord_ref: ADFLOW ``chordRef``; scalar or one value per sample.
        area_ref: ADFLOW ``areaRef``; defaults to ``chord_ref`` for a 2D
            unit-span section.
        moment_center: ADFLOW ``(xRef, yRef)`` in absolute mesh coordinates,
            either shape ``(2,)`` or ``(B, 2)``.
        compute_viscous: Whether to compute viscous lift, drag, and moment.
        T_inf: Reference temperature (K) for Sutherland viscosity.
        prepared_geometry: Optional output of ``prepare_surface_force_geometry``.

    Returns:
        Dictionary containing total and pressure/viscous component tensors.
    """
    del T_inf  # The shared viscous-flux kernel owns the temperature convention.
    single_sample = fields.dim() == 3
    if single_sample:
        fields = fields.unsqueeze(0)
    if fields.dim() != 4:
        raise ValueError(f"fields must have shape (C,H,W) or (B,C,H,W), got {tuple(fields.shape)}")
    if flow_conditions.dim() == 1:
        flow_conditions = flow_conditions.unsqueeze(0)

    B = fields.shape[0]
    device = fields.device
    dtype = fields.dtype
    flow_conditions = flow_conditions.to(device=device, dtype=dtype)
    if flow_conditions.shape[0] == 1 and B > 1:
        flow_conditions = flow_conditions.expand(B, -1)
    if flow_conditions.shape[0] != B:
        raise ValueError("flow_conditions batch dimension must match fields")

    coords_vertex = coords_vertex.to(device=device)
    coords_for_geometry = coords_vertex
    if coords_vertex.dim() == 3:
        coords_vertex = coords_vertex.unsqueeze(0).expand(B, -1, -1, -1)
    elif coords_vertex.dim() != 4 or coords_vertex.shape[0] != B:
        raise ValueError("coords_vertex must have shape (2,H+1,W+1) or (B,2,H+1,W+1)")

    Ma = flow_conditions[:, 0]          # (B,)
    AoA_deg = flow_conditions[:, 1]     # (B,)
    AoA_rad = AoA_deg * (math.pi / 180.0)

    # Wall fields (j=0)
    p_wall = fields[:, 3, 0, :]         # (B, W)

    # Wall vertex coordinates
    x_wall_v = coords_vertex[:, 0, 0, :]  # (B, W+1)
    y_wall_v = coords_vertex[:, 1, 0, :]  # (B, W+1)

    # Segment vectors
    dx = x_wall_v[:, 1:] - x_wall_v[:, :-1]  # (B, W)
    dy = y_wall_v[:, 1:] - y_wall_v[:, :-1]
    ds = torch.sqrt(dx**2 + dy**2)

    # Wall cell centers
    x_wall_c = 0.5 * (x_wall_v[:, :-1] + x_wall_v[:, 1:])  # (B, W)
    y_wall_c = 0.5 * (y_wall_v[:, :-1] + y_wall_v[:, 1:])

    # Outward normal (orientation from signed area)
    signed_area = 0.5 * torch.sum(
        x_wall_v[:, :-1] * y_wall_v[:, 1:] - x_wall_v[:, 1:] * y_wall_v[:, :-1],
        dim=1,
    )  # (B,)
    orient = torch.sign(signed_area)
    orient = torch.where(orient == 0, torch.ones_like(orient), orient)
    orient = orient.unsqueeze(1)  # (B, 1)

    nx = orient * dy / (ds + 1e-12)   # (B, W)
    ny = orient * (-dx) / (ds + 1e-12)

    with torch.no_grad():
        seg_mask = _wall_segment_mask_torch(
            x_wall_v,
            y_wall_v,
            dtype=fields.dtype,
        )

    # Pressure coefficient
    q_nondim = 0.5 * gamma * Ma**2  # (B,)
    Cp = (p_wall - 1.0) / (q_nondim.unsqueeze(1) + 1e-12)  # (B, W)

    # Pressure force integration
    Fx_p = -torch.sum(Cp * nx * ds * seg_mask, dim=1)  # (B,)
    Fy_p = -torch.sum(Cp * ny * ds * seg_mask, dim=1)

    cos_a = torch.cos(AoA_rad)  # (B,)
    sin_a = torch.sin(AoA_rad)

    chord = _batch_scalar(
        chord_ref,
        batch_size=B,
        device=device,
        dtype=dtype,
        name="chord_ref",
    )
    area = _batch_scalar(
        chord_ref if area_ref is None else area_ref,
        batch_size=B,
        device=device,
        dtype=dtype,
        name="area_ref",
    )
    reference_point = _batch_moment_center(
        moment_center,
        batch_size=B,
        device=device,
        dtype=dtype,
    )
    x_ref = reference_point[:, 0]
    y_ref = reference_point[:, 1]

    CLp = (Fy_p * cos_a - Fx_p * sin_a) / area
    CDp = (Fx_p * cos_a + Fy_p * sin_a) / area
    pressure_face_x = -Cp * nx * ds * seg_mask
    pressure_face_y = -Cp * ny * ds * seg_mask
    Mp_right_hand = torch.sum(
        (x_wall_c - x_ref[:, None]) * pressure_face_y
        - (y_wall_c - y_ref[:, None]) * pressure_face_x,
        dim=1,
    )
    Cmp = right_hand_cmz_to_standard_cm(Mp_right_hand / (area * chord))

    Fx_v = torch.zeros(B, device=device, dtype=dtype)
    Fy_v = torch.zeros(B, device=device, dtype=dtype)
    Mv_right_hand = torch.zeros(B, device=device, dtype=dtype)
    wall_cf = torch.zeros_like(Cp)
    if compute_viscous:
        if prepared_geometry is None:
            geometry_coords = coords_for_geometry
            if geometry_coords.dim() == 3:
                geometry_coords = geometry_coords.unsqueeze(0).expand(B, -1, -1, -1)
            prepared_geometry = prepare_surface_force_geometry(
                geometry_coords,
                periodic_xi=True,
                device=device,
                dtype=torch.float64,
            )
        viscous_force = compute_viscous_wall_force_adflow_like_torch(
            fields=fields,
            flow_conditions=flow_conditions,
            prepared_geometry=prepared_geometry,
            gamma=gamma,
            seg_mask=seg_mask.to(dtype=torch.float64),
            return_details=True,
        )
        Fx_v = viscous_force["Fx_v"].to(dtype=dtype)
        Fy_v = viscous_force["Fy_v"].to(dtype=dtype)
        wall_vx = viscous_force["wall_vx"].to(dtype=dtype)
        wall_vy = viscous_force["wall_vy"].to(dtype=dtype)
        Mv_right_hand = torch.sum(
            (
                (x_wall_c - x_ref[:, None]) * wall_vy
                - (y_wall_c - y_ref[:, None]) * wall_vx
            )
            * seg_mask,
            dim=1,
        )
        # ``wall_v*`` is the face-integrated viscous body force. Divide by
        # face length and dynamic pressure, then project onto a surface
        # tangent oriented from leading edge to trailing edge. Positive Cf
        # therefore denotes downstream wall shear on either surface.
        tangent_x = dx / (ds + 1.0e-12)
        tangent_y = dy / (ds + 1.0e-12)
        downstream_sign = torch.where(
            tangent_x < 0.0,
            -torch.ones_like(tangent_x),
            torch.ones_like(tangent_x),
        )
        tangent_x = tangent_x * downstream_sign
        tangent_y = tangent_y * downstream_sign
        wall_cf = (
            (wall_vx * tangent_x + wall_vy * tangent_y)
            / (ds + 1.0e-12)
            / (q_nondim[:, None] + 1.0e-12)
        ) * seg_mask

    viscous_scale = q_nondim * area + 1.0e-12
    CLv = (Fy_v * cos_a - Fx_v * sin_a) / viscous_scale
    CDv = (Fx_v * cos_a + Fy_v * sin_a) / viscous_scale
    Cmv = right_hand_cmz_to_standard_cm(
        Mv_right_hand / (q_nondim * area * chord + 1.0e-12)
    )

    result = {
        "CL": CLp + CLv,
        "CD": CDp + CDv,
        "Cm": Cmp + Cmv,
        "CLp": CLp,
        "CLv": CLv,
        "CDp": CDp,
        "CDv": CDv,
        "Cmp": Cmp,
        "Cmv": Cmv,
        "Fx_p": Fx_p,
        "Fy_p": Fy_p,
        "Fx_v": Fx_v,
        "Fy_v": Fy_v,
        "M_p": right_hand_cmz_to_standard_cm(Mp_right_hand),
        "M_v": right_hand_cmz_to_standard_cm(Mv_right_hand),
        "Cp": Cp,
        "Cf": wall_cf,
        "wall_x": x_wall_c,
        "wall_y": y_wall_c,
    }
    if single_sample:
        return {name: value.squeeze(0) for name, value in result.items()}
    return result


def compute_force_coefficients_ogrid_torch(
    fields: torch.Tensor,
    coords_vertex: torch.Tensor,
    flow_conditions: torch.Tensor,
    gamma: float = 1.4,
    chord_ref: float = 1.0,
    area_ref=None,
    moment_center: tuple = STANDARD_MOMENT_REFERENCE,
    compute_viscous: bool = True,
    T_inf: float = 300.0,
    prepared_geometry=None,
) -> tuple:
    """Compatibility tuple API returning ADFLOW-aligned total ``(CL, CD, Cm)``."""
    result = compute_force_components_ogrid_torch(
        fields=fields,
        coords_vertex=coords_vertex,
        flow_conditions=flow_conditions,
        gamma=gamma,
        chord_ref=chord_ref,
        area_ref=area_ref,
        moment_center=moment_center,
        compute_viscous=compute_viscous,
        T_inf=T_inf,
        prepared_geometry=prepared_geometry,
    )
    return result["CL"], result["CD"], result["Cm"]
