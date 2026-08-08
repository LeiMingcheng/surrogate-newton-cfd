"""AoA solving utilities for transport-neutral serving predictors."""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any, Optional

import numpy as np
import torch

from surrogate.physics.forces import compute_force_coefficients_ogrid_torch
from surrogate.physics.pde.surface_forces import prepare_surface_force_geometry
from surrogate.serving._tensors import as_1d_tensor, expand_geometry, expand_spatial
from surrogate.serving.contracts import AoARequest, AoAResult, PredictionRequest
from surrogate.utils.timing_profile import emit_profile_event


@dataclass
class AoASolverConfig:
    """Controls for fixed-AoA evaluation and target-CL solving."""

    aoa_range: tuple[float, float] = (-5.0, 10.0)
    max_iter: int = 15
    tol: float = 1.0e-2
    fd_step: float = 0.1
    device: str = "cuda"
    force_viscous_for_solve: bool = False
    force_viscous_for_final: bool = True
    gamma: float = 1.4
    chord_ref: float = 1.0
    area_ref: float = 1.0
    moment_center: tuple[float, float] = (0.0, 0.0)
    t_inf: float = 300.0

    def validate(self) -> None:
        if len(self.aoa_range) != 2 or float(self.aoa_range[0]) >= float(self.aoa_range[1]):
            raise ValueError("aoa_range must contain increasing lower and upper bounds")
        if int(self.max_iter) <= 0:
            raise ValueError("max_iter must be positive")
        if float(self.tol) <= 0:
            raise ValueError("tol must be positive")
        if float(self.fd_step) <= 0:
            raise ValueError("fd_step must be positive")
        if float(self.gamma) <= 0 or float(self.chord_ref) <= 0 or float(self.area_ref) <= 0 or float(self.t_inf) <= 0:
            raise ValueError("gamma, chord_ref, area_ref, and t_inf must be positive")


def _resolve_reynolds(value: Any, *, count: int, device: torch.device) -> torch.Tensor:
    reynolds = as_1d_tensor(value, device=device)
    if int(reynolds.numel()) == 1:
        return reynolds.repeat(int(count))
    if int(reynolds.numel()) != int(count):
        raise ValueError(f"reynolds must be scalar or length {count}, got {int(reynolds.numel())}")
    return reynolds

def _flow_conditions(mach: torch.Tensor, aoa: torch.Tensor, reynolds: torch.Tensor) -> torch.Tensor:
    if not (mach.shape == aoa.shape == reynolds.shape):
        raise ValueError("mach, aoa, and reynolds tensors must have matching shapes")
    return torch.stack([mach, aoa, reynolds], dim=1).float()


def _to_numpy(value: torch.Tensor) -> Any:
    return value.detach().cpu().numpy()


@dataclass(frozen=True)
class _PreparedAoARequest:
    count: int
    geometry: torch.Tensor
    coords: torch.Tensor
    coords_vertex: torch.Tensor
    mach: torch.Tensor
    reynolds: torch.Tensor
    initial_field: Optional[torch.Tensor]


class SurrogateAoASolver:
    """Solve AoA using a serving predictor plus force coefficient evaluation."""

    def __init__(self, predictor: Any, *, config: Optional[AoASolverConfig] = None) -> None:
        self.predictor = predictor
        self.config = config or AoASolverConfig()
        self.config.validate()
        self.device = torch.device(self.config.device)
        self._force_geometry_cache_key = None
        self._force_geometry_cache = None

    def _prepared_force_geometry(self, coords_vertex: torch.Tensor):
        key = (
            int(coords_vertex.data_ptr()),
            0 if coords_vertex.is_inference() else int(coords_vertex._version),
            tuple(coords_vertex.shape),
            coords_vertex.dtype,
            coords_vertex.device,
        )
        if key != self._force_geometry_cache_key:
            self._force_geometry_cache = prepare_surface_force_geometry(
                coords_vertex,
                periodic_xi=True,
                device=self.device,
                dtype=torch.float64,
            )
            self._force_geometry_cache_key = key
        return self._force_geometry_cache

    def _prepare_request(self, request: AoARequest) -> _PreparedAoARequest:
        mach = as_1d_tensor(request.mach, device=self.device)
        count = int(mach.numel())
        initial_field = None
        if request.initial_field is not None:
            initial_field = expand_spatial(
                request.initial_field,
                count=count,
                device=self.device,
                dtype=torch.float32,
                name="initial_field",
            )
        return _PreparedAoARequest(
            count=count,
            geometry=expand_geometry(request.geometry, count=count, device=self.device),
            coords=expand_spatial(
                request.coords,
                count=count,
                device=self.device,
                dtype=torch.float32,
                name="coords",
            ),
            coords_vertex=expand_spatial(
                request.coords_vertex,
                count=count,
                device=self.device,
                dtype=torch.float64,
                name="coords_vertex",
            ),
            mach=mach,
            reynolds=_resolve_reynolds(request.reynolds, count=count, device=self.device),
            initial_field=initial_field,
        )

    def _aoa_values(self, request: AoARequest, *, count: int, allow_default: bool) -> torch.Tensor:
        if request.aoa is None:
            if not allow_default:
                raise ValueError("Fixed-AoA evaluation requires request.aoa")
            lo, hi = self.config.aoa_range
            return torch.full(
                (count,),
                (float(lo) + float(hi)) * 0.5,
                dtype=torch.float32,
                device=self.device,
            )
        aoa = as_1d_tensor(request.aoa, device=self.device)
        if int(aoa.numel()) != count:
            label = "initial aoa" if allow_default else "aoa"
            raise ValueError(f"{label} must have {count} values, got {int(aoa.numel())}")
        if allow_default:
            return torch.clamp(
                aoa,
                float(self.config.aoa_range[0]),
                float(self.config.aoa_range[1]),
            )
        return aoa

    def _combine_requests(
        self,
        requests: list[AoARequest],
        *,
        target_cl_mode: bool,
    ) -> tuple[AoARequest, list[int], Optional[list[torch.Tensor]]]:
        prepared = [self._prepare_request(request) for request in requests]
        counts = [item.count for item in prepared]
        initial_fields = [item.initial_field for item in prepared]
        if any(value is None for value in initial_fields) and not all(
            value is None for value in initial_fields
        ):
            raise ValueError(
                "Batched AoA requests must either all provide initial_field or all omit it"
            )
        combined_initial = None
        if all(value is not None for value in initial_fields):
            combined_initial = torch.cat(
                [value for value in initial_fields if value is not None],
                dim=0,
            )

        target_cls = None
        if target_cl_mode:
            target_cls = []
            for request, count in zip(requests, counts):
                if request.target_cl is None:
                    raise ValueError("Target-CL AoA solve requires request.target_cl")
                target_cl = as_1d_tensor(request.target_cl, device=self.device)
                if int(target_cl.numel()) != count:
                    raise ValueError(
                        f"target_cl must have {count} values, got {int(target_cl.numel())}"
                    )
                target_cls.append(target_cl)

        combined = AoARequest(
            geometry=torch.cat([item.geometry for item in prepared], dim=0),
            coords=torch.cat([item.coords for item in prepared], dim=0),
            coords_vertex=torch.cat([item.coords_vertex for item in prepared], dim=0),
            mach=torch.cat([item.mach for item in prepared]),
            reynolds=torch.cat([item.reynolds for item in prepared]),
            target_cl=None if target_cls is None else torch.cat(target_cls),
            aoa=torch.cat(
                [
                    self._aoa_values(request, count=count, allow_default=target_cl_mode)
                    for request, count in zip(requests, counts)
                ]
            ),
            initial_field=combined_initial,
            metadata=dict(requests[0].metadata),
        )
        return combined, counts, target_cls

    def _predict(
        self,
        *,
        geometry: torch.Tensor,
        flow_conditions: torch.Tensor,
        coords: torch.Tensor,
        initial_field: Any = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> torch.Tensor:
        started = time.perf_counter()
        response = self.predictor.predict(
            PredictionRequest(
                geometry=geometry,
                flow_conditions=flow_conditions,
                coords=coords,
                initial_field=initial_field,
                metadata=metadata or {},
            )
        )
        fields = torch.as_tensor(response.fields, device=self.device)
        emit_profile_event(
            "surrogate_predict",
            samples=int(flow_conditions.shape[0]),
            wall_time_s=float(time.perf_counter() - started),
        )
        return fields

    def _forces(
        self,
        fields: torch.Tensor,
        coords_vertex: torch.Tensor,
        flow_conditions: torch.Tensor,
        *,
        compute_viscous: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        coords_vertex = coords_vertex.to(device=self.device, dtype=torch.float64)
        coefficients = compute_force_coefficients_ogrid_torch(
            fields.float().to(self.device),
            coords_vertex,
            flow_conditions.to(device=self.device, dtype=torch.float32),
            gamma=self.config.gamma,
            chord_ref=self.config.chord_ref,
            area_ref=self.config.area_ref,
            moment_center=self.config.moment_center,
            compute_viscous=compute_viscous,
            T_inf=self.config.t_inf,
            prepared_geometry=(
                self._prepared_force_geometry(coords_vertex)
                if compute_viscous
                else None
            ),
        )
        return tuple(
            coefficient.to(device=self.device, dtype=flow_conditions.dtype)
            for coefficient in coefficients
        )

    def evaluate_fixed(self, request: AoARequest) -> AoAResult:
        """Evaluate fields and forces at explicitly provided AoA values."""
        prepared = self._prepare_request(request)
        aoa = self._aoa_values(request, count=prepared.count, allow_default=False)
        flow_conditions = _flow_conditions(prepared.mach, aoa, prepared.reynolds)

        with torch.no_grad():
            fields = self._predict(
                geometry=prepared.geometry,
                flow_conditions=flow_conditions,
                coords=prepared.coords,
                initial_field=prepared.initial_field,
                metadata=dict(request.metadata),
            )
            cl, cd, cm = self._forces(
                fields,
                prepared.coords_vertex,
                flow_conditions,
                compute_viscous=self.config.force_viscous_for_final,
            )
        return AoAResult(
            aoa=_to_numpy(aoa),
            fields=fields,
            cl=_to_numpy(cl),
            cd=_to_numpy(cd),
            cm=_to_numpy(cm),
            converged=True,
            n_iter=0,
            converged_mask=np.ones((prepared.count,), dtype=bool),
            metadata=dict(request.metadata),
        )

    def evaluate_fixed_batch(self, requests: list[AoARequest]) -> list[AoAResult]:
        """Evaluate compatible fixed-AoA requests in one model tensor batch."""

        if not requests:
            return []
        combined, counts, _ = self._combine_requests(
            requests,
            target_cl_mode=False,
        )
        result = self.evaluate_fixed(combined)
        return self._split_batched_result(
            result,
            requests=requests,
            counts=counts,
            target_cls=None,
        )

    def solve_target_cl(self, request: AoARequest) -> AoAResult:
        """Solve AoA with finite-difference secant updates to match target CL."""
        started = time.perf_counter()
        if request.target_cl is None:
            raise ValueError("Target-CL AoA solve requires request.target_cl")
        prepared = self._prepare_request(request)
        target_cl = as_1d_tensor(request.target_cl, device=self.device)
        if int(target_cl.numel()) != prepared.count:
            raise ValueError(
                f"target_cl must have {prepared.count} values, got {int(target_cl.numel())}"
            )
        aoa = self._aoa_values(request, count=prepared.count, allow_default=True)

        n_iter = 0
        fields_final: Optional[torch.Tensor] = None
        cl = torch.zeros_like(target_cl)
        frozen = torch.zeros_like(target_cl, dtype=torch.bool)
        previous_aoa = torch.zeros_like(aoa)
        previous_cl = torch.zeros_like(target_cl)

        with torch.no_grad():
            flow = _flow_conditions(prepared.mach, aoa, prepared.reynolds)
            fields_final = self._predict(
                geometry=prepared.geometry,
                flow_conditions=flow,
                coords=prepared.coords,
                initial_field=prepared.initial_field,
                metadata=dict(request.metadata),
            )
            cl, _, _ = self._forces(
                fields_final,
                prepared.coords_vertex,
                flow,
                compute_viscous=self.config.force_viscous_for_solve,
            )
            error = cl - target_cl

            for iteration in range(int(self.config.max_iter)):
                n_iter = iteration + 1
                active = (error.abs() >= float(self.config.tol)) & (~frozen)
                if not bool(active.any().item()):
                    break
                idx = active.nonzero(as_tuple=False).squeeze(1)

                if iteration == 0:
                    aoa_pert = aoa[idx] + float(self.config.fd_step)
                    flow_pert = _flow_conditions(
                        prepared.mach[idx],
                        aoa_pert,
                        prepared.reynolds[idx],
                    )
                    fields_pert = self._predict(
                        geometry=prepared.geometry[idx],
                        flow_conditions=flow_pert,
                        coords=prepared.coords[idx],
                        initial_field=(
                            None
                            if prepared.initial_field is None
                            else prepared.initial_field[idx]
                        ),
                        metadata=dict(request.metadata),
                    )
                    cl_pert, _, _ = self._forces(
                        fields_pert,
                        prepared.coords_vertex[idx],
                        flow_pert,
                        compute_viscous=self.config.force_viscous_for_solve,
                    )
                    dcl_daoa = (cl_pert - cl[idx]) / float(self.config.fd_step)
                else:
                    dcl_daoa = (cl[idx] - previous_cl[idx]) / (
                        aoa[idx] - previous_aoa[idx]
                    )
                grad_safe = torch.where(
                    dcl_daoa.abs() > 0.01,
                    dcl_daoa,
                    torch.full_like(dcl_daoa, 0.1),
                )
                update = torch.clamp(error[idx] / grad_safe, -3.0, 3.0)
                current_aoa = aoa[idx].clone()
                current_cl = cl[idx].clone()
                aoa[idx] = torch.clamp(
                    aoa[idx] - update,
                    float(self.config.aoa_range[0]),
                    float(self.config.aoa_range[1]),
                )
                stalled = torch.isclose(
                    aoa[idx],
                    current_aoa,
                    atol=1.0e-6,
                    rtol=0.0,
                )

                flow_new = _flow_conditions(
                    prepared.mach[idx],
                    aoa[idx],
                    prepared.reynolds[idx],
                )
                fields_new = self._predict(
                    geometry=prepared.geometry[idx],
                    flow_conditions=flow_new,
                    coords=prepared.coords[idx],
                    initial_field=(
                        None if prepared.initial_field is None else prepared.initial_field[idx]
                    ),
                    metadata=dict(request.metadata),
                )
                cl_new, _, _ = self._forces(
                    fields_new,
                    prepared.coords_vertex[idx],
                    flow_new,
                    compute_viscous=self.config.force_viscous_for_solve,
                )
                cl[idx] = cl_new
                error[idx] = cl_new - target_cl[idx]
                fields_final[idx] = fields_new.to(dtype=fields_final.dtype)
                previous_aoa[idx] = current_aoa
                previous_cl[idx] = current_cl
                stalled_unresolved = stalled & (error[idx].abs() >= float(self.config.tol))
                if bool(stalled_unresolved.any().item()):
                    frozen[idx[stalled_unresolved]] = True

            converged_mask = error.abs() < float(self.config.tol)
            final_flow = _flow_conditions(prepared.mach, aoa, prepared.reynolds)
            final_cl, final_cd, final_cm = self._forces(
                fields_final,
                prepared.coords_vertex,
                final_flow,
                compute_viscous=self.config.force_viscous_for_final,
            )

        emit_profile_event(
            "target_cl_solve",
            samples=int(prepared.count),
            n_iter=int(n_iter),
            converged_samples=int(converged_mask.sum().item()),
            wall_time_s=float(time.perf_counter() - started),
        )
        return AoAResult(
            aoa=_to_numpy(aoa),
            fields=fields_final,
            cl=_to_numpy(final_cl),
            cd=_to_numpy(final_cd),
            cm=_to_numpy(final_cm),
            converged=bool(converged_mask.all().item()),
            n_iter=n_iter,
            converged_mask=_to_numpy(converged_mask),
            metadata={
                **dict(request.metadata),
                "target_cl": _to_numpy(target_cl),
            },
        )

    def solve_target_cl_batch(self, requests: list[AoARequest]) -> list[AoAResult]:
        """Solve compatible target-CL requests in one model tensor batch."""

        if not requests:
            return []
        combined, counts, target_cls = self._combine_requests(
            requests,
            target_cl_mode=True,
        )
        result = self.solve_target_cl(combined)
        return self._split_batched_result(
            result,
            requests=requests,
            counts=counts,
            target_cls=target_cls,
        )

    def _split_batched_result(
        self,
        result: AoAResult,
        *,
        requests: list[AoARequest],
        counts: list[int],
        target_cls: Optional[list[torch.Tensor]],
    ) -> list[AoAResult]:
        outputs: list[AoAResult] = []
        start = 0
        aoa = np.asarray(result.aoa)
        cl = np.asarray(result.cl)
        cd = np.asarray(result.cd)
        cm = np.asarray(result.cm)
        converged_mask = result.converged_mask
        if converged_mask is None:
            converged_mask = np.full((sum(counts),), bool(result.converged), dtype=bool)
        else:
            converged_mask = np.asarray(converged_mask, dtype=bool)
        for index, (request, count) in enumerate(zip(requests, counts)):
            stop = start + count
            converged = True
            metadata = dict(request.metadata)
            if target_cls is not None:
                target_cl = _to_numpy(target_cls[index])
                converged = bool(np.all(converged_mask[start:stop]))
                metadata["target_cl"] = target_cl
            outputs.append(
                AoAResult(
                    aoa=aoa[start:stop],
                    fields=result.fields[start:stop],
                    cl=cl[start:stop],
                    cd=cd[start:stop],
                    cm=cm[start:stop],
                    converged=converged,
                    n_iter=int(result.n_iter),
                    converged_mask=converged_mask[start:stop],
                    metadata=metadata,
                )
            )
            start = stop
        return outputs

    def solve(self, request: AoARequest) -> AoAResult:
        """Dispatch to fixed-AoA or target-CL mode based on the request."""
        if request.target_cl is None:
            return self.evaluate_fixed(request)
        return self.solve_target_cl(request)


__all__ = [
    "AoASolverConfig",
    "SurrogateAoASolver",
]
