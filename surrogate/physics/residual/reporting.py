from typing import Dict, Iterable, Optional, Sequence

import torch


PDE_COMPONENTS: Sequence[str] = ("Rc", "Rmx", "Rmy", "RE", "RSA")
L2_COMPONENTS: Sequence[str] = ("totalR", "totalR0", "l2_ratio")

SUMMARY_COMPONENTS: Dict[str, Sequence[str]] = {
    "momentum3": ("Rc", "Rmx", "Rmy"),
    "energy4": ("Rc", "Rmx", "Rmy", "RE"),
    "five_channel": PDE_COMPONENTS,
}

_PER_VOLUME_KEYS = {
    "Rc": ("Rc",),
    "Rmx": ("Rmx",),
    "Rmy": ("Rmy",),
    "RE": ("RE", "RE_norm", "R_energy"),
    "RSA": ("RSA", "RSA_norm", "R_SA"),
}

_FLUX_KEYS = {
    "Rc": ("Rc_flux",),
    "Rmx": ("Rmx_flux",),
    "Rmy": ("Rmy_flux",),
    "RE": ("RE_flux",),
    "RSA": ("RSA_flux",),
}


def make_residual_totals() -> Dict[str, float]:
    return {key: 0.0 for key in (*PDE_COMPONENTS, "residual_score")}


def make_l2_totals() -> Dict[str, float]:
    return {key: 0.0 for key in L2_COMPONENTS}


def _extract_raw_value(residual_dict: Optional[Dict], keys: Iterable[str]):
    if residual_dict is None:
        return None
    for key in keys:
        value = residual_dict.get(key, None)
        if value is not None:
            return value
    return None


def extract_residual_component(
    residual_dict: Optional[Dict],
    component: str,
    mode: str = "per_volume",
):
    if mode == "per_volume":
        return _extract_raw_value(residual_dict, _PER_VOLUME_KEYS[component])
    if mode == "flux":
        return _extract_raw_value(residual_dict, _FLUX_KEYS[component])
    raise ValueError(f"Unknown residual mode: {mode}")


def component_to_per_sample_list(value, batch_size: int):
    if value is None:
        return [0.0] * batch_size
    if isinstance(value, torch.Tensor):
        value_cpu = value.detach().cpu().reshape(-1)
        if value_cpu.numel() == 1:
            return [float(value_cpu.item())] * batch_size
        if value_cpu.numel() != batch_size:
            raise ValueError(
                f"Residual component batch size mismatch: expected {batch_size}, got {value_cpu.numel()}"
            )
        return [float(x) for x in value_cpu.tolist()]
    return [float(value)] * batch_size


def accumulate_residual_totals(
    totals: Dict[str, float],
    residual_dict: Optional[Dict],
    batch_size: int,
    mode: str = "per_volume",
    residual_score=None,
):
    for component in PDE_COMPONENTS:
        value = extract_residual_component(residual_dict, component, mode=mode)
        if value is None:
            continue
        if isinstance(value, torch.Tensor):
            totals[component] += value.detach().sum().item()
        else:
            totals[component] += float(value) * batch_size

    if residual_score is None:
        return
    if isinstance(residual_score, torch.Tensor):
        totals["residual_score"] += residual_score.detach().sum().item()
    else:
        totals["residual_score"] += float(residual_score) * batch_size


def finalize_residual_totals(totals: Dict[str, float], n_samples: int) -> Dict[str, float]:
    if n_samples <= 0:
        raise ValueError("n_samples must be positive when finalizing residual totals")
    return {key: value / n_samples for key, value in totals.items()}


def accumulate_l2_totals(
    totals: Dict[str, float],
    residual_dict: Optional[Dict],
    batch_size: int,
):
    if residual_dict is None:
        return
    for component in L2_COMPONENTS:
        value = residual_dict.get(component, None)
        if value is None:
            continue
        if isinstance(value, torch.Tensor):
            totals[component] += value.detach().sum().item()
        else:
            totals[component] += float(value) * batch_size


def finalize_l2_totals(totals: Dict[str, float], n_samples: int) -> Dict[str, float]:
    if n_samples <= 0:
        raise ValueError("n_samples must be positive when finalizing l2 totals")
    return {key: value / n_samples for key, value in totals.items()}


def safe_ratio(numerator: float, denominator: float) -> Optional[float]:
    if abs(denominator) <= 1e-12:
        return None
    return numerator / denominator


def build_pde_summary(
    pred: Dict[str, float],
    true: Dict[str, float],
    components: Sequence[str],
    mode: str,
    label: str,
) -> Dict[str, object]:
    avg_pred = sum(float(pred.get(key, 0.0)) for key in components) / float(len(components))
    avg_true = sum(float(true.get(key, 0.0)) for key in components) / float(len(components))
    return {
        "label": label,
        "mode": mode,
        "components": list(components),
        "average_residual_pred": avg_pred,
        "average_residual_true": avg_true,
        "residual_ratio": safe_ratio(avg_pred, avg_true),
    }


def build_summary_bundle(
    pred_per_volume: Dict[str, float],
    true_per_volume: Dict[str, float],
    pred_flux: Optional[Dict[str, float]] = None,
    true_flux: Optional[Dict[str, float]] = None,
) -> Dict[str, Dict[str, Dict[str, object]]]:
    summaries = {
        "per_volume": {
            "momentum3": build_pde_summary(
                pred_per_volume,
                true_per_volume,
                SUMMARY_COMPONENTS["momentum3"],
                mode="per_volume",
                label="Legacy 3-channel average",
            ),
            "energy4": build_pde_summary(
                pred_per_volume,
                true_per_volume,
                SUMMARY_COMPONENTS["energy4"],
                mode="per_volume",
                label="Legacy 4-channel average",
            ),
            "five_channel": build_pde_summary(
                pred_per_volume,
                true_per_volume,
                SUMMARY_COMPONENTS["five_channel"],
                mode="per_volume",
                label="5-channel average",
            ),
        }
    }
    if pred_flux is not None and true_flux is not None:
        summaries["flux"] = {
            "momentum3": build_pde_summary(
                pred_flux,
                true_flux,
                SUMMARY_COMPONENTS["momentum3"],
                mode="flux",
                label="Legacy 3-channel flux average",
            ),
            "energy4": build_pde_summary(
                pred_flux,
                true_flux,
                SUMMARY_COMPONENTS["energy4"],
                mode="flux",
                label="Legacy 4-channel flux average",
            ),
            "five_channel": build_pde_summary(
                pred_flux,
                true_flux,
                SUMMARY_COMPONENTS["five_channel"],
                mode="flux",
                label="5-channel flux average",
            ),
        }
    return summaries


def canonical_residual_dict(
    Rc: float,
    Rmx: float,
    Rmy: float,
    RE: float,
    RSA: float,
    residual_score: float,
) -> Dict[str, float]:
    return {
        "Rc": float(Rc),
        "Rmx": float(Rmx),
        "Rmy": float(Rmy),
        "RE": float(RE),
        "RSA": float(RSA),
        "residual_score": float(residual_score),
    }
