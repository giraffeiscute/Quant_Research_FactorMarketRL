"""CUDA device parsing helpers shared by user-facing CLI entrypoints."""

from __future__ import annotations


def normalize_cuda_gpu_ids(gpu_ids: list[int]) -> list[int]:
    if len(gpu_ids) == 0:
        raise ValueError("--devices must include at least one GPU id.")
    if any(gpu_id < 0 for gpu_id in gpu_ids):
        raise ValueError("--devices GPU ids must be non-negative integers.")
    if len(set(gpu_ids)) != len(gpu_ids):
        raise ValueError("--devices cannot contain duplicate GPU ids.")
    return list(gpu_ids)


def parse_cuda_gpu_ids(raw_devices: str) -> list[int]:
    normalized = str(raw_devices).strip()
    if not normalized:
        raise ValueError("--devices cannot be empty.")

    parsed_gpu_ids: list[int] = []
    for token in normalized.split(","):
        stripped_token = token.strip()
        if not stripped_token:
            raise ValueError("--devices contains an empty GPU id entry.")
        try:
            parsed_gpu_id = int(stripped_token)
        except ValueError as exc:
            raise ValueError("--devices must be GPU id integers separated by commas.") from exc
        parsed_gpu_ids.append(parsed_gpu_id)

    return normalize_cuda_gpu_ids(parsed_gpu_ids)


def resolve_holdout_cuda_gpu_ids(
    devices: str | int | list[int] | tuple[int, ...] | None,
    *,
    int_mode: str = "count",
) -> list[int]:
    if devices is None:
        return [0]
    if isinstance(devices, bool):
        raise ValueError("--devices must be GPU ids or a comma-separated GPU id list.")
    if isinstance(devices, str):
        return parse_cuda_gpu_ids(devices)
    if isinstance(devices, int):
        resolved_device = int(devices)
        if int_mode == "count":
            if resolved_device <= 0:
                raise ValueError("--devices must be positive when provided as a device count.")
            return list(range(resolved_device))
        return normalize_cuda_gpu_ids([resolved_device])
    if isinstance(devices, (list, tuple)):
        if not all(isinstance(device, int) and not isinstance(device, bool) for device in devices):
            raise ValueError("--devices list entries must be integers.")
        return normalize_cuda_gpu_ids([int(device) for device in devices])
    raise ValueError("--devices must be a GPU id string, integer, or list of GPU ids.")


def resolve_lightning_cuda_devices(
    devices: str | int | list[int] | tuple[int, ...] | None,
) -> tuple[int | list[int], list[int]]:
    if devices is None:
        return [0], [0]
    if isinstance(devices, bool):
        raise ValueError("--devices must be GPU ids or a comma-separated GPU id list.")
    if isinstance(devices, int):
        requested_count = int(devices)
        if requested_count <= 0:
            raise ValueError("--devices must be positive when provided as a count.")
        return requested_count, list(range(requested_count))
    if isinstance(devices, str):
        normalized_value = devices.strip()
        if not normalized_value:
            raise ValueError("--devices cannot be empty.")
        explicit_gpu_ids = parse_cuda_gpu_ids(normalized_value)
        return explicit_gpu_ids, list(explicit_gpu_ids)
    if isinstance(devices, (list, tuple)):
        if not all(isinstance(device, int) and not isinstance(device, bool) for device in devices):
            raise ValueError("--devices explicit GPU ids must be integers.")
        explicit_gpu_ids = [int(device) for device in devices]
        return explicit_gpu_ids, normalize_cuda_gpu_ids(explicit_gpu_ids)
    raise ValueError("--devices must be GPU ids or a comma-separated GPU id list.")
