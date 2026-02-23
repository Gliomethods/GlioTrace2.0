from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ValidatedInit:
    stackfiles: list[Path]
    metadata: Path
    detection_sensitivity: float
    channel_roles: dict
    fcols: list[str]
    hmm_param: dict
    control: str
    patient_id: list
    sets_by_patient: dict[str, list[int]]
    treatment: list[str]
