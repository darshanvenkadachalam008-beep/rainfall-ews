"""Centralized, typed configuration loader for SIH26071 REWS.

Reads YAML configuration files from configs/ and merges with environment
variables using pydantic-settings. All paths, bounds, thresholds, and
horizons must be loaded through this module.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional
import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


PROJECT_ROOT = Path(__file__).resolve().parents[2]


class DomainBBox(BaseModel):
    west: float
    south: float
    east: float
    north: float

    @property
    def bounds_tuple(self) -> tuple[float, float, float, float]:
        """Returns (west, south, east, north) bounds tuple."""
        return (self.west, self.south, self.east, self.north)


class RegionInfo(BaseModel):
    id: str
    name: str
    state: str
    country: str
    timezone: str
    geographic_crs: str
    projected_crs: str


class EventInfo(BaseModel):
    name: str
    window_start_utc: str
    window_end_utc: str
    focus_start_utc: str
    focus_end_utc: str


class RainfallGridConfig(BaseModel):
    resolution_deg: float
    downscaled_resolution_deg: float = 0.02
    timestep_minutes: int


class InundationGridConfig(BaseModel):
    resolution_m: int
    coarse_resolution_m: int = 90


class ForecastHorizonConfig(BaseModel):
    nowcast_horizon_minutes: int
    short_range_horizon_hours: int
    issue_interval_hours: int


class RegionConfig(BaseModel):
    region: RegionInfo
    event: EventInfo
    domains: Dict[str, DomainBBox]
    grids: Dict[str, Any]
    forecast: ForecastHorizonConfig
    validation: Dict[str, List[str]]


class IMDRainfallClass(BaseModel):
    id: str
    label: str
    min_mm: float
    max_mm: Optional[float] = None


class AlertTrigger(BaseModel):
    class_min: str
    probability_min: float


class AlertLevel(BaseModel):
    id: str
    label: str
    action: str
    triggers: List[AlertTrigger]


class VerificationWindow(BaseModel):
    id: str
    hours: int
    thresholds_mm: List[float]


class LossThresholds(BaseModel):
    thresholds: List[float]
    weights: List[float]


class ThresholdsConfig(BaseModel):
    imd_daily_classes: Dict[str, Any]
    event_definition: Dict[str, str]
    verification: Dict[str, Any]
    alert_levels: List[AlertLevel]
    flood_depth_classes: Dict[str, Any]
    inundation_verification: Dict[str, Any]
    lead_time_bands_hours: Dict[str, List[int]]
    loss_thresholds: Dict[str, LossThresholds]


class PipelineBudget(BaseModel):
    training_budget_gb: float = 25.0
    michaung_window_budget_gb: float = 6.0
    corpus_event_budget_gb: float = 19.0


class PipelineConfig(BaseModel):
    budget: PipelineBudget
    sampling: Dict[str, Any]
    gfs: Dict[str, Any]
    splits: Dict[str, Any]
    paths: Dict[str, str]


class REWSSettings(BaseSettings):
    """Global REWS settings integrating environment variables with YAML configurations."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Credentials
    earthdata_username: Optional[str] = Field(default=None, alias="EARTHDATA_USERNAME")
    earthdata_password: Optional[str] = Field(default=None, alias="EARTHDATA_PASSWORD")
    cdsapi_url: Optional[str] = Field(default=None, alias="CDSAPI_URL")
    cdsapi_key: Optional[str] = Field(default=None, alias="CDSAPI_KEY")
    mosdac_username: Optional[str] = Field(default=None, alias="MOSDAC_USERNAME")
    mosdac_password: Optional[str] = Field(default=None, alias="MOSDAC_PASSWORD")
    telegram_bot_token: Optional[str] = Field(default=None, alias="TELEGRAM_BOT_TOKEN")
    telegram_chat_id: Optional[str] = Field(default=None, alias="TELEGRAM_CHAT_ID")

    # Directory roots
    project_root: Path = PROJECT_ROOT
    raw_dir: Path = PROJECT_ROOT / "data" / "raw"
    processed_dir: Path = PROJECT_ROOT / "data" / "processed"
    models_dir: Path = PROJECT_ROOT / "models"
    outputs_dir: Path = PROJECT_ROOT / "outputs"
    configs_dir: Path = PROJECT_ROOT / "configs"

    # Loaded YAML configs
    pilot: RegionConfig = Field(default_factory=lambda: load_pilot_config())
    thresholds: ThresholdsConfig = Field(default_factory=lambda: load_thresholds_config())
    pipeline: PipelineConfig = Field(default_factory=lambda: load_pipeline_config())


def load_yaml(path: Path) -> dict[str, Any]:
    """Reads and parses a YAML file safely."""
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_pilot_config(config_path: Optional[Path] = None) -> RegionConfig:
    path = config_path or (PROJECT_ROOT / "configs" / "regions" / "pilot.yaml")
    data = load_yaml(path)
    return RegionConfig(**data)


def load_thresholds_config(config_path: Optional[Path] = None) -> ThresholdsConfig:
    path = config_path or (PROJECT_ROOT / "configs" / "thresholds.yaml")
    data = load_yaml(path)
    return ThresholdsConfig(**data)


def load_pipeline_config(config_path: Optional[Path] = None) -> PipelineConfig:
    path = config_path or (PROJECT_ROOT / "configs" / "pipeline.yaml")
    data = load_yaml(path)
    return PipelineConfig(**data)


_settings_instance: Optional[REWSSettings] = None


def get_settings(reload: bool = False) -> REWSSettings:
    """Returns singleton settings instance."""
    global _settings_instance
    if _settings_instance is None or reload:
        _settings_instance = REWSSettings()
    return _settings_instance
