"""Request/response contracts. Every request is validated here before it reaches the model."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

# Hard limits reject physically implausible input. Values that are plausible but outside the
# training range are accepted and flagged with a warning in the response.
LIMITS = {
    "air_temp_k": (250.0, 350.0),
    "process_temp_k": (250.0, 400.0),
    "rotational_speed_rpm": (1.0, 10000.0),
    "torque_nm": (0.0, 500.0),
    "tool_wear_min": (0.0, 1000.0),
}


class MachineInput(BaseModel):
    """One machine operating record (the six raw inputs of the AI4I dataset)."""

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "example": {
                "type": "L",
                "air_temp_k": 300.1,
                "process_temp_k": 310.2,
                "rotational_speed_rpm": 1350,
                "torque_nm": 58.5,
                "tool_wear_min": 210,
            }
        },
    )

    type: Literal["L", "M", "H"] = Field(description="Product quality variant: L(ow), M(edium), H(igh)")
    air_temp_k: float = Field(ge=LIMITS["air_temp_k"][0], le=LIMITS["air_temp_k"][1], description="Kelvin")
    process_temp_k: float = Field(
        ge=LIMITS["process_temp_k"][0], le=LIMITS["process_temp_k"][1], description="Kelvin"
    )
    rotational_speed_rpm: float = Field(
        ge=LIMITS["rotational_speed_rpm"][0], le=LIMITS["rotational_speed_rpm"][1], description="rpm"
    )
    torque_nm: float = Field(ge=LIMITS["torque_nm"][0], le=LIMITS["torque_nm"][1], description="Nm")
    tool_wear_min: float = Field(ge=LIMITS["tool_wear_min"][0], le=LIMITS["tool_wear_min"][1], description="minutes")


class MachineChanges(BaseModel):
    """Partial update for the what-if simulator; omitted fields keep their base value."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["L", "M", "H"] | None = None
    air_temp_k: float | None = Field(default=None, ge=LIMITS["air_temp_k"][0], le=LIMITS["air_temp_k"][1])
    process_temp_k: float | None = Field(
        default=None, ge=LIMITS["process_temp_k"][0], le=LIMITS["process_temp_k"][1]
    )
    rotational_speed_rpm: float | None = Field(
        default=None, ge=LIMITS["rotational_speed_rpm"][0], le=LIMITS["rotational_speed_rpm"][1]
    )
    torque_nm: float | None = Field(default=None, ge=LIMITS["torque_nm"][0], le=LIMITS["torque_nm"][1])
    tool_wear_min: float | None = Field(
        default=None, ge=LIMITS["tool_wear_min"][0], le=LIMITS["tool_wear_min"][1]
    )


class ThresholdOverride(BaseModel):
    model_config = ConfigDict(extra="forbid")

    medium: float = Field(gt=0, lt=1)
    high: float = Field(gt=0, lt=1)

    @model_validator(mode="after")
    def ordered(self) -> ThresholdOverride:
        if self.medium > self.high:
            raise ValueError("medium threshold must be <= high threshold")
        return self


class PredictRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    machine: MachineInput
    thresholds: ThresholdOverride | None = None


class BatchPredictRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    machines: list[MachineInput] = Field(min_length=1)
    thresholds: ThresholdOverride | None = None
    explain: bool = False


class SimulateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    base: MachineInput
    changes: MachineChanges
    thresholds: ThresholdOverride | None = None

    @model_validator(mode="after")
    def has_change(self) -> SimulateRequest:
        if not any(v is not None for v in self.changes.model_dump().values()):
            raise ValueError("changes must modify at least one parameter")
        return self
