from pydantic import BaseModel, Field

class BatteryConfig(BaseModel):
    capacity_mwh: float = Field(50.0, description="Max energy capacity in MWh")
    max_charge_mw: float = Field(20.0, description="Max charging power in MW")
    max_discharge_mw: float = Field(20.0, description="Max discharging power in MW")
    efficiency_charge: float = Field(0.95, description="Charging efficiency")
    efficiency_discharge: float = Field(0.95, description="Discharging efficiency")
    initial_soc: float = Field(0.5, description="Initial State of Charge (0.0 to 1.0)")
    cell_price: float = Field(200.0, description="Cost per MWh of capacity for degradation")
    cycles: float = Field(5000.0, description="Lifecycle cycles (2N)")

class ObservationModel(BaseModel):
    hour_of_day: float
    soc: float
    price_lmp: float
    p_avg: float
    freq_regd: float
    load_mw: float
from typing import List

class ActionModel(BaseModel):
    action: List[float] = Field(..., description="Continuous actions [a_PS, a_EA, a_FR] from -1.0 to 1.0")

class StepResult(BaseModel):
    observation: ObservationModel
    reward: float
    terminated: bool
    truncated: bool
    info: dict

class ResetConfig(BaseModel):
    seed: int | None = None
    task: str = Field("hard", description="'easy' (EA), 'medium' (EA+FR), 'hard' (EA+FR+PS)")
