from pydantic import BaseModel
from typing import List, Optional

class CountryBase(BaseModel):
    country_code: str
    lpi: float
    gdp_growth: float
    inflation: float
    conflict_intensity: float
    military_expenditure: float
    risk_cluster: int

class SimulationRequest(BaseModel):
    lpi_override: Optional[float] = None
    gdp_growth_override: Optional[float] = None
    inflation_override: Optional[float] = None
    conflict_intensity_override: Optional[float] = None
    military_expenditure_override: Optional[float] = None

class PredictionPoint(BaseModel):
    year: int
    business_viability: float
    supply_chain_stability: float
    
class CountryDetail(CountryBase):
    success_ratio: float
    historical_data: List[PredictionPoint]
    lstm_forecast: List[PredictionPoint]
