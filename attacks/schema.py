from dataclasses import dataclass, field
from typing import List

@dataclass
class AttackSpec:
    attack_id: str
    name: str
    category: str
    payment_rail: str
    description: str
    signals: List[str] = field(default_factory=list)
    simulation_level: str = "research"
