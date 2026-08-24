from dataclasses import dataclass
from typing import Dict, List


@dataclass
class ArenaRound:
    round_number: int
    attack_count: int
    detected_count: int
    detection_rate: float


class FraudArena:
    def __init__(self):
        self.rounds: List[ArenaRound] = []

    def record_round(self, round_number, attack_count, detected_count):
        detection_rate = (
            detected_count / attack_count
            if attack_count > 0
            else 0.0
        )

        result = ArenaRound(
            round_number=round_number,
            attack_count=attack_count,
            detected_count=detected_count,
            detection_rate=detection_rate,
        )

        self.rounds.append(result)
        return result

    def summary(self) -> Dict:
        if not self.rounds:
            return {
                "rounds": 0,
                "best_detection_rate": 0.0,
            }

        return {
            "rounds": len(self.rounds),
            "best_detection_rate": max(
                r.detection_rate for r in self.rounds
            ),
            "latest_detection_rate": self.rounds[-1].detection_rate,
        }
