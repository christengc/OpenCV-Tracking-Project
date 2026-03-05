from dataclasses import dataclass


@dataclass
class BallState:
    x: float
    y: float
    speed: float
    dx: float
    dy: float
