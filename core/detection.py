"""Detection domain objects shared by inference and decision components."""
from dataclasses import dataclass, field


@dataclass
class Detection:
    class_id: int
    class_name: str
    confidence: float
    bbox_xyxy: list[float]
    center: tuple[int, int] = field(init=False)

    def __post_init__(self):
        self.refresh_geometry()

    def refresh_geometry(self):
        x1, y1, x2, y2 = self.bbox_xyxy
        self.center = (int((x1 + x2) / 2), int((y1 + y2) / 2))
