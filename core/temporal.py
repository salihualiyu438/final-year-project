"""Temporal multi-frame confirmation for YOLO detections."""
from __future__ import annotations
from typing import Iterable


def _iou(a: list[float], b: list[float]) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if inter <= 0:
        return 0.0
    aa = max(1.0, (a[2]-a[0]) * (a[3]-a[1]))
    ab = max(1.0, (b[2]-b[0]) * (b[3]-b[1]))
    return inter / (aa + ab - inter)


class TemporalMatcher:
    """Tracks same-class boxes and releases them only after enough matches.

    A track survives a configurable number of missed inference frames. A
    detection matches a track when its IoU exceeds ``match_iou``. Once the
    hit count reaches the class-specific requirement, the current detection
    is emitted as confirmed.
    """

    def __init__(
        self,
        enabled: bool = True,
        target_classes: Iterable[str] = (),
        match_iou: float = 0.30,
        required_matches: dict[str, int] | None = None,
        max_misses: int = 2,
        max_tracks_per_class: int = 20,
    ) -> None:
        self.enabled = bool(enabled)
        self.target_classes = set(target_classes)
        self.match_iou = float(match_iou)
        self.required_matches = dict(required_matches or {})
        self.max_misses = int(max_misses)
        self.max_tracks_per_class = int(max_tracks_per_class)
        self.tracks: dict[str, list[dict]] = self._empty_tracks()

    def _empty_tracks(self) -> dict[str, list[dict]]:
        return {name: [] for name in self.target_classes}

    def set_enabled(self, enabled: bool) -> None:
        self.enabled = bool(enabled)
        if not self.enabled:
            self.reset()

    def reset(self) -> None:
        self.tracks = self._empty_tracks()

    def update(self, detections: list) -> list:
        if not self.enabled:
            self.reset()
            return detections

        passthrough = [d for d in detections if d.class_name not in self.target_classes]
        by_class = {name: [] for name in self.target_classes}
        for det in detections:
            if det.class_name in by_class:
                by_class[det.class_name].append(det)

        confirmed = []
        for name in self.target_classes:
            tracks = self.tracks[name]
            used: set[int] = set()
            next_tracks: list[dict] = []
            required = max(1, int(self.required_matches.get(name, 3)))

            for track in tracks:
                best_i, best_iou = -1, 0.0
                for i, det in enumerate(by_class[name]):
                    if i in used:
                        continue
                    score = _iou(track["bbox"], det.bbox_xyxy)
                    if score > best_iou:
                        best_i, best_iou = i, score

                if best_i >= 0 and best_iou >= self.match_iou:
                    det = by_class[name][best_i]
                    used.add(best_i)
                    track["bbox"] = det.bbox_xyxy
                    track["hits"] += 1
                    track["misses"] = 0
                    track["last"] = det
                    if track["hits"] >= required:
                        confirmed.append(det)
                    next_tracks.append(track)
                else:
                    track["misses"] += 1
                    if track["misses"] <= self.max_misses:
                        next_tracks.append(track)

            for i, det in enumerate(by_class[name]):
                if i in used or len(next_tracks) >= self.max_tracks_per_class:
                    continue
                next_tracks.append({
                    "bbox": list(det.bbox_xyxy),
                    "hits": 1,
                    "misses": 0,
                    "last": det,
                })
                # A class configured for one required match (e.g. large_gun)
                # is confirmed immediately on its first detection.
                if required <= 1:
                    confirmed.append(det)

            self.tracks[name] = next_tracks

        # De-duplicate confirmed detections by object identity/bounding box.
        unique = []
        seen = set()
        for det in confirmed:
            key = (det.class_name, tuple(round(v, 1) for v in det.bbox_xyxy))
            if key not in seen:
                seen.add(key)
                unique.append(det)
        return passthrough + unique
