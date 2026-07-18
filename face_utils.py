from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, UnidentifiedImageError

try:
    import insightface
    from insightface.app import FaceAnalysis
except ModuleNotFoundError:
    insightface = None
    FaceAnalysis = None


REQUIRED_COLUMNS = {"Name", "Photo"}
MODEL_NAME = "buffalo_l"
MODEL_ROOT = Path(__file__).parent / ".insightface"
DETECTION_THRESHOLD = 0.50
DETECTION_SIZE = (640, 640)


class FaceLoadingError(Exception):
    """Raised when known student face data cannot be prepared."""


@dataclass(frozen=True)
class KnownFaces:
    student_names: list[str]
    encodings: list[np.ndarray]


@dataclass(frozen=True)
class RecognitionResult:
    status: str
    name: str | None = None
    distance: float | None = None


def _require_insightface() -> None:
    if insightface is None or FaceAnalysis is None:
        raise FaceLoadingError(
            "The 'insightface' package is not installed in this environment. "
            "Make sure requirements.txt includes insightface and onnxruntime."
        )


def _validate_student_sheet(student_df: pd.DataFrame) -> None:
    missing_columns = REQUIRED_COLUMNS.difference(student_df.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise FaceLoadingError(f"Student data file is missing required columns: {missing}")


@lru_cache(maxsize=1)
def _get_face_analyzer() -> FaceAnalysis:
    _require_insightface()

    try:
        analyzer = FaceAnalysis(
            name=MODEL_NAME,
            root=str(MODEL_ROOT),
            providers=["CPUExecutionProvider"],
        )
        analyzer.prepare(ctx_id=-1, det_thresh=DETECTION_THRESHOLD, det_size=DETECTION_SIZE)
        return analyzer
    except Exception as exc:  # pragma: no cover - depends on runtime/model download
        raise FaceLoadingError(
            "InsightFace could not initialize. On first run it downloads the face model pack, "
            "so the runtime needs network access and enough disk space for model files."
        ) from exc


def _load_photo(photo_path: Path) -> np.ndarray:
    try:
        with Image.open(photo_path) as image:
            rgb_image = image.convert("RGB")
            return np.array(rgb_image)
    except FileNotFoundError as exc:
        raise FaceLoadingError(f"Missing photo: {photo_path.name}") from exc
    except UnidentifiedImageError as exc:
        raise FaceLoadingError(f"Invalid or corrupted photo file: {photo_path.name}") from exc


def scan_known_faces(student_df: pd.DataFrame, photos_dir: Path) -> KnownFaces:
    analyzer = _get_face_analyzer()
    _validate_student_sheet(student_df)

    if not photos_dir.exists():
        raise FaceLoadingError(f"Photos folder not found: {photos_dir}")

    student_names: list[str] = []
    encodings: list[np.ndarray] = []

    for row in student_df.itertuples(index=False):
        name = str(row.Name).strip()
        photo_name = str(row.Photo).strip()

        if not name or not photo_name or photo_name.lower() == "nan":
            raise FaceLoadingError(f"Invalid photo filename for student: {name or '<blank name>'}")

        photo_path = photos_dir / photo_name
        image_array = _load_photo(photo_path)
        faces = analyzer.get(image_array)

        if not faces:
            raise FaceLoadingError(f"No detectable face found in photo: {photo_name}")
        if len(faces) > 1:
            raise FaceLoadingError(f"Multiple faces found in photo: {photo_name}")

        student_names.append(name)
        encodings.append(np.asarray(faces[0].normed_embedding, dtype=np.float32))

    if not encodings:
        raise FaceLoadingError("No student face encodings could be loaded.")

    return KnownFaces(student_names=student_names, encodings=encodings)


def recognize_student(image_bytes: bytes, known_faces: KnownFaces, threshold: float) -> RecognitionResult:
    analyzer = _get_face_analyzer()

    try:
        with Image.open(BytesIO(image_bytes)) as image:
            rgb_image = np.array(image.convert("RGB"))
    except UnidentifiedImageError:
        return RecognitionResult(status="unknown")

    faces = analyzer.get(rgb_image)
    if not faces:
        return RecognitionResult(status="no_face")
    if len(faces) > 1:
        return RecognitionResult(status="multiple_faces")

    candidate_embedding = np.asarray(faces[0].normed_embedding, dtype=np.float32)
    known_matrix = np.asarray(known_faces.encodings, dtype=np.float32)
    similarities = known_matrix @ candidate_embedding
    distances = 1.0 - similarities
    best_match_index = int(np.argmin(distances))
    best_distance = float(distances[best_match_index])

    if best_distance <= threshold:
        return RecognitionResult(
            status="recognized",
            name=known_faces.student_names[best_match_index],
            distance=best_distance,
        )

    return RecognitionResult(status="unknown", distance=best_distance)
