from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

import face_recognition
import numpy as np
import pandas as pd
from PIL import Image, UnidentifiedImageError


REQUIRED_COLUMNS = {"Name", "Photo"}


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


def _validate_student_sheet(student_df: pd.DataFrame) -> None:
    missing_columns = REQUIRED_COLUMNS.difference(student_df.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise FaceLoadingError(f"Excel is missing required columns: {missing}")


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
        face_encodings = face_recognition.face_encodings(image_array)

        if not face_encodings:
            raise FaceLoadingError(f"No detectable face found in photo: {photo_name}")

        student_names.append(name)
        encodings.append(face_encodings[0])

    if not encodings:
        raise FaceLoadingError("No student face encodings could be loaded.")

    return KnownFaces(student_names=student_names, encodings=encodings)


def recognize_student(image_bytes: bytes, known_faces: KnownFaces, threshold: float) -> RecognitionResult:
    try:
        with Image.open(BytesIO(image_bytes)) as image:
            rgb_image = np.array(image.convert("RGB"))
    except UnidentifiedImageError:
        return RecognitionResult(status="unknown")

    face_locations = face_recognition.face_locations(rgb_image)
    if not face_locations:
        return RecognitionResult(status="no_face")
    if len(face_locations) > 1:
        return RecognitionResult(status="multiple_faces")

    face_encodings = face_recognition.face_encodings(rgb_image, known_face_locations=face_locations)
    if not face_encodings:
        return RecognitionResult(status="unknown")

    candidate_encoding = face_encodings[0]
    distances = face_recognition.face_distance(known_faces.encodings, candidate_encoding)
    best_match_index = int(np.argmin(distances))
    best_distance = float(distances[best_match_index])

    if best_distance <= threshold:
        return RecognitionResult(
            status="recognized",
            name=known_faces.student_names[best_match_index],
            distance=best_distance,
        )

    return RecognitionResult(status="unknown", distance=best_distance)
