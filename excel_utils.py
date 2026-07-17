from __future__ import annotations

from io import BytesIO
from pathlib import Path

import pandas as pd


REQUIRED_COLUMNS = {"Name", "Photo"}


class AttendanceDataError(Exception):
    """Raised when the student Excel sheet is unavailable or invalid."""


def load_student_sheet(data_path: Path) -> pd.DataFrame:
    if not data_path.exists():
        raise AttendanceDataError(f"Student data file not found: {data_path.name}")

    try:
        if data_path.suffix.lower() == ".csv":
            student_df = pd.read_csv(data_path)
        else:
            student_df = pd.read_excel(data_path)
    except Exception as exc:  # pragma: no cover - depends on engine/runtime
        raise AttendanceDataError(f"Unable to read student data file: {data_path.name}") from exc

    missing_columns = REQUIRED_COLUMNS.difference(student_df.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise AttendanceDataError(f"Student data file is missing required columns: {missing}")

    return student_df.copy()


def build_download_csv(
    original_df: pd.DataFrame,
    student_names: list[str],
    attendance_state: dict[str, bool],
    date_column: str,
) -> bytes:
    output_df = original_df.copy()
    if date_column not in output_df.columns:
        output_df[date_column] = ""

    attendance_map = {name: 1 if attendance_state.get(name, False) else "" for name in student_names}
    output_df[date_column] = output_df["Name"].map(attendance_map).fillna("")

    csv_text = output_df.to_csv(index=False)
    buffer = BytesIO(csv_text.encode("utf-8"))
    return buffer.getvalue()
