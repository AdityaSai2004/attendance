from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd
import streamlit as st

from attendance import (
    ensure_attendance_state,
    get_attendance_summary,
    get_today_attendance,
    get_today_column_name,
    mark_present,
)
from excel_utils import (
    AttendanceDataError,
    build_download_csv,
    load_student_sheet,
)
from face_utils import FaceLoadingError, recognize_student, scan_known_faces


BASE_DIR = Path(__file__).parent
DATA_PATH = BASE_DIR / "sample_students.csv"
PHOTOS_DIR = BASE_DIR / "photos"
DEFAULT_THRESHOLD = 0.45


@st.cache_data(show_spinner=False)
def get_student_dataframe(data_path: Path) -> pd.DataFrame:
    return load_student_sheet(data_path)


@st.cache_resource(show_spinner="Computing face encodings...")
def get_known_faces(data_path: Path, photos_dir: Path):
    student_df = get_student_dataframe(data_path)
    return scan_known_faces(student_df, photos_dir)


def render_status_panel(message: str | None, level: str) -> None:
    st.subheader("Status")
    if not message:
        st.info("Ready to record attendance.")
        return

    if level == "success":
        st.success(message)
    elif level == "warning":
        st.warning(message)
    elif level == "error":
        st.error(message)
    else:
        st.info(message)


def render_attendance_table(student_names: list[str]) -> None:
    st.subheader("Today's Attendance")
    today_records = get_today_attendance(student_names)
    if today_records.empty:
        st.info("No attendance recorded yet.")
        return
    st.dataframe(today_records, hide_index=True, use_container_width=True)


def render_download(student_df: pd.DataFrame, student_names: list[str]) -> None:
    csv_file = build_download_csv(
        original_df=student_df,
        student_names=student_names,
        attendance_state=st.session_state.attendance,
        date_column=get_today_column_name(date.today()),
    )
    st.download_button(
        label="Download Updated CSV",
        data=csv_file,
        file_name=f"attendance_{date.today().isoformat()}.csv",
        mime="text/csv",
        use_container_width=True,
    )


def process_recognition(image_bytes: bytes, known_faces, threshold: float) -> None:
    with st.spinner("Recognizing face..."):
        recognition = recognize_student(
            image_bytes=image_bytes,
            known_faces=known_faces,
            threshold=threshold,
        )

    if recognition.status == "recognized" and recognition.name is not None:
        if mark_present(recognition.name):
            st.session_state.status_message = f"Good Morning, {recognition.name}!"
            st.session_state.status_level = "success"
        else:
            st.session_state.status_message = "Attendance already recorded."
            st.session_state.status_level = "warning"
    elif recognition.status == "no_face":
        st.session_state.status_message = "No face detected. Please try again."
        st.session_state.status_level = "error"
    elif recognition.status == "multiple_faces":
        st.session_state.status_message = (
            "Multiple faces detected. Please ensure only one person is in front of the camera."
        )
        st.session_state.status_level = "error"
    else:
        st.session_state.status_message = "Student not recognized."
        st.session_state.status_level = "warning"


def main() -> None:
    st.set_page_config(page_title="Face Attendance", page_icon="📸", layout="centered")
    st.title("Face Recognition Attendance System")
    st.caption(f"Date: {date.today():%A, %d %B %Y}")

    st.session_state.setdefault("status_message", None)
    st.session_state.setdefault("status_level", "info")
    st.session_state.setdefault("capture_requested", False)

    try:
        student_df = get_student_dataframe(DATA_PATH)
        known_faces = get_known_faces(DATA_PATH, PHOTOS_DIR)
    except (AttendanceDataError, FaceLoadingError) as exc:
        st.error(str(exc))
        st.stop()
    except Exception as exc:  # pragma: no cover - defensive Streamlit fallback
        st.error(f"Unexpected startup error: {exc}")
        st.stop()

    student_names = known_faces.student_names
    ensure_attendance_state(student_names)

    total_students, present_students, progress = get_attendance_summary(student_names)
    stats_columns = st.columns(3)
    stats_columns[0].metric("Students", total_students)
    stats_columns[1].metric("Present", present_students)
    stats_columns[2].metric("Progress", f"{progress:.0%}")
    st.progress(progress if progress > 0 else 0.0)

    threshold = st.slider(
        "Recognition threshold",
        min_value=0.20,
        max_value=0.80,
        value=DEFAULT_THRESHOLD,
        step=0.01,
        help="Lower values are stricter. With InsightFace this uses embedding distance derived from normalized similarity.",
    )

    if st.button("Record Attendance", use_container_width=True):
        st.session_state.capture_requested = True
        st.session_state.status_message = "Allow camera access and capture a photo to continue."
        st.session_state.status_level = "info"

    st.subheader("Test with Uploaded Photo")
    uploaded_image = st.file_uploader(
        "Upload a student photo for testing",
        type=["jpg", "jpeg", "png"],
        help="This uses the same recognition flow as the camera input.",
    )
    if uploaded_image is not None:
        st.image(uploaded_image, caption="Uploaded test photo", use_container_width=True)
        if st.button("Check Uploaded Photo", use_container_width=True):
            process_recognition(uploaded_image.getvalue(), known_faces, threshold)

    if st.session_state.capture_requested:
        camera_image = st.camera_input("Capture student photo")
        if camera_image is not None:
            process_recognition(camera_image.getvalue(), known_faces, threshold)
            st.session_state.capture_requested = False

    render_status_panel(st.session_state.status_message, st.session_state.status_level)
    render_attendance_table(student_names)
    render_download(student_df, student_names)


if __name__ == "__main__":
    main()
