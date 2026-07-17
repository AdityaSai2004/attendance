from __future__ import annotations

from datetime import date

import pandas as pd
import streamlit as st


def ensure_attendance_state(student_names: list[str]) -> None:
    if "attendance" not in st.session_state:
        st.session_state.attendance = {name: False for name in student_names}
        return

    for name in student_names:
        st.session_state.attendance.setdefault(name, False)


def mark_present(student_name: str) -> bool:
    if st.session_state.attendance.get(student_name):
        return False
    st.session_state.attendance[student_name] = True
    return True


def get_today_attendance(student_names: list[str]) -> pd.DataFrame:
    rows = [
        {"Name": name, "Status": "Present"}
        for name in student_names
        if st.session_state.attendance.get(name, False)
    ]
    return pd.DataFrame(rows)


def get_attendance_summary(student_names: list[str]) -> tuple[int, int, float]:
    total_students = len(student_names)
    present_students = sum(1 for name in student_names if st.session_state.attendance.get(name, False))
    progress = (present_students / total_students) if total_students else 0.0
    return total_students, present_students, progress


def get_today_column_name(today: date) -> str:
    return today.isoformat()
