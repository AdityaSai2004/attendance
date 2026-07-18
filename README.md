# Face Recognition Attendance System

A simple Streamlit-based attendance app for school projects that recognizes student faces from a local `photos/` folder and keeps attendance in Streamlit session state.

## Features

- Face recognition using `InsightFace`
- One-page Streamlit UI
- In-memory attendance tracking
- CSV export generated fully in memory
- Configurable recognition threshold
- Progress indicator and present count
- Graceful handling for missing files and invalid inputs

## Folder Structure

```text
face-attendance/
├── app.py
├── attendance.py
├── excel_utils.py
├── face_utils.py
├── requirements.txt
├── README.md
├── photos/
│   └── .gitkeep
└── sample_students.csv
```

## Installation

1. Install Python 3.11 or newer.
2. Install dependencies:

```bash
uv pip install -r requirements.txt
```

On the first run, InsightFace downloads its model files automatically. Make sure the environment has internet access for that initial model download.

## Run Locally with uv

```bash
uv run streamlit run app.py
```

## Deploying to Streamlit Community Cloud

1. Push this project to GitHub.
2. Create a new app in Streamlit Community Cloud.
3. Set `app.py` as the entry point.
4. Ensure `requirements.txt`, `packages.txt`, `runtime.txt`, `sample_students.csv`, and the `photos/` folder are included in the repository.
5. Reboot the app after deployment if dependency installation changes.

### Streamlit Cloud note

InsightFace uses ONNX Runtime and downloads its model pack on first startup. This repo includes `runtime.txt` to pin Streamlit Community Cloud to Python 3.11. `packages.txt` can remain in place for cloud builds, but the app no longer depends on `face_recognition` or `dlib`.

## CSV Format

The CSV file must include these columns:

| Name | Photo |
| --- | --- |
| Aditya | aditya.jpg |
| Rahul | rahul.jpg |

- `Name`: Student name to display in attendance.
- `Photo`: Filename stored inside `photos/`.

On download, the app adds a new column for today's date and marks:

- `1` for present
- blank for absent

## Photo Format

- Store one image per student in the `photos/` folder.
- Use clear front-facing images.
- Each image should contain exactly one detectable face.
- The filename must match the value in the CSV `Photo` column.

## Screenshots

- Home screen screenshot: add here
- Camera capture screenshot: add here
- Attendance download screenshot: add here
