import streamlit as st
import tempfile
import os
import shutil
import pandas as pd
import cv2

st.set_page_config(page_title="Automatic Number Plate Recognition", layout="wide")
st.title("Automatic Number Plate Recognition Workflow")

uploaded_video = st.file_uploader("Upload a video of cars for license plate detection", type=["mp4", "avi", "mov"])

if uploaded_video:
    with tempfile.TemporaryDirectory() as tmpdir:
        video_path = os.path.join(tmpdir, uploaded_video.name)
        with open(video_path, "wb") as f:
            f.write(uploaded_video.read())



        import subprocess
        PYTHON_EXEC = os.path.join(os.getcwd(), ".venv", "Scripts", "python.exe")

        def run_step(cmd, step_name, progress, step_idx, total_steps):
            with st.spinner(f"{step_name} in progress. Please wait..."):
                progress.progress(step_idx / total_steps, text=f"{step_name}...")
                result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
                progress.progress((step_idx + 1) / total_steps, text=f"{step_name} done.")
            if result.returncode != 0:
                st.error(f"{step_name} failed!\nStdout:\n{result.stdout}\nStderr:\n{result.stderr}")
                st.stop()
            else:
                st.success(f"{step_name} completed successfully.")
                if result.stdout:
                    st.text(f"Stdout:\n{result.stdout}")
                if result.stderr:
                    st.text(f"Stderr:\n{result.stderr}")

        total_steps = 4
        progress = st.progress(0, text="Starting pipeline...")

        # Step 1: Run video_plate_detection.py as a module
        run_step(f'"{PYTHON_EXEC}" -m app.video_plate_detection {video_path} {tmpdir}/test.csv', "Video Plate Detection", progress, 0, total_steps)

        # Step 2: Run interpolate_missing_data.py as a module
        run_step(f'"{PYTHON_EXEC}" -m app.interpolate_missing_data {tmpdir}/test.csv {tmpdir}/test_interpolated.csv', "Interpolate Missing Data", progress, 1, total_steps)

        # Step 3: Run filter_unique_vehicles.py as a module
        run_step(f'"{PYTHON_EXEC}" -m app.filter_unique_vehicles {tmpdir}/test_interpolated.csv {tmpdir}/unique_vehicles.csv', "Filter Unique Vehicles", progress, 2, total_steps)

        # Step 4: Run visualize_results.py as a module
        run_step(f'"{PYTHON_EXEC}" -m app.visualize_results {video_path} {tmpdir}/test_interpolated.csv {tmpdir}/out.mp4', "Visualize Results", progress, 3, total_steps)

        # Display output video
        st.subheader("Output Video with Detected License Plates")
        out_video_path = f"{tmpdir}/out.mp4"
        if os.path.exists(out_video_path):
            with open(out_video_path, "rb") as vid_file:
                st.video(vid_file.read())
        else:
            st.error("Output video was not generated. Please check for errors in the processing pipeline.")

        # Display filtered CSV
        st.subheader("Filtered Unique Vehicles")
        unique_csv_path = f"{tmpdir}/unique_vehicles.csv"
        if os.path.exists(unique_csv_path):
            df = pd.read_csv(unique_csv_path)
            st.dataframe(df)
        else:
            st.error("Filtered unique vehicles CSV was not generated.")
