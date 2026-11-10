
import streamlit as st
import cv2
import numpy as np
import pandas as pd
import tempfile
import os
import matplotlib.pyplot as plt
from ultralytics import YOLO

st.set_page_config(
    page_title="Yerevan Smart Traffic AI",
    page_icon="🚦",
    layout="wide"
)
st.markdown("""
<style>

/* Main background */
.stApp {
    background: linear-gradient(135deg, #080d18 0%, #101827 50%, #080d18 100%);
    color: #f1f5f9;
}

/* Main content */
.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
    max-width: 1450px;
}

/* Headings */
h1 {
    color: #38bdf8 !important;
    font-weight: 800 !important;
    letter-spacing: 1px;
}

h2, h3 {
    color: #e2e8f0 !important;
}

/* Text */
p, label, span {
    color: #cbd5e1;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    gap: 12px;
    border-bottom: 1px solid #263449;
}

.stTabs [data-baseweb="tab"] {
    background: #172235;
    border-radius: 10px 10px 0 0;
    padding: 12px 22px;
}

/* Upload box */
[data-testid="stFileUploader"] {
    background: #172235;
    border: 1px solid #334155;
    border-radius: 14px;
    padding: 18px;
}

/* Metrics */
[data-testid="stMetric"] {
    background: linear-gradient(145deg, #172235, #1e293b);
    border: 1px solid #334155;
    border-radius: 14px;
    padding: 20px;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(90deg, #0284c7, #2563eb);
    color: white;
    border: none;
    border-radius: 10px;
    padding: 10px 24px;
    font-weight: bold;
    transition: 0.3s;
}

.stButton > button:hover {
    background: linear-gradient(90deg, #0ea5e9, #3b82f6);
    transform: translateY(-2px);
}

/* Dataframes */
[data-testid="stDataFrame"] {
    border: 1px solid #334155;
    border-radius: 12px;
}

/* Divider */
hr {
    border-color: #334155;
}

</style>
""", unsafe_allow_html=True)

with st.container(border=True):

    st.title("🚦 YEREVAN SMART TRAFFIC AI")

    st.subheader(
        "AI-Powered Intelligent Traffic Monitoring System"
    )

    st.success("● SYSTEM READY")

@st.cache_resource
def load_model():
    return YOLO("yolo11n.pt")

model = load_model()

VEHICLE_CLASSES = {
    2: "Car",
    3: "Motorcycle",
    5: "Bus",
    7: "Truck"
}

def analyze_image(image):
    results = model.predict(image, conf=0.25, verbose=False)
    result = results[0]

    counts = {name: 0 for name in VEHICLE_CLASSES.values()}
    zones = {"Left": 0, "Center": 0, "Right": 0}

    height, width = image.shape[:2]

    for box in result.boxes:
        class_id = int(box.cls[0])

        if class_id not in VEHICLE_CLASSES:
            continue

        name = VEHICLE_CLASSES[class_id]
        counts[name] += 1

        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        center_x = (x1 + x2) / 2

        if center_x < width / 3:
            zones["Left"] += 1
        elif center_x < 2 * width / 3:
            zones["Center"] += 1
        else:
            zones["Right"] += 1

    annotated = result.plot()

    return annotated, counts, zones


def traffic_level(total):
    if total <= 5:
        return "LOW", 20
    elif total <= 15:
        return "MEDIUM", 35
    else:
        return "HIGH", 50
def green_times(zones):
    total = sum(zones.values())

    if total == 0:
        return {
            "Left": 20,
            "Center": 20,
            "Right": 20
        }

    remaining = 30

    exact = {
        key: remaining * value / total
        for key, value in zones.items()
    }

    times = {
        key: 10 + int(value)
        for key, value in exact.items()
    }

    difference = 60 - sum(times.values())

    fractions = sorted(
        zones.keys(),
        key=lambda key: exact[key] - int(exact[key]),
        reverse=True
    )

    for i in range(difference):
        times[fractions[i]] += 1

    return times

```python
def show_results(counts, zones, is_average=False):
    total = sum(counts.values())
    level, suggested_time = traffic_level(total)

    st.markdown("---")
    st.header("📊 Traffic Analysis")

    col1, col2, col3, col4 = st.columns(4)

    if is_average:
        col1.metric("Average Vehicles / Frame", f"{total:.2f}")
    else:
        col1.metric("Total Vehicles", int(total))

    if level == "LOW":
        traffic_color = "🟢"
    elif level == "MEDIUM":
        traffic_color = "🟡"
    else:
        traffic_color = "🔴"

    col2.metric("Traffic Level", f"{traffic_color} {level}")
    col3.metric("Suggested Green Time", f"{suggested_time} sec")
    col4.metric("AI Detection", "ACTIVE", delta="Analysis completed")

    st.subheader("🚗 Vehicle Statistics")

    df = pd.DataFrame(
        list(counts.items()),
        columns=["Vehicle Type", "Count"]
    )
    st.dataframe(df, use_container_width=True)

    st.subheader("🛣️ Traffic Zones")

    zone_df = pd.DataFrame(
        list(zones.items()),
        columns=["Zone", "Vehicles"]
    )
    st.dataframe(zone_df, use_container_width=True)

    times = green_times(zones)

    st.subheader("🚦 Smart Traffic Light Recommendations")

    for direction, count in zones.items():
        st.write(
            f"**{direction}:** {count:.2f} vehicles — "
            f"Recommended green light: {times[direction]} seconds"
        )

    st.success(
        f"Total recommended green-light time: {sum(times.values())} seconds"
    )

    fig, ax = plt.subplots()
    ax.bar(zones.keys(), zones.values())
    ax.set_title("Vehicles by Image Zone")
    ax.set_ylabel("Average Vehicles" if is_average else "Number of Vehicles")

    st.pyplot(fig)
    plt.close(fig)
```


tab1, tab2 = st.tabs([
    "🖼️ Image Analysis",
    "🎥 Video Analysis"
])


with tab1:

    st.header("Upload Traffic Image")

    uploaded_image = st.file_uploader(
        "Choose an image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_image is not None:

        file_bytes = np.asarray(
            bytearray(uploaded_image.read()),
            dtype=np.uint8
        )

        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        if image is None:
            st.error("Could not read the image.")
        else:
            with st.spinner("YOLO is analyzing the image..."):
                annotated, counts, zones = analyze_image(image)

            col1, col2 = st.columns(2)

            with col1:
                st.image(
                    cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
                    caption="Original Image",
                    use_container_width=True
                )

            with col2:
                st.image(
                    cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
                    caption="YOLO Detection",
                    use_container_width=True
                )

            show_results(counts, zones)


with tab2:

    st.header("Upload Traffic Video")

    uploaded_video = st.file_uploader(
    "Choose a video",
    type=["mp4", "avi", "mov"],
    key="traffic_video"
)
    

    frame_step = st.slider(
        "Analyze every N frames",
        min_value=5,
        max_value=60,
        value=15
    )

    if uploaded_video is not None:

        if st.button("Start Video Analysis"):

            suffix = os.path.splitext(uploaded_video.name)[1]

            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=suffix
            ) as temp:
                temp.write(uploaded_video.read())
                video_path = temp.name

            cap = cv2.VideoCapture(video_path)

            tracking_model = load_model()

            track_info = {}

            frame_number = 0
            analyzed_frames = 0
            preview = None

            progress = st.progress(0)

            total_frames = int(
                cap.get(cv2.CAP_PROP_FRAME_COUNT)
            )

            while cap.isOpened():

                ret, frame = cap.read()

                if not ret:
                    break

                results = tracking_model.track(
                    frame,
                    persist=True,
                    tracker="custom_bytetrack.yaml",
                    classes=list(VEHICLE_CLASSES.keys()),
                    conf=0.20,
                    iou=0.5,
                    verbose=False
                )

                result = results[0]

                if result.boxes is not None and result.boxes.id is not None:

                    boxes = result.boxes.xyxy.cpu().numpy()
                    ids = result.boxes.id.int().cpu().tolist()
                    classes = result.boxes.cls.int().cpu().tolist()

                    height, width = frame.shape[:2]

                    for box, track_id, class_id in zip(
                        boxes, ids, classes
                    ):

                        x1, y1, x2, y2 = box
                        center_x = (x1 + x2) / 2

                        if center_x < width / 3:
                            zone = "Left"
                        elif center_x < 2 * width / 3:
                            zone = "Center"
                        else:
                            zone = "Right"

                        vehicle_name = VEHICLE_CLASSES[class_id]

                        track_info[track_id] = {
                            "class": vehicle_name,
                            "zone": zone
                        }

                if frame_number % frame_step == 0:
                    preview = result.plot()
                    analyzed_frames += 1

                frame_number += 1

                if total_frames > 0:
                    progress.progress(
                        min(frame_number / total_frames, 1.0)
                    )

            cap.release()
            os.remove(video_path)

            if track_info:

                unique_counts = {
                    name: 0 for name in VEHICLE_CLASSES.values()
                }

                unique_zones = {
                    "Left": 0,
                    "Center": 0,
                    "Right": 0
                }

                for vehicle in track_info.values():
                    unique_counts[vehicle["class"]] += 1
                    unique_zones[vehicle["zone"]] += 1

                st.success("Video tracking completed!")

                if preview is not None:
                    st.image(
                        cv2.cvtColor(preview, cv2.COLOR_BGR2RGB),
                        caption="ByteTrack Vehicle Tracking",
                        use_container_width=True
                    )

                col1, col2 = st.columns(2)

                col1.metric("Unique Vehicles", len(track_info))
                col2.metric("Analyzed Frames", analyzed_frames)

                show_results(unique_counts, unique_zones)

            else:
                st.warning(
                    "No vehicles could be tracked in this video."
                )           


st.markdown("---")
st.caption(
    "Yerevan Smart Traffic AI | Educational Prototype | "
    "Traffic-light timings are recommendations only."
)
