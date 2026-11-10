import streamlit as st
import cv2
import numpy as np
from PIL import Image, ImageOps

# Optional HEIC support
try:
    import pillow_heif
    pillow_heif.register_heif_opener()
except ImportError:
    pass

WORK_WIDTH = 1600

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------

st.set_page_config(
    page_title="Sharpest Region Visualizer",
    page_icon="📷",
    layout="wide"
)

# --------------------------------------------------
# IMAGE LOADING
# --------------------------------------------------

def load_image(uploaded_file):

    image = Image.open(uploaded_file)

    image = ImageOps.exif_transpose(image)

    image = image.convert("RGB")

    scale = WORK_WIDTH / image.width

    image = image.resize(
        (
            WORK_WIDTH,
            max(1, round(image.height * scale))
        ),
        Image.LANCZOS
    )

    rgb = np.array(image)

    gray = cv2.cvtColor(
        rgb,
        cv2.COLOR_RGB2GRAY
    )

    return rgb, gray


# --------------------------------------------------
# FOCUS MAP
# --------------------------------------------------

def create_focus_map(gray):

    lap = cv2.Laplacian(
        gray,
        cv2.CV_64F
    )

    lap = np.abs(lap)

    # Blur to create larger focus regions
    focus_map = cv2.blur(
        lap,
        (51, 51)
    )

    focus_map = cv2.normalize(
        focus_map,
        None,
        0,
        255,
        cv2.NORM_MINMAX
    )

    focus_map = focus_map.astype(
        np.uint8
    )

    return focus_map


def create_heatmap(focus_map):

    heatmap = cv2.applyColorMap(
        focus_map,
        cv2.COLORMAP_JET
    )

    heatmap = cv2.cvtColor(
        heatmap,
        cv2.COLOR_BGR2RGB
    )

    return heatmap


def create_overlay(rgb, heatmap):

    overlay = cv2.addWeighted(
        rgb,
        0.7,
        heatmap,
        0.3,
        0
    )

    return overlay


# --------------------------------------------------
# SHARPEST REGION DETECTION
# --------------------------------------------------

def find_sharpest_region(rgb, gray):

    output = rgb.copy()

    h, w = gray.shape

    box_size = int(min(h, w) * 0.20)

    box_size = max(box_size, 150)

    step = max(box_size // 5, 25)

    best_score = -1
    best_box = None

    for y in range(
        0,
        h - box_size,
        step
    ):

        for x in range(
            0,
            w - box_size,
            step
        ):

            patch = gray[
                y:y + box_size,
                x:x + box_size
            ]

            gx = cv2.Sobel(
                patch,
                cv2.CV_64F,
                1,
                0,
                ksize=3
            )

            gy = cv2.Sobel(
                patch,
                cv2.CV_64F,
                0,
                1,
                ksize=3
            )

            score = np.mean(
                gx ** 2 + gy ** 2
            )

            if score > best_score:

                best_score = score

                best_box = (
                    x,
                    y,
                    box_size,
                    box_size
                )

    if best_box:

        x, y, bw, bh = best_box

        cv2.rectangle(
            output,
            (x, y),
            (x + bw, y + bh),
            (0, 255, 0),
            8
        )

        cv2.putText(
            output,
            "Sharpest Area",
            (x, max(50, y - 15)),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.1,
            (0, 255, 0),
            3
        )

    return output


# --------------------------------------------------
# USER INTERFACE
# --------------------------------------------------

st.title("📷 Sharpest Region Visualizer")

st.markdown("""
Upload one or more images and discover which area contains the highest concentration of visual detail.

### The system will:

✅ Analyse image focus

✅ Detect the sharpest area

✅ Draw a green box around the sharpest region

✅ Generate a focus heatmap overlay

### How to read the results

🟩 **Green Box** = Sharpest region detected by the algorithm

🟥 **Red Areas** = Strong visual detail

🟨 **Yellow Areas** = Moderate visual detail

🟦 **Blue Areas** = Weak visual detail or blur
""")

# --------------------------------------------------
# FILE UPLOAD
# --------------------------------------------------

uploaded_files = st.file_uploader(
    "📂 Select one or more images",
    type=[
        "jpg",
        "jpeg",
        "png",
        "bmp",
        "tif",
        "tiff",
        "webp",
        "heic",
        "heif"
    ],
    accept_multiple_files=True
)

# --------------------------------------------------
# PROCESS IMAGES
# --------------------------------------------------

if uploaded_files:

    progress = st.progress(0)

    for index, uploaded_file in enumerate(uploaded_files):

        try:

            rgb, gray = load_image(uploaded_file)

            focus_map = create_focus_map(gray)

            heatmap = create_heatmap(
                focus_map
            )

            overlay = create_overlay(
                rgb,
                heatmap
            )

            boxed_image = find_sharpest_region(
                rgb,
                gray
            )

            st.divider()

            st.subheader(uploaded_file.name)

            col1, col2 = st.columns(2)

            with col1:
                st.image(
                    boxed_image,
                    caption="Sharpest Region Detected",
                    width=450
                )

            with col2:
                st.image(
                    overlay,
                    caption="Focus Heatmap Overlay",
                    width=450
                )

            with st.expander("View Focus Heatmap"):
                st.image(
                    heatmap,
                    width=600
                )

        except Exception as e:

            st.error(
                f"Could not process {uploaded_file.name}: {e}"
            )

        progress.progress(
            (index + 1) / len(uploaded_files)
        )

    st.success("Analysis complete.")