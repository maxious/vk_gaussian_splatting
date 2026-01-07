import os
import sys
import glob
import cv2
import yaml
import numpy as np
import threading
from concurrent.futures import ThreadPoolExecutor
from queue import Queue, Empty
from pathlib import Path
import time

# Attempt to import Dear PyGui
try:
    import dearpygui.dearpygui as dpg
except ImportError:
    print("Dear PyGui is not installed. Please install it with: uv pip install dearpygui")
    sys.exit(1)

# Attempt to import PyNvJpeg (or similar) for GPU acceleration
# Note: The pypi package is 'nvidia-nvjpeg', but python bindings might be different.
# Often people use `pip install pynvjpeg` or `pip install nvjpeg-python`.
# Since we can't easily install system libs, we'll implement a flexible loader
# that prefers GPU but falls back to CPU (cv2/turbojpeg).
try:
    import nvjpeg

    GPU_DECODING_AVAILABLE = True
except ImportError:
    GPU_DECODING_AVAILABLE = False

# Dataset Paths
DATASET_ROOTS = [
    r"D:\SIGA2025VVC-Dataset\compression\test\006_1_seq1",
    r"D:\SIGA2025VVC-Dataset\compression\test\007_0_seq1",
]


class Camera:
    def __init__(self, name, K, R, T, dist_coeffs):
        self.name = name
        self.K = K
        self.R = R
        self.T = T
        self.dist_coeffs = dist_coeffs
        self.enabled = False
        self.position = -np.dot(R.T, T).flatten()  # C = -R^T * t
        self.color = (255, 0, 0, 255)  # Default disabled color (Red)
        self.subset = "Unknown"  # Train/Test/Unknown

    def set_enabled(self, enabled):
        self.enabled = enabled
        self.color = (0, 255, 0, 255) if enabled else (255, 0, 0, 255)


class Dataset:
    def __init__(self, root_path):
        self.root = Path(root_path)
        self.name = self.root.name
        self.images_dir = self.root / "images.tar" / "images"
        if not self.images_dir.exists():
            # Fallback if tar is extracted differently
            self.images_dir = self.root / "images"

        self.cameras = {}
        self.num_frames = 300  # Default based on description
        self.load_cameras()
        self.scan_frames()

    def load_cameras(self):
        # Load intrinsics and extrinsics
        # Preference: no prefix (all), then test/train if needed.
        # Based on file listing: extri.yml, intri.yml are present.

        intri_path = self.root / "intri.yml"
        extri_path = self.root / "extri.yml"

        if not intri_path.exists() or not extri_path.exists():
            print(f"Warning: Calibration files not found in {self.root}")
            return

        with open(intri_path, "r") as f:
            # OpenCV YAML format is weird, PyYAML might not parse !!opencv-matrix directly
            # without custom constructor.
            # We'll use a robust parser or text parsing for simplicity/speed if PyYAML fails.
            try:
                intri_data = yaml.safe_load(f)
            except yaml.YAMLError:
                f.seek(0)
                intri_data = self.parse_opencv_yaml_manual(f.read())

        with open(extri_path, "r") as f:
            try:
                extri_data = yaml.safe_load(f)
            except yaml.YAMLError:
                f.seek(0)
                extri_data = self.parse_opencv_yaml_manual(f.read())

        names = intri_data.get("names", [])
        if not names and "names" in extri_data:
            names = extri_data["names"]

        # If yaml loading failed to produce a dict with keys, manually parse
        if not isinstance(intri_data, dict) or not names:
            # re-read for manual parse
            with open(intri_path, "r") as f:
                intri_text = f.read()
            with open(extri_path, "r") as f:
                extri_text = f.read()
            intri_data = self.parse_opencv_yaml_manual(intri_text)
            extri_data = self.parse_opencv_yaml_manual(extri_text)
            names = intri_data.get("names", [])

        # Load train/test subsets
        train_path = self.root / "train_intri.yml"
        test_path = self.root / "test_intri.yml"
        train_names = []
        test_names = []

        if train_path.exists():
            with open(train_path, "r") as f:
                train_names = self.parse_opencv_yaml_manual(f.read()).get("names", [])
        if test_path.exists():
            with open(test_path, "r") as f:
                test_names = self.parse_opencv_yaml_manual(f.read()).get("names", [])

        # Clean names
        train_names = [str(n).strip("\"'") for n in train_names]
        test_names = [str(n).strip("\"'") for n in test_names]

        for name in names:
            # Clean name (remove quotes if needed)
            name = str(name).strip("\"'")

            # Intrinsics
            K_list = intri_data.get(f"K_{name}")
            D_list = intri_data.get(f"D_{name}")
            K = np.array(K_list).reshape(3, 3) if K_list else np.eye(3)

            # Extrinsics
            # extri.yml has R (Rodrigues?) or Rot (3x3)?
            # File content shows R_XX (3x1 Rodrigues) and Rot_XX (3x3 matrix) and T_XX (3x1)
            R_list = extri_data.get(f"Rot_{name}")
            if not R_list:
                # Try Rodrigues
                rvec = np.array(extri_data.get(f"R_{name}"))
                if rvec is not None:
                    R_list, _ = cv2.Rodrigues(rvec)

            R = np.array(R_list).reshape(3, 3) if R_list is not None else np.eye(3)
            T = (
                np.array(extri_data.get(f"T_{name}")).reshape(3, 1)
                if f"T_{name}" in extri_data
                else np.zeros((3, 1))
            )

            cam = Camera(name, K, R, T, D_list)
            if name in train_names:
                cam.subset = "Train"
            elif name in test_names:
                cam.subset = "Test"
            self.cameras[name] = cam

    def parse_opencv_yaml_manual(self, text):
        # Quick and dirty parser for the specific OpenCV YAML format shown
        data = {}
        lines = text.split("\n")
        current_key = None
        current_data = []
        reading_data = False

        names_mode = False

        for line in lines:
            line = line.strip()
            if not line:
                continue
            if line.startswith("%YAML"):
                continue
            if line.startswith("---"):
                continue

            if line.startswith("names:"):
                names_mode = True
                data["names"] = []
                continue

            if names_mode:
                if line.startswith("-"):
                    val = line[1:].strip().strip("\"'")
                    data["names"].append(val)
                elif ":" in line:
                    names_mode = False
                else:
                    continue

            if names_mode:
                continue

            if ":" in line and not line.startswith("-"):
                key = line.split(":")[0].strip()
                if "!!opencv-matrix" in line:
                    current_key = key
                    reading_data = False
                elif current_key and key == "data":
                    # data: [x, y, z]
                    val_str = line.split("[")[1].split("]")[0]
                    vals = [float(x) for x in val_str.split(",")]
                    data[current_key] = vals
                    current_key = None
        return data

    def scan_frames(self):
        # Check first camera folder to count frames
        if not self.cameras:
            return
        first_cam = list(self.cameras.keys())[0]
        cam_dir = self.images_dir / first_cam
        if cam_dir.exists():
            files = sorted(glob.glob(str(cam_dir / "*.jpg")))
            self.num_frames = len(files)
            print(f"Found {self.num_frames} frames in {self.name}")

    def get_image_path(self, cam_name, frame_idx):
        # 000000.jpg format
        filename = f"{frame_idx:06d}.jpg"
        return str(self.images_dir / cam_name / filename)


class ImageLoader:
    def __init__(self, use_gpu=False):
        self.use_gpu = use_gpu and GPU_DECODING_AVAILABLE
        if self.use_gpu:
            self.nvjpeg = nvjpeg.NvJpeg()  # Hypothetical API
        self.executor = ThreadPoolExecutor(max_workers=8)
        self.cache = {}  # Simple cache (could use LRU)
        self.lock = threading.Lock()

    def load_image(self, path, scale=0.25):
        # Downscale for viewing performance (4K is too big for grid view)
        # Returns (width, height, channels, data_rgba)
        if path in self.cache:
            return self.cache[path]

        # Determine target size if scaling
        # We decode full then resize, or use turbojpeg scaling if available.
        # For CV2:
        img = cv2.imread(path)
        if img is None:
            return None

        # Convert BGR to RGBA
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGBA)

        if scale != 1.0:
            h, w = img.shape[:2]
            new_w, new_h = int(w * scale), int(h * scale)
            img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # Normalize to 0-1 float for DPG? No, DPG texture takes float or int.
        # Actually DPG raw texture needs flat array.

        h, w, c = img.shape
        data = img.flatten() / 255.0  # DPG expects float array

        # Cache
        # self.cache[path] = (w, h, c, data)
        return (w, h, c, data)


class ViewerApp:
    def __init__(self):
        self.datasets = []
        self.current_dataset_idx = 0
        self.frame_idx = 0
        self.is_playing = False
        self.play_speed = 30.0  # FPS
        self.last_update_time = time.time()

        self.loader = ImageLoader(use_gpu=False)  # Enable if package found
        self.texture_registry = {}  # path -> dpg_tag

        self.load_datasets()
        self.init_gui()

    def load_datasets(self):
        for path in DATASET_ROOTS:
            if os.path.exists(path):
                print(f"Loading dataset: {path}")
                ds = Dataset(path)
                self.datasets.append(ds)

        if not self.datasets:
            print("No datasets found!")
            sys.exit(1)

        # Enable some cameras by default
        ds = self.datasets[0]
        # Enable first 4 cameras
        count = 0
        for name in sorted(ds.cameras.keys()):
            if count < 4:
                ds.cameras[name].set_enabled(True)
                count += 1

    def init_gui(self):
        dpg.create_context()
        dpg.create_viewport(title="High Performance Multi-View Viewer", width=1600, height=900)

        with dpg.texture_registry(show=False):
            # Reserve some static textures or dynamic
            pass

        with dpg.window(tag="Primary Window"):
            with dpg.group(horizontal=True):
                # Left Panel: Controls
                with dpg.child_window(width=300):
                    dpg.add_text("Datasets")
                    dpg.add_combo(
                        [d.name for d in self.datasets],
                        default_value=self.datasets[0].name,
                        callback=self.change_dataset,
                    )

                    dpg.add_separator()
                    dpg.add_text("Playback")
                    with dpg.group(horizontal=True):
                        dpg.add_button(label="Play/Pause", callback=self.toggle_play)
                        dpg.add_slider_float(
                            label="FPS",
                            default_value=30.0,
                            min_value=1.0,
                            max_value=60.0,
                            width=100,
                            callback=lambda s, a: setattr(self, "play_speed", a),
                        )

                    self.scrubber = dpg.add_slider_int(
                        label="Frame",
                        default_value=0,
                        min_value=0,
                        max_value=self.datasets[0].num_frames - 1,
                        callback=self.scrub_callback,
                    )

                    dpg.add_separator()
                    dpg.add_text("Cameras")
                    self.camera_list_group = dpg.add_group()
                    self.update_camera_list()

                # Middle: Image Grid
                with dpg.child_window(width=-300):
                    self.image_grid = dpg.add_group()

                # Right: 3D Visualization
                with dpg.child_window(width=300):
                    dpg.add_text("3D Scene")
                    with dpg.plot(label="Camera Positions", height=-1, width=-1, tag="plot_3d"):
                        dpg.add_plot_legend()
                        # X-Y Plane (Top down?) or we project 3D to 2D
                        # DPG plots are 2D. We'll plot X-Z (top down) or X-Y.
                        # Standard computer vision: Y down, Z forward. So X-Z is top-down.
                        self.plot_xaxis = dpg.add_plot_axis(dpg.mvXAxis, label="X")
                        self.plot_yaxis = dpg.add_plot_axis(dpg.mvYAxis, label="Z")

                        # We will add scatter series here
                        self.camera_scatter_series = dpg.add_scatter_series(
                            [], [], label="Cameras", parent=self.plot_yaxis
                        )

        dpg.setup_dearpygui()
        dpg.show_viewport()
        dpg.set_primary_window("Primary Window", True)

        # Start render loop
        while dpg.is_dearpygui_running():
            self.render_loop()
            dpg.render_dearpygui_frame()

        dpg.destroy_context()

    def render_loop(self):
        # Update Playback
        if self.is_playing:
            dt = time.time() - self.last_update_time
            if dt > (1.0 / self.play_speed):
                self.frame_idx = (self.frame_idx + 1) % self.datasets[
                    self.current_dataset_idx
                ].num_frames
                dpg.set_value(self.scrubber, self.frame_idx)
                self.last_update_time = time.time()
                self.update_images()

        # Update 3D plot
        self.update_3d_plot()

    def change_dataset(self, sender, app_data):
        # Find dataset index
        for i, ds in enumerate(self.datasets):
            if ds.name == app_data:
                self.current_dataset_idx = i
                break

        # Update slider limits
        ds = self.datasets[self.current_dataset_idx]
        dpg.configure_item(self.scrubber, max_value=ds.num_frames - 1)
        self.update_camera_list()
        self.update_images()

    def update_camera_list(self):
        dpg.delete_item(self.camera_list_group, children_only=True)
        ds = self.datasets[self.current_dataset_idx]

        # Filter buttons?
        with dpg.group(parent=self.camera_list_group):
            dpg.add_button(label="Enable All", callback=lambda: self.set_all_cameras(True))
            dpg.add_button(label="Disable All", callback=lambda: self.set_all_cameras(False))

        # Scrollable list
        with dpg.child_window(parent=self.camera_list_group, height=-1):
            sorted_cams = sorted(ds.cameras.keys())
            for name in sorted_cams:
                cam = ds.cameras[name]
                label_text = f"Cam {name} [{cam.subset}]"
                dpg.add_checkbox(
                    label=label_text,
                    default_value=cam.enabled,
                    user_data=name,
                    callback=self.toggle_camera,
                )

    def set_all_cameras(self, enabled):
        ds = self.datasets[self.current_dataset_idx]
        for cam in ds.cameras.values():
            cam.set_enabled(enabled)
        self.update_camera_list()
        self.update_images()

    def toggle_camera(self, sender, app_data, user_data):
        ds = self.datasets[self.current_dataset_idx]
        if user_data in ds.cameras:
            ds.cameras[user_data].set_enabled(app_data)
        self.update_images()

    def scrub_callback(self, sender, app_data):
        self.frame_idx = app_data
        self.update_images()

    def toggle_play(self):
        self.is_playing = not self.is_playing
        self.last_update_time = time.time()

    def update_images(self):
        ds = self.datasets[self.current_dataset_idx]
        enabled_cams = [c for c in ds.cameras.values() if c.enabled]

        # Re-layout grid if number of cameras changed substantially?
        # For now just clear and rebuild. Efficient enough for DPG?
        dpg.delete_item(self.image_grid, children_only=True)

        if not enabled_cams:
            return

        # Calculate grid size
        n = len(enabled_cams)
        cols = int(np.ceil(np.sqrt(n)))
        rows = int(np.ceil(n / cols))

        # Image size (scaled)
        # We need to load them first

        # Launch async loading
        # Ideally we want to just update textures, not delete/add items.
        # But for dynamic camera lists, rebuilding is easier.

        with dpg.group(parent=self.image_grid):
            for i in range(rows):
                with dpg.group(horizontal=True):
                    for j in range(cols):
                        idx = i * cols + j
                        if idx >= n:
                            break

                        cam = enabled_cams[idx]
                        img_path = ds.get_image_path(cam.name, self.frame_idx)

                        # Load image (cached)
                        # We use a downscale of 0.125 (1/8) for 4K -> 500px roughly
                        img_data = self.loader.load_image(img_path, scale=0.125)

                        if img_data:
                            w, h, c, data = img_data

                            # DPG Texture - Use Dataset Name in tag to prevent cache collisions
                            tag = f"tex_{ds.name}_{cam.name}_{self.frame_idx}"

                            # Check if texture exists in registry
                            if not dpg.does_item_exist(tag):
                                with dpg.texture_registry():
                                    dpg.add_static_texture(
                                        width=w, height=h, default_value=data, tag=tag
                                    )

                            # Add image
                            with dpg.group():
                                dpg.add_text(f"{cam.name} [{cam.subset}]")
                                dpg.add_image(tag)
                        else:
                            dpg.add_text(f"Error loading {cam.name}")

    def update_3d_plot(self):
        # Update scatter points
        ds = self.datasets[self.current_dataset_idx]

        xs = []
        zs = []  # Using Z as Y axis for 2D plot
        # Colors not supported easily in single scatter series in DPG unless we use themes or separate series
        # We'll use two series: enabled and disabled

        xs_en, zs_en = [], []
        xs_dis, zs_dis = [], []

        for cam in ds.cameras.values():
            pos = cam.position
            # OpenCV camera: X right, Y down, Z forward.
            # Plot X-Z (top down view)
            if cam.enabled:
                xs_en.append(pos[0])
                zs_en.append(pos[2])
            else:
                xs_dis.append(pos[0])
                zs_dis.append(pos[2])

        # We need distinct tags for updating
        if not dpg.does_item_exist("series_enabled"):
            dpg.add_scatter_series(
                xs_en, zs_en, label="Enabled", parent=self.plot_yaxis, tag="series_enabled"
            )
            dpg.add_scatter_series(
                xs_dis, zs_dis, label="Disabled", parent=self.plot_yaxis, tag="series_disabled"
            )
        else:
            dpg.configure_item("series_enabled", x=xs_en, y=zs_en)
            dpg.configure_item("series_disabled", x=xs_dis, y=zs_dis)

        # Add labels over points
        # Clear old annotations (naive way: delete and recreate plot or track tags)
        # Since we don't store tags for annotations, let's use a fixed set of tags if camera count is constant
        # Or just delete all children of plot that are annotations?
        # Simpler: Generate tags based on cam name and update them.

        for name, cam in ds.cameras.items():
            pos = cam.position
            tag = f"anno_{ds.name}_{name}"
            # Check if annotation exists
            if not dpg.does_item_exist(tag):
                dpg.add_plot_annotation(
                    label=name,
                    default_value=(pos[0], pos[2]),
                    offset=(5, 5),
                    color=cam.color,
                    parent="plot_3d",
                    tag=tag,
                )
            else:
                dpg.configure_item(tag, color=cam.color, default_value=(pos[0], pos[2]))


if __name__ == "__main__":
    app = ViewerApp()
