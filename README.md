# Jason-1 Satellite 6D Pose Estimation

**6-Degrees-of-Freedom (6D) object pose estimation**

This repository implements a computer vision pipeline to accurately determine the **3D position** and **3D orientation** of the JASON1 satellite from 2D image data. The core of the solution relies on the classic **Perspective-n-Point (PnP)** algorithm and **Blender** generated synthetic database images.

## 🌟 Features

-   **Full 6D Pose Estimation**: Calculates the (x,y,z) translation and the full 3D rotation (e.g., as a rotation matrix or Euler angles) of the object.

-   **Perspective-n-Point (PnP) Implementation**: Utilizes the PnP algorithm for efficient and accurate pose estimation based on correspondences between known 3D object points and their 2D image projections.

---

## 🛠️ Prerequisites

Ensure you have **Python 3.x** installed on your system.

---

## 🚀 Installation and Setup

Follow these steps to get the project running locally:

1.  **Clone the repository:**

    ```
    git clone https://github.com/dluksa20/jason1_6d_pose_estimation.git
    cd jason1_6d_pose_estimation
    ```

2.  **Create and activate a virtual environment:**

    ```
    python3 -m venv venv
    source venv/bin/activate  # macOS/Linux
    # .\\venv\\Scripts\\activate  # Windows (PowerShell)
    ```

3.  **Install the required dependencies:** .
    ```
    pip install -r requirements.txt
    ```


---

## 💡 Usage

The primary script for executing the pose estimation pipeline is `main.py`.

### Running the Estimation

To run the pose estimation process, execute the main script:

Bash

```
python main.py
```

**_Note:_** _You will likely need to ensure the **`database/`** directory contains the required **3D model data** and **camera calibration parameters** for the JASON1 object before running the script successfully._

---

## 📂 Project Structure

| File/Folder | Description |
| --- | --- |
| `main.py` | The entry point for the application. It handles image loading, calling the pose estimation functions, and visualization. |
| `PosePnP.py` | Contains the **core function** for computing the 6D pose using the PnP algorithm, including input preparation and output formatting. |
| `requirements.txt` | Lists all necessary Python libraries for the project. |
| `database/` | for storing database images 3D points, **camera intrinsic parameters**, and ground-truth data. |
| `scripts/` | Contains supplementary scripts, potentially for data generation, pre-processing, or evaluation. |

Export to Sheets

---

## 🤝 Contributing

Contributions are highly welcome! If you find a bug, have an idea for a new feature, or want to improve the existing code (e.g., adding RANSAC for robustness, or improving documentation), please:

1.  **Fork** the repository.

2.  Create a new **feature branch**.

3.  Commit your changes.

4.  Open a **Pull Request**.
