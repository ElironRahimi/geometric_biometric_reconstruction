# Binary-Feedback Biometric Template Reconstruction

Research code for reconstructing a biometric template using **binary authentication feedback only** (`ACCEPT` / `REJECT`). The implementation demonstrates the geometry of a squared-Euclidean-distance (SED) matcher and compares the reconstruction method with an averaging baseline.

---

## Overview

A biometric matcher typically accepts a probe when its distance from the enrolled template is below a threshold:

```text
ACCEPT  if ||probe - target||² < threshold
REJECT  otherwise
```

Under squared Euclidean distance, the acceptance region is a hypersphere centered at the unknown target template. The reconstruction procedure exploits only binary matcher responses to estimate points on this decision boundary and recover the sphere center.

For each target template, the notebook:

1. Searches the breaking set for an accepted probe.
2. Moves from the accepted probe until an outside point is found.
3. Uses binary search to localize a point on the authentication boundary.
4. Repeats this process to obtain `d + 1` boundary points for a `d`-dimensional template.
5. Recovers the template by solving the resulting sphere equations with linear least squares.
6. Compares the reconstruction with an averaging baseline over accepted probes.

---

## Repository Structure

```text
Biometric_Reconstruction/
├── Algorithms/
│   └── binary_algorithms.py
├── main.ipynb
├── requirements.txt
└── README.md
```

The experiment expects the embedding files to be placed under a local `data/` directory:

```text
data/
└── lfw/
    └── facenet/
        ├── train_validation/
        │   └── train_validation.npy
        └── test_targets/
            └── test_targets.npy
```

The data files are **not included** in this repository.

---


## Installation

### 1. Clone the repository

```bash
git clone <YOUR-REPOSITORY-URL>
cd Biometric_Reconstruction
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it on Linux/macOS:

```bash
source .venv/bin/activate
```

or on Windows:

```powershell
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

Python **3.10+** is recommended.

---

## Running the Experiment

Place the required `.npy` embedding files in the directory structure shown above, then launch Jupyter:

```bash
jupyter lab
```

Open:

```text
main.ipynb
```

and run the notebook from top to bottom.

The notebook reports reconstruction statistics including:

- attack coverage,
- attack success rate,
- average query count,
- query standard deviation,
- average reconstruction loss,
- reconstruction-loss standard deviation,
- runtime per target.

The reconstruction loss is the squared Euclidean distance between the recovered and true templates:

$$
\mathcal{L}_{\mathrm{SED}} = \|\hat{t} - t\|_2^2.
$$

The notebook additionally reports whether the reconstruction loss is below 1% of the authentication threshold.

---

## Method Summary

### Binary reconstruction

Given one accepted probe, the method samples an outside point and repeatedly queries the matcher to localize the boundary between acceptance and rejection. After collecting enough boundary points, the center of the SED hypersphere is recovered using a linear least-squares system.

### Averaging baseline

The baseline evaluates the available breaking-set probes, keeps those accepted by the matcher, and estimates the target as their mean.

Targets for which no accepted starting probe exists in the evaluated breaking set are reported separately and excluded from reconstruction-loss statistics.

---