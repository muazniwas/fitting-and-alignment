# IT5437 Computer Vision — Assignment 2: Fitting and Alignment

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install numpy pandas opencv-python pillow
```

## Scripts

### `lines_fitting.py` — Line Fitting

Fits lines to point-scatter data in `lines.csv` (columns `x1–x3`, `y1–y3`).

**Part 1 — Total Least Squares (TLS)**
Fits a line to the first 5 rows of Line 1 using SVD-based orthogonal regression. Minimises perpendicular distances rather than vertical residuals.

**Part 2 — RANSAC + TLS**
Pools all 300 points (100 rows × 3 lines), then iteratively applies RANSAC to find each of the three lines. After each fit the inlier consensus set is masked out before fitting the next line.

```bash
python lines_fitting.py
```

---

### `earring_size.py` — Earring Size Estimation

Estimates the real-world diameter of hoop earrings from `images/earrings.jpg` using camera parameters and Hough circle detection.

**Camera parameters used:**
| Parameter | Value |
|---|---|
| Focal length | 8 mm |
| Pixel size | 2.2 µm × 2.2 µm |
| Object distance | 720 mm |

**Method:** thin lens equation → magnification → mm/pixel scale → Hough circle radius × scale = real diameter.

```bash
python earring_size.py
```

Output saved to `outputs/earrings_annotated.jpg`.

---

### `homography.py` — Homography and Image Alignment

Aligns two circuit board images (`images/c1.jpg`, `images/c2.jpg`) and finds differences between them.

```bash
python homography.py
```

All windows close with **Q**, **Esc**, or the window's X button. Comment/uncomment parts in `__main__` to run only what you need.

**Part (a) — Manual homography**
Click 6 corresponding points in each image. Computes H via plain DLT (`cv2.findHomography` with no RANSAC) and warps `c1` to the perspective of `c2`.
Output: `outputs/a_warped.jpg`

**Part (b) — Difference image (manual)**
Pixel-wise absolute difference between `c2` and the warped `c1`. Bright regions indicate differences between the two boards.
Output: `outputs/b_difference.jpg`

**Part (c) — SIFT keypoint matching**
Detects SIFT keypoints and descriptors in both images, matches them with a brute-force matcher, and filters with Lowe's ratio test (threshold 0.75). Displays matched keypoint pairs.
Output: `outputs/c_matches.jpg`

**Part (d) — Automatic homography + difference**
Uses the SIFT matches from (c) to compute H via RANSAC (`ransacReprojThreshold=5.0`), warps `c1`, and computes the difference image.
Outputs: `outputs/d_warped.jpg`, `outputs/d_difference.jpg`

## Outputs

| File | Description |
|---|---|
| `outputs/a_warped.jpg` | c1 warped to c2's perspective (manual points) |
| `outputs/b_difference.jpg` | Difference image from manual homography |
| `outputs/c_matches.jpg` | SIFT keypoint matches between c1 and c2 |
| `outputs/d_warped.jpg` | c1 warped to c2's perspective (SIFT + RANSAC) |
| `outputs/d_difference.jpg` | Difference image from automatic homography |
| `outputs/earrings_annotated.jpg` | Detected earring circles with measured diameters |

## Project Structure

```
.
├── images/
│   ├── c1.jpg              # Circuit board 1
│   ├── c2.jpg              # Circuit board 2
│   └── earrings.jpg        # Hoop earrings image
├── outputs/                # Generated result images
├── lines.csv               # Point scatter data for three lines
├── lines_fitting.py        # TLS and RANSAC line fitting
├── earring_size.py         # Camera-based earring size estimation
└── homography.py           # Homography estimation and image alignment
```
