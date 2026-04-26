import cv2
import numpy as np

# --- Camera parameters ---
f_mm        = 8.0       # focal length (mm)
pixel_um    = 2.2       # pixel size (µm)
pixel_mm    = pixel_um * 1e-3  # pixel size (mm)
u_mm        = 720.0     # object distance: lens to earring plane (mm)

# Image distance via thin lens equation: 1/v = 1/f - 1/u
v_mm = 1.0 / (1.0/f_mm - 1.0/u_mm)
m    = v_mm / u_mm                   # magnification
mm_per_pixel = pixel_mm / m          # real-world mm each pixel represents

print("Camera setup")
print("============")
print(f"  Image distance v     : {v_mm:.4f} mm")
print(f"  Magnification m      : {m:.6f}")
print(f"  Scale (mm/pixel)     : {mm_per_pixel:.4f} mm/pixel")
print()

# --- Load image and detect circles ---
img   = cv2.imread("images/earrings.jpg")
gray  = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blur  = cv2.GaussianBlur(gray, (9, 9), 2)

circles = cv2.HoughCircles(
    blur,
    cv2.HOUGH_GRADIENT,
    dp=1,
    minDist=100,
    param1=50,
    param2=30,
    minRadius=100,
    maxRadius=450,
)

if circles is None:
    print("No circles detected — try adjusting HoughCircles parameters.")
else:
    circles = np.round(circles[0]).astype(int)
    # Sort left to right so earring numbering is consistent
    circles = sorted(circles, key=lambda c: c[0])

    print(f"Detected {len(circles)} earring(s)")
    print("=" * 40)
    for i, (cx, cy, r) in enumerate(circles, start=1):
        diameter_px  = 2 * r
        diameter_mm  = diameter_px * mm_per_pixel
        print(f"Earring {i}")
        print(f"  Centre (pixels)  : ({cx}, {cy})")
        print(f"  Radius (pixels)  : {r}")
        print(f"  Diameter (pixels): {diameter_px}")
        print(f"  Diameter (mm)    : {diameter_mm:.2f} mm")
        print()

    # --- Annotate and save result image ---
    annotated = img.copy()
    for i, (cx, cy, r) in enumerate(circles, start=1):
        cv2.circle(annotated, (cx, cy), r, (0, 255, 0), 2)
        cv2.circle(annotated, (cx, cy), 3, (0, 0, 255), -1)
        cv2.putText(annotated, f"E{i}: {2*r*mm_per_pixel:.1f}mm",
                    (cx - r, cy - r - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imwrite("outputs/earrings_annotated.jpg", annotated)
    print("Annotated image saved to outputs/earrings_annotated.jpg")
