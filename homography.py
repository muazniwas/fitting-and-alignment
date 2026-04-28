import cv2
import numpy as np

N = 6  # number of corresponding points to click
SCALE = 0.3  # display scale factor (images are large)

# ── helpers ──────────────────────────────────────────────────────────────────

def resize(img):
    h, w = img.shape[:2]
    return cv2.resize(img, (int(w * SCALE), int(h * SCALE)))


# ── point collection via mouse clicks ────────────────────────────────────────

points = []

def on_click(event, x, y, flags, param):
    img_display, pts, scale = param
    if event == cv2.EVENT_LBUTTONDOWN:
        pts.append((x / scale, y / scale))   # store in original-image coords
        cv2.circle(img_display, (x, y), 6, (0, 0, 255), -1)
        cv2.putText(img_display, str(len(pts)),
                    (x + 8, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)


def collect_points(window_name, img):
    """Open a window, let the user click N points, return them as (N,2) array."""
    display = resize(img.copy())
    pts = []
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(window_name, on_click, param=(display, pts, SCALE))
    print(f"[{window_name}] Click {N} points, then press any key.")
    while True:
        cv2.imshow(window_name, display)
        if len(pts) >= N:
            cv2.waitKey(500)
            break
        if cv2.waitKey(20) & 0xFF == 27:
            break
    cv2.destroyWindow(window_name)
    return np.array(pts, dtype=np.float32)


# ── part (a): manual homography + warp ───────────────────────────────────────

def part_a(im1, im2):
    print("=== Part (a): Manual Homography ===")
    print("Click corresponding points in Image 1 then Image 2.")

    p1 = collect_points("Image 1 — click points", im1)
    p2 = collect_points("Image 2 — click points", im2)

    print(f"Points in Image 1:\n{p1}")
    print(f"Points in Image 2:\n{p2}")

    H, mask = cv2.findHomography(p1, p2, method=0)  # plain DLT, no RANSAC
    print(f"\nHomography matrix H:\n{H}")

    # Warp im1 to the perspective of im2
    h2, w2 = im2.shape[:2]
    warped = cv2.warpPerspective(im1, H, (w2, h2))

    cv2.imwrite("outputs/a_warped.jpg", warped)
    print("Warped image saved to outputs/a_warped.jpg")

    # Display side by side (resized)
    combined = np.hstack([resize(im2), resize(warped)])
    win = "Part (a) — Original im2  |  Warped im1  [press Q or close to exit]"
    cv2.imshow(win, combined)
    while True:
        key = cv2.waitKey(20) & 0xFF
        if key == ord('q') or key == 27:
            break
        if cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
            break
    cv2.destroyAllWindows()

    return H, warped


# ── part (b): difference image ────────────────────────────────────────────────

def part_b(im2, warped):
    print("\n=== Part (b): Difference Image ===")

    diff = cv2.absdiff(im2, warped)

    cv2.imwrite("outputs/b_difference.jpg", diff)
    print("Difference image saved to outputs/b_difference.jpg")

    win = "Part (b) — Difference image  [press Q or close to exit]"
    cv2.imshow(win, resize(diff))
    while True:
        key = cv2.waitKey(20) & 0xFF
        if key == ord('q') or key == 27:
            break
        if cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
            break
    cv2.destroyAllWindows()

    return diff


# ── part (c): SIFT keypoints, descriptors, and matches ───────────────────────

def part_c(im1, im2):
    print("\n=== Part (c): SIFT Keypoints & Matching ===")

    gray1 = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)

    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(gray1, None)
    kp2, des2 = sift.detectAndCompute(gray2, None)
    print(f"Keypoints — Image 1: {len(kp1)},  Image 2: {len(kp2)}")

    # BFMatcher + Lowe's ratio test to keep only good matches
    bf = cv2.BFMatcher(cv2.NORM_L2)
    raw = bf.knnMatch(des1, des2, k=2)
    good = [m for m, n in raw if m.distance < 0.75 * n.distance]
    print(f"Good matches after ratio test: {len(good)}")

    match_img = cv2.drawMatches(
        im1, kp1, im2, kp2, good, None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )

    cv2.imwrite("outputs/c_matches.jpg", match_img)
    print("Match image saved to outputs/c_matches.jpg")

    win = "Part (c) — SIFT matches  [press Q or close to exit]"
    cv2.imshow(win, resize(match_img))
    while True:
        key = cv2.waitKey(20) & 0xFF
        if key == ord('q') or key == 27:
            break
        if cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
            break
    cv2.destroyAllWindows()

    return kp1, kp2, good


# ── part (d): automatic homography + warp + diff ─────────────────────────────

def part_d(im1, im2, kp1, kp2, good):
    print("\n=== Part (d): Automatic Homography via SIFT Matches ===")

    if len(good) < 4:
        print("Not enough matches to compute homography.")
        return None, None

    src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, ransacReprojThreshold=5.0)
    inliers = int(mask.sum())
    print(f"Homography matrix H:\n{H}")
    print(f"RANSAC inliers: {inliers} / {len(good)}")

    # Warp im1 to perspective of im2
    h2, w2 = im2.shape[:2]
    warped = cv2.warpPerspective(im1, H, (w2, h2))

    cv2.imwrite("outputs/d_warped.jpg", warped)
    print("Warped image saved to outputs/d_warped.jpg")

    # Display warped side by side with im2
    combined = np.hstack([resize(im2), resize(warped)])
    win = "Part (d) — Original im2  |  Auto-warped im1  [press Q or close to exit]"
    cv2.imshow(win, combined)
    while True:
        key = cv2.waitKey(20) & 0xFF
        if key == ord('q') or key == 27:
            break
        if cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
            break
    cv2.destroyAllWindows()

    # Difference image
    diff = cv2.absdiff(im2, warped)
    cv2.imwrite("outputs/d_difference.jpg", diff)
    print("Difference image saved to outputs/d_difference.jpg")

    win = "Part (d) — Difference image  [press Q or close to exit]"
    cv2.imshow(win, resize(diff))
    while True:
        key = cv2.waitKey(20) & 0xFF
        if key == ord('q') or key == 27:
            break
        if cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
            break
    cv2.destroyAllWindows()

    return H, diff


# ── main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import os
    os.makedirs("outputs", exist_ok=True)

    im1 = cv2.imread("images/c1.jpg")
    im2 = cv2.imread("images/c2.jpg")

    # Comment out parts you don't want to re-run
    # H_manual, warped_manual = part_a(im1, im2)
    # diff_manual             = part_b(im2, warped_manual)

    kp1, kp2, good    = part_c(im1, im2)
    H_auto, diff_auto = part_d(im1, im2, kp1, kp2, good)
