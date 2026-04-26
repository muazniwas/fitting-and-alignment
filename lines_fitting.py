import numpy as np


def fit_line_tls(x, y):
    """Total Least Squares line fit via SVD.
    Returns (a, b, c) for ax + by + c = 0, with a^2 + b^2 = 1.
    """
    x_mean, y_mean = np.mean(x), np.mean(y)
    A = np.column_stack([x - x_mean, y - y_mean])
    _, _, Vt = np.linalg.svd(A)
    a, b = Vt[-1]
    c = -(a * x_mean + b * y_mean)
    return a, b, c


def point_line_distances(x, y, a, b, c):
    """Perpendicular distances from points to line ax + by + c = 0."""
    return np.abs(a * x + b * y + c)  # a,b already unit-normalised


def ransac_line(x, y, n_iter=1000, threshold=0.3, rng=None):
    """RANSAC line fit. Returns (a, b, c) and boolean inlier mask."""
    if rng is None:
        rng = np.random.default_rng(42)

    n = len(x)
    best_inliers = np.zeros(n, dtype=bool)

    for _ in range(n_iter):
        idx = rng.choice(n, size=2, replace=False)
        a, b, c = fit_line_tls(x[idx], y[idx])
        dists = point_line_distances(x, y, a, b, c)
        inliers = dists < threshold
        if inliers.sum() > best_inliers.sum():
            best_inliers = inliers

    # Refit with TLS on all consensus points
    a, b, c = fit_line_tls(x[best_inliers], y[best_inliers])
    # Recompute inliers after the refined fit
    best_inliers = point_line_distances(x, y, a, b, c) < threshold
    return a, b, c, best_inliers


def report(line_num, a, b, c, n_inliers):
    print(f"Line {line_num}")
    print(f"  Equation : {a:+.6f}*x {b:+.6f}*y {c:+.6f} = 0")
    if abs(b) > 1e-10:
        print(f"  Slope    : {-a/b:.6f}")
        print(f"  Intercept: {-c/b:.6f}")
    print(f"  Inliers  : {n_inliers}")
    print()


if __name__ == "__main__":
    # --- Part 1: TLS on the first 5 rows of Line 1 only ---
    x1_sample = np.array([-4.4422, -5.3055, -5.5404, -4.9821, -4.4957])
    y1_sample = np.array([-12.4150, -12.6663, -11.0077, -11.6973, -11.9780])
    a, b, c = fit_line_tls(x1_sample, y1_sample)
    print("Total Least Squares — Line 1 (first 5 rows)")
    print("=" * 50)
    print(f"  Equation : {a:+.6f}*x {b:+.6f}*y {c:+.6f} = 0")
    print(f"  a        : {a:.6f}")
    print(f"  b        : {b:.6f}")
    print(f"  c        : {c:.6f}")
    if abs(b) > 1e-10:
        print(f"  Slope    : {-a/b:.6f}")
        print(f"  Intercept: {-c/b:.6f}")
    print()

    # --- Part 2: RANSAC + TLS on full dataset ---
    D = np.genfromtxt("lines.csv", delimiter=",", skip_header=1)
    X_cols = D[:, :3]
    Y_cols = D[:, 3:]
    X_all = X_cols.flatten()
    Y_all = Y_cols.flatten()

    rng = np.random.default_rng(42)
    active = np.ones(len(X_all), dtype=bool)

    print("RANSAC + Total Least Squares — Three-Line Fit")
    print("=" * 50)

    for line_num in range(1, 4):
        x_active = X_all[active]
        y_active = Y_all[active]

        a, b, c, local_inliers = ransac_line(x_active, y_active, rng=rng)

        # Map local inlier indices back to the full array
        active_indices = np.where(active)[0]
        global_inliers = np.zeros(len(X_all), dtype=bool)
        global_inliers[active_indices[local_inliers]] = True

        report(line_num, a, b, c, local_inliers.sum())

        # Mask out the consensus set before next iteration
        active[global_inliers] = False
