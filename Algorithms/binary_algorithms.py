EPS = 1e-12


class NoAcceptedStartingPointError(RuntimeError):
    pass


def squared_euclidean_distance(a, b):
    # Squared Euclidean distance used by the matcher.
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return float(np.sum((a - b) ** 2))


def authenticate(target, probe, threshold=THRESHOLD):
    # Binary matcher: True = accept, False = reject.
    return squared_euclidean_distance(probe, target) < threshold


def random_step_sed(x, step_size):
    # Random Euclidean step with squared length = step_size.
    direction = np.random.randn(*x.shape)
    norm = np.linalg.norm(direction)

    while norm <= EPS:
        direction = np.random.randn(*x.shape)
        norm = np.linalg.norm(direction)

    direction /= norm
    return np.asarray(x, dtype=np.float64) + direction * math.sqrt(step_size)


def find_inside_point(data, target, threshold, max_items):
    max_items = min(max_items, len(data))

    for i in range(max_items):
        probe = np.asarray(data[i], dtype=np.float64)
        if authenticate(target, probe, threshold):
            return probe.copy(), i + 1

    raise NoAcceptedStartingPointError(
        f"No accepted probe among the first {max_items} breaking-set vectors."
    )


def find_outside_point(point_inside, target, threshold):
    # Keep trying until the random step is genuinely outside.
    queries = 0

    while True:
        candidate = random_step_sed(point_inside, threshold * 2.0)
        queries += 1

        if not authenticate(target, candidate, threshold):
            return candidate, queries


def find_boundary_point(point_inside, target, threshold, binary_steps):
    inside = np.asarray(point_inside, dtype=np.float64).copy()
    outside, outside_queries = find_outside_point(
        inside, target, threshold
    )

    middle = inside.copy()

    for _ in range(binary_steps):
        middle = (inside + outside) / 2.0

        if authenticate(target, middle, threshold):
            inside = middle
        else:
            outside = middle

    return middle, outside_queries + binary_steps, outside_queries


def solve_sphere_center(points):
    # Equal-radius sphere equations become linear after subtracting one equation.
    q = np.asarray(points, dtype=np.float64)

    if q.ndim != 2 or len(q) < 2:
        raise ValueError("Need a 2-D array with at least two boundary points.")

    q_last = q[-1]
    A = 2.0 * (q[:-1] - q_last)
    b = (
        np.sum(q[:-1] ** 2, axis=1)
        - np.sum(q_last ** 2)
    )

    recovered, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    return recovered


def binary_reconstruct(
    data,
    target,
    threshold=THRESHOLD,
    binary_steps=BINARY_STEPS,
    max_breaking=N_BREAKING_SET,
):
    point_inside, inside_queries = find_inside_point(
        data, target, threshold, max_breaking
    )

    d = target.shape[0]
    boundary_points = []
    boundary_queries = 0
    outside_queries = 0

    # d+1 boundary points for a d-dimensional SED sphere.
    for _ in range(d + 1):
        point, q, q_outside = find_boundary_point(
            point_inside,
            target,
            threshold,
            binary_steps,
        )
        boundary_points.append(point)
        boundary_queries += q
        outside_queries += q_outside

    recovered = solve_sphere_center(boundary_points)

    return recovered, {
        "queries": inside_queries + boundary_queries,
        "inside_queries": inside_queries,
        "boundary_queries": boundary_queries,
        "outside_search_queries": outside_queries,
    }


def averaging_baseline(
    data,
    target,
    threshold=THRESHOLD,
    max_breaking=N_BREAKING_SET,
):
    max_items = min(max_breaking, len(data))
    accepted = []

    for i in range(max_items):
        probe = np.asarray(data[i], dtype=np.float64)

        if authenticate(target, probe, threshold):
            accepted.append(probe)

    if not accepted:
        raise NoAcceptedStartingPointError(
            f"No accepted probe among the first {max_items} breaking-set vectors."
        )

    recovered = np.mean(np.stack(accepted, axis=0), axis=0)

    return recovered, {
        "queries": max_items,
        "accepted_count": len(accepted),
    }