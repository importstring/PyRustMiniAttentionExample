"""Visualize attention dot products using two-dimensional vector arrows."""

import math

import numpy as np

from rustml import matmul, transpose

QUERY_COLOR = "#2563eb"
KEY_COLOR = "#ea580c"
TEXT_COLOR = "#172033"
BACKGROUND = "#f5f7fb"


def visualize_attention(tokens, Q, K, K_T, scores, scaled_scores):
    """Show each query-key dot product on an x,y graph.

    Controls:
        Space / Right: next step
        Left: previous step
        Home: restart
        Escape: close

    Q and K must have shape (number of tokens, 2).

    The supplied matrices remain the actual calculation results.
    Component products are reconstructed for explanation only.
    """
    tokens = [str(token) for token in tokens]

    Q, K, K_T, scores, scaled_scores = [
        np.asarray(array, dtype=float)
        for array in (Q, K, K_T, scores, scaled_scores)
    ]

    _validate(tokens, Q, K, K_T, scores, scaled_scores)

    import turtle

    n = len(tokens)
    divisor = math.sqrt(Q.shape[1])

    # Each token pair has five explanation stages.
    stages = [
        "Select query and key",
        "Multiply horizontal components",
        "Multiply vertical components",
        "Add the products",
        "Scale and store the score",
    ]

    total_steps = n * n * len(stages)
    step = 0

    screen = turtle.Screen()
    screen.setup(width=1200, height=850)
    screen.title("Attention: vector dot products")
    screen.bgcolor(BACKGROUND)
    screen.tracer(0)

    pen = turtle.Turtle(visible=False)
    pen.speed(0)
    pen.penup()

    # Graph occupies the left-hand side of the window.
    origin = (-290, 45)
    graph_radius = 225

    # Equal scaling on both axes preserves vector geometry.
    largest_component = float(
        max(np.max(np.abs(Q)), np.max(np.abs(K)))
    )
    graph_limit = max(1.0, largest_component * 1.25)
    pixels_per_unit = graph_radius / graph_limit

    def text(x, y, message, size=13, color=TEXT_COLOR):
        pen.goto(x, y)
        pen.pencolor(color)
        pen.write(
            message,
            align="left",
            font=("Arial", size, "normal"),
        )

    def line(start, end, color, width=1):
        pen.penup()
        pen.goto(start)
        pen.pencolor(color)
        pen.pensize(width)
        pen.pendown()
        pen.goto(end)
        pen.penup()
        pen.pensize(1)

    def point(vector):
        """Convert mathematical coordinates into screen coordinates."""
        return (
            origin[0] + float(vector[0]) * pixels_per_unit,
            origin[1] + float(vector[1]) * pixels_per_unit,
        )

    def arrow(vector, color, width):
        """Draw an arrow from the graph origin to a vector endpoint."""
        end = point(vector)
        dx = end[0] - origin[0]
        dy = end[1] - origin[1]
        length = math.hypot(dx, dy)

        if length < 0.001:
            pen.goto(origin)
            pen.dot(10, color)
            return

        line(origin, end, color, width)

        angle = math.atan2(dy, dx)
        head_length = min(15, length * 0.3)

        for offset in (-math.pi / 6, math.pi / 6):
            head = (
                end[0] - head_length * math.cos(angle + offset),
                end[1] - head_length * math.sin(angle + offset),
            )
            line(end, head, color, width)

    def draw_graph():
        """Draw grid lines, axes, and coordinate labels."""
        tick_spacing = graph_limit / 4

        for tick in range(-4, 5):
            value = tick * tick_spacing
            x, y = point((value, value))

            line(
                (x, origin[1] - graph_radius),
                (x, origin[1] + graph_radius),
                "#e0e5ee",
            )
            line(
                (origin[0] - graph_radius, y),
                (origin[0] + graph_radius, y),
                "#e0e5ee",
            )

            if tick != 0:
                text(x - 12, origin[1] - 23, f"{value:.2g}", 10)
                text(origin[0] + 8, y + 3, f"{value:.2g}", 10)

        line(
            (origin[0] - graph_radius, origin[1]),
            (origin[0] + graph_radius, origin[1]),
            TEXT_COLOR,
            2,
        )
        line(
            (origin[0], origin[1] - graph_radius),
            (origin[0], origin[1] + graph_radius),
            TEXT_COLOR,
            2,
        )

        text(origin[0] + graph_radius + 10, origin[1] - 5, "x")
        text(origin[0] + 8, origin[1] + graph_radius + 8, "y")
        text(origin[0] - 15, origin[1] - 23, "0", 10)

    def draw_components(vector, color, horizontal, offset):
        """Highlight a component without moving the actual vector."""
        x, y = map(float, vector)
        end = point(vector)
        corner = point((x, 0))

        # Coordinate guides connect the tip to the axes.
        line(end, corner, color, 1)
        line(end, point((0, y)), color, 1)

        if horizontal:
            start = (origin[0], origin[1] + offset)
            finish = (corner[0], origin[1] + offset)
            line(start, finish, color, 4)
        else:
            start = (corner[0] + offset, origin[1])
            finish = (corner[0] + offset, end[1])
            line(start, finish, color, 4)

    def box(x, y, width, height, fill):
        pen.goto(x, y)
        pen.setheading(0)
        pen.pencolor("#b9c3d3")
        pen.fillcolor(fill)
        pen.pendown()
        pen.begin_fill()

        for length in (width, height, width, height):
            pen.forward(length)
            pen.right(90)

        pen.end_fill()
        pen.penup()

    def draw_matrix(name, values, x, y, pair_index, stage):
        """Display completed cells and the current calculation cell."""
        cell_width = 66
        cell_height = 34

        text(x, y + 28, name, size=15)

        for column in range(n):
            text(
                x + column * cell_width + 24,
                y + 5,
                str(column),
                size=11,
            )

        for row in range(n):
            text(x - 22, y - row * cell_height - 23, str(row), 11)

            for column in range(n):
                cell_index = row * n + column
                current = cell_index == pair_index

                # Raw score appears at the sum stage.
                # Scaled score appears at the scaling stage.
                reveal_stage = 3 if name == "Raw scores" else 4
                visible = (
                    cell_index < pair_index
                    or (current and stage >= reveal_stage)
                )

                cell_x = x + column * cell_width
                cell_y = y - row * cell_height

                box(
                    cell_x,
                    cell_y,
                    cell_width,
                    cell_height,
                    "#ffe49a" if current else "#ffffff",
                )

                label = (
                    f"{values[row, column]:.3f}"
                    if visible else "..."
                )
                text(cell_x + 6, cell_y - 23, label, 11)

    def draw():
        pen.clear()

        pair_index, stage = divmod(step, len(stages))
        query_index, key_index = divmod(pair_index, n)

        query = Q[query_index]
        key = K[key_index]

        qx, qy = map(float, query)
        kx, ky = map(float, key)

        x_product = qx * kx
        y_product = qy * ky

        text(
            -550, 370,
            f"Query {query_index}: {tokens[query_index]}"
            f"  ->  Key {key_index}: {tokens[key_index]}",
            size=21,
        )
        text(-550, 337, stages[stage], size=16)

        draw_graph()

        # Draw the key thicker so both colors remain visible
        # when query and key overlap exactly.
        arrow(key, KEY_COLOR, 7)
        arrow(query, QUERY_COLOR, 3)

        if stage in (1, 2):
            horizontal = stage == 1
            draw_components(key, KEY_COLOR, horizontal, -5)
            draw_components(query, QUERY_COLOR, horizontal, 5)

        text(
            -550, -215,
            f"Blue query: ({qx:.4g}, {qy:.4g})",
            color=QUERY_COLOR,
        )
        text(
            -550, -242,
            f"Orange key: ({kx:.4g}, {ky:.4g})",
            color=KEY_COLOR,
        )

        draw_matrix(
            "Raw scores", scores,
            130, 260, pair_index, stage,
        )
        draw_matrix(
            "Scaled scores", scaled_scores,
            130, -10, pair_index, stage,
        )

        text(130, -225, "Matrix rows = queries; columns = keys.", 12)
        text(
            130, -250,
            "Token order: " + ", ".join(
                f"{i}={token[:10]}" for i, token in enumerate(tokens)
            ),
            11,
        )

        if stage == 0:
            explanation = [
                f"Q row {query_index} = ({qx:.4g}, {qy:.4g})",
                f"K row {key_index} becomes K_T column {key_index}.",
                "Multiply matching components, then add the products.",
            ]
        elif stage == 1:
            explanation = [
                "Horizontal contribution: q_x * k_x",
                f"{qx:.6g} * {kx:.6g} = {x_product:.6g}",
                "The highlighted segments represent the x components.",
            ]
        elif stage == 2:
            explanation = [
                "Vertical contribution: q_y * k_y",
                f"{qy:.6g} * {ky:.6g} = {y_product:.6g}",
                f"Previous horizontal contribution: {x_product:.6g}",
            ]
        elif stage == 3:
            explanation = [
                f"Dot product = {x_product:.6g} + {y_product:.6g}"
                f" = {x_product + y_product:.6g}",
                f"Rust score[{query_index}, {key_index}]"
                f" = {scores[query_index, key_index]:.6g}",
                "The result is a scalar, stored in one matrix cell.",
            ]
        else:
            explanation = [
                f"Scale by sqrt(2) = {divisor:.6g}",
                f"{scores[query_index, key_index]:.6g}"
                f" / {divisor:.6g}"
                f" = {scaled_scores[query_index, key_index]:.6g}",
                "Scaling changes the score; it does not move these arrows.",
            ]

        for line_index, message in enumerate(explanation):
            text(-550, -295 - line_index * 25, message, 14)

        text(
            -550, -395,
            f"Step {step + 1}/{total_steps}"
            " | Space / Right: next | Left: back"
            " | Home: restart | Esc: close",
            12,
        )

        screen.update()

    def move(amount):
        nonlocal step
        step = max(0, min(total_steps - 1, step + amount))
        draw()

    def restart():
        nonlocal step
        step = 0
        draw()

    screen.onkey(lambda: move(1), "Right")
    screen.onkey(lambda: move(1), "space")
    screen.onkey(lambda: move(-1), "Left")
    screen.onkey(restart, "Home")
    screen.onkey(screen.bye, "Escape")

    draw()
    screen.listen()
    screen.mainloop()


def _validate(tokens, Q, K, K_T, scores, scaled_scores):
    """Check dimensions and verify supplied results using rustml."""
    n = len(tokens)
    arrays = (Q, K, K_T, scores, scaled_scores)

    if not 1 <= n <= 5:
        raise ValueError("This viewer supports one to five tokens.")

    if any(array.ndim != 2 for array in arrays):
        raise ValueError("All inputs must be two-dimensional matrices.")

    if Q.shape != (n, 2) or K.shape != (n, 2):
        raise ValueError(
            "An x,y graph requires Q and K to have shape (tokens, 2)."
        )

    if K_T.shape != (2, n):
        raise ValueError("K_T must have shape (2, tokens).")

    if scores.shape != (n, n) or scaled_scores.shape != (n, n):
        raise ValueError("Score matrices must have shape (tokens, tokens).")

    if not all(np.isfinite(array).all() for array in arrays):
        raise ValueError("All values must be finite.")

    # Recalculate with Rust to verify the supplied matrices.
    expected_K_T = transpose(K)
    expected_scores = matmul(Q, expected_K_T)
    expected_scaled_scores = expected_scores / math.sqrt(Q.shape[1])

    if not np.allclose(K_T, expected_K_T):
        raise ValueError("K_T does not match rustml.transpose(K).")

    if not np.allclose(scores, expected_scores):
        raise ValueError(
            "The supplied scores do not match "
            "rustml.matmul(Q, rustml.transpose(K))."
        )

    if not np.allclose(scaled_scores, expected_scaled_scores):
        raise ValueError("The supplied scaled scores are inconsistent.")