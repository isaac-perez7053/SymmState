#!/usr/bin/env python3
import sys
import re
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from mpl_toolkits.mplot3d import Axes3D
from itertools import combinations
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Tuple, List, Callable, Dict

# --------------------------
# Data Handling Classes
# --------------------------


class Scaler:
    """Handles scaling and unscaling of data"""

    def __init__(self, data: np.ndarray):
        self.min = np.min(data)
        self.max = np.max(data)
        self.range = self.max - self.min + 1e-12  # Prevent division by zero

    def scale(self, data: np.ndarray) -> np.ndarray:
        return (data - self.min) / self.range

    def unscale(self, data: np.ndarray) -> np.ndarray:
        return data * self.range + self.min


# --------------------------
# Core Fitting Functions
# --------------------------


def parse_term(term: str) -> Tuple[int, int]:
    """Parse polynomial term into exponents using regex"""
    term = term.lower().replace(" ", "")
    if term == "const":
        return (0, 0)

    x_match = re.search(r"x\^?(\d*)", term)
    y_match = re.search(r"y\^?(\d*)", term)

    x_exp = (
        int(x_match.group(1))
        if x_match and x_match.group(1)
        else 1 if "x" in term else 0
    )
    y_exp = (
        int(y_match.group(1))
        if y_match and y_match.group(1)
        else 1 if "y" in term else 0
    )

    return (x_exp, y_exp)


def create_polynomial_function(terms: List[str]) -> Callable:
    """Create polynomial function with proper term handling"""
    term_exponents = [parse_term(term) for term in terms]

    def polynomial(xy: Tuple[np.ndarray, np.ndarray], *params: float) -> np.ndarray:
        x, y = xy
        result = np.zeros_like(x)
        for param, (x_exp, y_exp) in zip(params, term_exponents):
            result += param * (x**x_exp) * (y**y_exp)
        return result

    return polynomial


# --------------------------
# Optimization & Validation
# --------------------------


def validate_input_terms(terms: List[str]) -> None:
    """Validate input term format using regex"""
    valid_pattern = re.compile(r"^(x(\^\d+)?y?(\^\d+)?|y(\^\d+)?|const)$")
    for term in terms:
        if not valid_pattern.match(term):
            raise ValueError(
                f"Invalid term format: {term}. Valid examples: 'x^2', 'xy^3', 'const'"
            )


def get_initial_guess(
    x: np.ndarray, y: np.ndarray, z: np.ndarray, terms: List[str]
) -> np.ndarray:
    """Calculate initial parameters using linear least squares"""
    X = np.column_stack(
        [(x**xe) * (y**ye) for xe, ye in [parse_term(t) for t in terms]]
    )
    if "const" in terms:
        X = np.hstack([X, np.ones((len(x), 1))])
    return np.linalg.lstsq(X, z, rcond=None)[0]


# --------------------------
# Stepwise Fitting Algorithm
# --------------------------


def forward_selection(
    x: np.ndarray, y: np.ndarray, z: np.ndarray, terms: List[str], max_terms: int = None
) -> List[str]:
    """Efficient forward selection of polynomial terms"""
    selected = []
    remaining = list(terms)
    best_r2 = -np.inf

    while remaining and (max_terms is None or len(selected) < max_terms):
        best_term = None
        current_r2 = best_r2

        for term in remaining:
            current_terms = selected + [term]
            poly_func = create_polynomial_function(current_terms)

            try:
                initial_guess = get_initial_guess(x, y, z, current_terms)
                popt, _, r2, _ = fit_and_evaluate(x, y, z, poly_func, initial_guess)

                if r2 > current_r2:
                    current_r2 = r2
                    best_term = term
            except np.linalg.LinAlgError:
                continue

        if best_term and current_r2 > best_r2:
            selected.append(best_term)
            remaining.remove(best_term)
            best_r2 = current_r2
        else:
            break

    return selected


# --------------------------
# Visualization Functions
# --------------------------


def generate_latex_equation(popt: np.ndarray, terms: List[str]) -> str:
    """Generate LaTeX formatted equation for plots"""
    equation = "z = "
    for coef, term in zip(popt, terms):
        if term == "const":
            equation += f"{coef:.2f} + "
        else:
            parts = []
            if "x" in term:
                x_part = "x" + (
                    f"^{{{term.count('x')}}}" if term.count("x") > 1 else ""
                )
                parts.append(x_part)
            if "y" in term:
                y_part = "y" + (
                    f"^{{{term.count('y')}}}" if term.count("y") > 1 else ""
                )
                parts.append(y_part)
            equation += f"{coef:.2f}" + "".join(parts) + " + "
    return (
        equation.rstrip(" + ").replace("^1", "").replace("x^0", "").replace("y^0", "")
    )


# --------------------------
# Main Workflow Functions
# --------------------------


def parse_args() -> Tuple[str, List[str]]:
    """Parse command line arguments with improved help"""
    if len(sys.argv) < 3 or "-h" in sys.argv:
        print(
            """Polynomial Surface Fitter
Usage: python fit_surface.py <data.txt> [terms...]

Examples:
  Basic quadratic: python fit_surface.py data.txt x^2 xy y^2
  Full cubic: python fit_surface.py data.txt x^3 x^2y xy^2 y^3 x^2 xy y^2 x y const

Data file format:
  Three columns containing x, y, z values
  Example row: 1.2 3.4 5.6"""
        )
        sys.exit(1)
    return sys.argv[1], sys.argv[2:]


def load_data(filename: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load and validate data file"""
    try:
        data = np.loadtxt(filename)
        if data.shape[1] != 3:
            raise ValueError("Data file must have exactly 3 columns")
        return data[:, 0], data[:, 1], data[:, 2]
    except Exception as e:
        print(f"Error loading data file: {str(e)}")
        sys.exit(1)


def fit_and_evaluate(
    x: np.ndarray, y: np.ndarray, z: np.ndarray, func: Callable, p0: List[float]
) -> Tuple[np.ndarray, float, float, np.ndarray]:
    """Perform curve fitting with error handling"""
    try:
        popt, pcov = curve_fit(func, (x, y), z, p0=p0, maxfev=10000)
        z_pred = func((x, y), *popt)
        mse = np.mean((z - z_pred) ** 2)
        r2 = 1 - np.sum((z - z_pred) ** 2) / np.sum((z - np.mean(z)) ** 2)
        return popt, mse, r2, z_pred
    except RuntimeError as e:
        print(f"Fitting failed: {str(e)}")
        return np.zeros_like(p0), np.inf, -np.inf, np.zeros_like(z)


def plot_results(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    func: Callable,
    popt: np.ndarray,
    terms: List[str],
    step: int,
    filename: str,
) -> None:
    """Generate 3D visualization plots"""
    # Static matplotlib plot
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection="3d")

    x_smooth = np.linspace(0, 1, 100)
    y_smooth = np.linspace(0, 1, 100)
    X, Y = np.meshgrid(x_smooth, y_smooth)
    Z = func((X, Y), *popt)

    ax.scatter(x, y, z, c="blue", s=20, depthshade=False, label="Data Points")
    ax.plot_surface(X, Y, Z, cmap="viridis", alpha=0.7, label="Fitted Surface")

    ax.set_xlabel("X (scaled)", fontsize=10)
    ax.set_ylabel("Y (scaled)", fontsize=10)
    ax.set_zlabel("Z (scaled)", fontsize=10)
    ax.set_title(
        f"Surface Fit - Step {step}\n{generate_latex_equation(popt, terms)}",
        fontsize=12,
    )

    plt.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close()

    # Interactive plotly visualization
    fig = go.Figure()
    fig.add_trace(
        go.Scatter3d(
            x=x,
            y=y,
            z=z,
            mode="markers",
            marker=dict(size=4, color="blue", opacity=0.8),
            name="Data Points",
        )
    )
    fig.add_trace(
        go.Surface(
            x=X, y=Y, z=Z, colorscale="Viridis", opacity=0.7, name="Fitted Surface"
        )
    )

    fig.update_layout(
        title=f"Interactive Surface Fit - Step {step}",
        scene=dict(
            xaxis_title="X (scaled)",
            yaxis_title="Y (scaled)",
            zaxis_title="Z (scaled)",
            camera=dict(eye=dict(x=1.5, y=1.5, z=0.8)),
        ),
        margin=dict(l=0, r=0, b=0, t=40),
        annotations=[
            dict(
                x=0.05,
                y=0.95,
                xref="paper",
                yref="paper",
                text=f"<b>Equation:</b><br>{generate_latex_equation(popt, terms)}",
                showarrow=False,
                font=dict(size=12),
            )
        ],
    )

    fig.write_html(f"interactive_step_{step}.html")


def main():
    # Load and preprocess data
    data_file, terms = parse_args()
    validate_input_terms(terms)

    x, y, z = load_data(data_file)
    x_scaler = Scaler(x)
    y_scaler = Scaler(y)
    z_scaler = Scaler(z)

    x_scaled = x_scaler.scale(x)
    y_scaled = y_scaler.scale(y)
    z_scaled = z_scaler.scale(z)

    # Perform optimized stepwise fitting
    selected_terms = forward_selection(x_scaled, y_scaled, z_scaled, terms)

    # Final fit with best terms
    final_func = create_polynomial_function(selected_terms)
    initial_guess = get_initial_guess(x_scaled, y_scaled, z_scaled, selected_terms)
    popt, mse, r2, _ = fit_and_evaluate(
        x_scaled, y_scaled, z_scaled, final_func, initial_guess
    )

    # Generate final outputs
    print("\nFinal Best Fit:")
    print(f"Selected Terms: {', '.join(selected_terms)}")
    print(f"Equation: {generate_latex_equation(popt, selected_terms)}")
    print(f"MSE (scaled): {mse:.4e}")
    print(f"R² (scaled): {r2:.4f}")

    # Create final visualization
    plot_results(
        x_scaled,
        y_scaled,
        z_scaled,
        final_func,
        popt,
        selected_terms,
        "Final",
        "final_fit.png",
    )


if __name__ == "__main__":
    main()
