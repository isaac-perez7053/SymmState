#!/usr/bin/env python3
import sys
import os
import numpy as np
import matplotlib

print("Matplotlib backend:", matplotlib.get_backend())
import matplotlib.pyplot as plt
from scipy.optimize import leastsq
from mpl_toolkits.mplot3d import Axes3D
from itertools import combinations
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def parse_args():
    if len(sys.argv) < 3:
        print("Usage: python script.py <data_file> <terms>")
        print("Example: python script.py data.txt x^2 y^2 x^4 y^4 x^2y^2")
        sys.exit(1)
    return sys.argv[1], sys.argv[2].split()  # Split the terms string into a list


def load_data(filename):
    data = np.loadtxt(filename)
    return data[:, 0], data[:, 1], data[:, 2]


def scale_data(x, y, z):
    x_scaled = (x - np.min(x)) / (np.max(x) - np.min(x))
    y_scaled = (y - np.min(y)) / (np.max(y) - np.min(y))
    z_scaled = (z - np.min(z)) / (np.max(z) - np.min(z))
    return x_scaled, y_scaled, z_scaled


def create_polynomial_function(terms):
    def polynomial_surface(x, y, *params):
        result = params[-1]  # Constant term
        for param, term in zip(params[:-1], terms):
            if "x" in term and "y" in term:
                if "^" in term:
                    x_part, y_part = term.split("y")
                    x_power = int(x_part.split("^")[1]) if "^" in x_part else 1
                    y_power = int(y_part.split("^")[1]) if "^" in y_part else 1
                else:
                    x_power = y_power = 1
                result += param * (x**x_power) * (y**y_power)
            elif "x" in term:
                power = int(term.split("^")[1]) if "^" in term else 1
                result += param * x**power
            elif "y" in term:
                power = int(term.split("^")[1]) if "^" in term else 1
                result += param * y**power
        return result

    return polynomial_surface


def fit_surface(x_data, y_data, z_data, func, initial_params):
    def residuals(params, x, y, z):
        z_pred = func(x, y, *params)
        return z - z_pred

    # Flatten data for optimization
    x_flat = x_data.flatten()
    y_flat = y_data.flatten()
    z_flat = z_data.flatten()

    # Least squares optimization
    optimized_params, _ = leastsq(
        residuals, initial_params, args=(x_flat, y_flat, z_flat)
    )
    return optimized_params


def create_2d_plot(x_scaled, y, y_pred, axis_label, terms, popt):
    plt.figure(figsize=(10, 6))
    plt.scatter(x_scaled, y, color="blue", label="Data points")
    plt.plot(x_scaled, y_pred, color="red", label="Fitted curve")
    plt.xlabel(axis_label)
    plt.ylabel("Z")
    plt.title(f"2D Polynomial Fit for {axis_label}-axis")
    plt.legend()

    equation = "z = " + " + ".join(
        [f"{coef:.6e}{term}" for coef, term in zip(popt[:-1], terms[:-1])]
        + [f"{popt[-1]:.6e}"]
    )
    plt.text(
        0.05,
        0.95,
        equation,
        transform=plt.gca().transAxes,
        verticalalignment="top",
        fontsize=9,
    )

    plt.savefig(f"2d_fit_{axis_label.lower()}.png")
    plt.close()


def create_interactive_plot(x, y, z, X, Y, Z, terms, popt):
    fig = make_subplots(rows=1, cols=1, specs=[[{"type": "surface"}]])

    fig.add_trace(
        go.Scatter3d(
            x=x,
            y=y,
            z=z,
            mode="markers",
            marker=dict(size=4, color="blue"),
            name="Data points",
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Surface(x=X, y=Y, z=Z, colorscale="Viridis", name="Fitted surface"),
        row=1,
        col=1,
    )

    fig.update_layout(
        title="Interactive Polynomial Surface Fit",
        scene=dict(xaxis_title="X", yaxis_title="Y", zaxis_title="Z"),
        autosize=False,
        width=900,
        height=700,
    )

    equation = print_equation(popt, terms)
    fig.add_annotation(
        xref="paper",
        yref="paper",
        x=0.0,
        y=1.05,
        text=equation,
        showarrow=False,
        font=dict(size=12),
        align="left",
    )

    fig.write_html("interactive_plot_final.html")


def print_equation(popt, terms):
    equation = "z = "
    for coef, term in zip(popt[:-1], terms[:-1]):
        if term != "1":
            if "^" in term:
                equation += f"{coef:.6e}{term} + "
            else:
                equation += f"{coef:.6e}{term} + "
    equation += f"{popt[-1]:.6e}"  # Add constant term
    return equation


def main():
    print("Current working directory:", os.getcwd())
    data_file, terms = parse_args()
    x, y, z = load_data(data_file)
    x_scaled, y_scaled, z_scaled = scale_data(x, y, z)

    # Create the polynomial function
    poly_func = create_polynomial_function(terms)

    # Initial guess for parameters (one for each term plus a constant term)
    initial_guess = [1.0] * (len(terms) + 1)

    # Fit the surface
    best_popt = fit_surface(x_scaled, y_scaled, z_scaled, poly_func, initial_guess)

    # Calculate predictions and metrics
    z_pred = poly_func(x_scaled, y_scaled, *best_popt)
    mse = np.mean((z_scaled - z_pred) ** 2)
    r2 = 1 - np.sum((z_scaled - z_pred) ** 2) / np.sum(
        (z_scaled - np.mean(z_scaled)) ** 2
    )

    print("\nFinal best fit:")
    print(f"Terms: {', '.join(terms + ['1'])}")
    print(print_equation(best_popt, terms + ["1"]))
    print(f"MSE: {mse:.6e}")
    print(f"R-squared: {r2:.6f}")

    # Create smooth grid for surface plot
    x_smooth = np.linspace(0, 1, 100)
    y_smooth = np.linspace(0, 1, 100)
    X, Y = np.meshgrid(x_smooth, y_smooth)
    Z = poly_func(X, Y, *best_popt)

    # Create interactive plot
    create_interactive_plot(
        x_scaled, y_scaled, z_scaled, X, Y, Z, terms + ["1"], best_popt
    )
    print("Interactive plot for final fit saved as 'interactive_plot_final.html'")


if __name__ == "__main__":
    main()
