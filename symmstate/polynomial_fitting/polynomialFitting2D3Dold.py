#!/usr/bin/env python3
import sys
import os
import numpy as np
import matplotlib

print("Matplotlib backend:", matplotlib.get_backend())
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
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
    def polynomial(xy, *params):
        x, y = xy
        result = np.full_like(x, params[-1])  # Constant term
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

    return polynomial


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


def fit_and_evaluate_2d(x, y, func, p0):
    popt, _ = curve_fit(func, x, y, p0=p0, maxfev=10000)
    y_pred = func(x, *popt)
    mse = np.mean((y - y_pred) ** 2)
    r2 = 1 - np.sum((y - y_pred) ** 2) / np.sum((y - np.mean(y)) ** 2)
    return popt, mse, r2, y_pred


def fit_and_evaluate_3d(x, y, z, func, p0):
    popt, _ = curve_fit(func, (x, y), z, p0=p0, maxfev=10000)
    z_pred = func((x, y), *popt)
    mse = np.mean((z - z_pred) ** 2)
    r2 = 1 - np.sum((z - z_pred) ** 2) / np.sum((z - np.mean(z)) ** 2)
    return popt, mse, r2, z_pred


def create_2d_polynomial_function(powers):
    def polynomial(x, *params):
        return (
            sum(param * x**power for param, power in zip(params[:-1], powers))
            + params[-1]
        )

    return polynomial


def stepwise_fitting_2d(x, z, terms, axis_label):
    powers = []
    for term in terms:
        if axis_label.lower() in term:
            if "^" in term:
                powers.append(int(term.split("^")[1]))
            else:
                powers.append(1)  # For terms like 'x' or 'y', assume power is 1
    powers.sort()  # Sort powers in ascending order

    best_terms = []
    best_popt = []

    for step, power in enumerate(powers, 1):
        current_powers = powers[:step]
        poly_func = create_2d_polynomial_function(current_powers)

        if step == 1:
            initial_guess = [1.0] * (step + 1)  # +1 for the constant term
        else:
            initial_guess = list(previous_popt) + [0.0]  # Add a new parameter

        popt, mse, r2, y_pred = fit_and_evaluate_2d(x, z, poly_func, initial_guess)

        print(f"2D Step {step}: Fitting terms up to {axis_label}^{power}")
        print(f"MSE: {mse:.6e}")
        print(f"R-squared: {r2:.6f}\n")

        current_terms = [
            f"{axis_label}^{p}" if p != 1 else axis_label for p in current_powers
        ] + ["1"]
        best_terms = current_terms
        best_popt = popt
        previous_popt = popt

        # Create and save 2D plot
        create_2d_plot(x, z, y_pred, axis_label, current_terms, popt)

    return best_terms, best_popt


def stepwise_fitting_3d(x, y, z, terms, uncoupled_terms, uncoupled_popt):
    coupled_terms = [term for term in terms if "x" in term and "y" in term]
    all_terms = list(
        dict.fromkeys(uncoupled_terms + coupled_terms + ["1"])
    )  # Remove duplicates and add constant term

    def poly_func(xy, *params):
        x, y = xy
        result = params[-1]  # Constant term
        for term, param in zip(all_terms[:-1], params[:-1]):
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

    # Rest of the function remains the same...

    # Create initial guess
    initial_guess = []
    for term in all_terms[:-1]:  # Exclude the constant term
        if term in uncoupled_terms:
            initial_guess.append(uncoupled_popt[uncoupled_terms.index(term)])
        else:
            initial_guess.append(1.0)
    initial_guess.append(0.0)  # Add initial guess for constant term

    # Try fitting with different methods and parameters
    try:
        popt, pcov = curve_fit(
            poly_func,
            (x, y),
            z,
            p0=initial_guess,
            maxfev=100000,
            method="trf",
            loss="soft_l1",
        )
    except RuntimeError:
        try:
            popt, pcov = curve_fit(
                poly_func, (x, y), z, p0=initial_guess, maxfev=100000, method="lm"
            )
        except RuntimeError:
            popt, pcov = curve_fit(
                poly_func, (x, y), z, p0=initial_guess, maxfev=100000
            )

    # Calculate predictions and metrics
    z_pred = poly_func((x, y), *popt)
    mse = np.mean((z - z_pred) ** 2)
    r2 = 1 - np.sum((z - z_pred) ** 2) / np.sum((z - np.mean(z)) ** 2)

    print(f"3D Fitting: All terms")
    print(print_equation(popt, all_terms))
    print(f"MSE: {mse:.6e}")
    print(f"R-squared: {r2:.6f}\n")

    return all_terms, popt, mse, r2


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


def filter_constant_axis(x, y, z, axis, tolerance=1e-6):
    if axis == "x":
        mask = np.isclose(y, np.min(y), atol=tolerance)
    else:  # axis == 'y'
        mask = np.isclose(x, np.min(x), atol=tolerance)

    return x[mask], y[mask], z[mask]


def main():
    print("Current working directory:", os.getcwd())
    data_file, terms = parse_args()
    x, y, z = load_data(data_file)
    x_scaled, y_scaled, z_scaled = scale_data(x, y, z)

    x_terms = [term for term in terms if "x" in term and "y" not in term]
    y_terms = [term for term in terms if "y" in term and "x" not in term]
    coupled_terms = [term for term in terms if "x" in term and "y" in term]
    print(f"X terms: {x_terms}")
    print(f"Y terms: {y_terms}")
    print(f"Coupled terms: {coupled_terms}")

    # Filter data for 2D fitting
    x_filtered, _, z_filtered_x = filter_constant_axis(
        x_scaled, y_scaled, z_scaled, "x"
    )
    _, y_filtered, z_filtered_y = filter_constant_axis(
        x_scaled, y_scaled, z_scaled, "y"
    )

    # 2D fitting for uncoupled terms
    uncoupled_terms_x, uncoupled_popt_x = stepwise_fitting_2d(
        x_filtered, z_filtered_x, x_terms, "X"
    )
    uncoupled_terms_y, uncoupled_popt_y = stepwise_fitting_2d(
        y_filtered, z_filtered_y, y_terms, "Y"
    )

    uncoupled_terms = (
        uncoupled_terms_x[:-1] + uncoupled_terms_y[:-1]
    )  # Exclude constant terms
    uncoupled_popt = list(uncoupled_popt_x[:-1]) + list(
        uncoupled_popt_y[:-1]
    )  # Exclude constant terms

    all_terms = uncoupled_terms + coupled_terms
    best_terms, best_popt, best_mse, best_r2 = stepwise_fitting_3d(
        x_scaled, y_scaled, z_scaled, all_terms, uncoupled_terms, uncoupled_popt
    )

    print("\nFinal best fit:")
    print(f"Terms: {', '.join(best_terms)}")
    print(print_equation(best_popt, best_terms))
    print(f"MSE: {best_mse:.6e}")
    print(f"R-squared: {best_r2:.6f}")

    x_smooth = np.linspace(0, 1, 100)
    y_smooth = np.linspace(0, 1, 100)
    X, Y = np.meshgrid(x_smooth, y_smooth)
    poly_func = create_polynomial_function(best_terms)
    Z = poly_func((X, Y), *best_popt)

    # Ensure Z is a 2D numpy array
    if Z.ndim == 1:
        Z = Z.reshape(X.shape)

    create_interactive_plot(
        x_scaled, y_scaled, z_scaled, X, Y, Z, best_terms, best_popt
    )
    print("Interactive plot for final fit saved as 'interactive_plot_final.html'")
    print(
        "2D plots for X-axis and Y-axis fits should be saved in the current directory"
    )


if __name__ == "__main__":
    main()
