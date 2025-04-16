import matplotlib.pyplot as plt
import numpy as np
import ast

# Add these label mappings at the top of your code
TENSOR_ROW_LABELS = ["xx", "yy", "zz", "yz", "xz", "xy", "zy", "zx", "yx"]
TENSOR_COL_LABELS = ["xx", "yy", "zz", "yz", "xz", "xy"]


def load_flexo_data(filename):
    """Load and parse all data from the results file."""

    def parse_tensors(content):
        """Helper function to parse tensor data with multi-line support"""
        tensors = []
        current_tensor = []
        current_row = []
        in_tensor_section = False
        for line in content.split("\n"):
            stripped = line.strip()
            if "List of Flexo Electric Tensors:" in line:
                in_tensor_section = True
                continue

            if in_tensor_section:
                if "[" in line or current_row:
                    cleaned = line.replace("[", "").replace("]", "").strip()
                    if cleaned:
                        try:
                            current_row.extend(list(map(float, cleaned.split())))
                        except:
                            continue

                    if len(current_row) == 6:
                        current_tensor.append(current_row)
                        current_row = []

                        if len(current_tensor) == 9:
                            tensors.append(np.array(current_tensor))
                            current_tensor = []
        return tensors

    with open(filename, "r") as file:
        content = file.read()

        # Parse basic data lists
        amplitudes = ast.literal_eval(
            content.split("List of Amplitudes:")[1].split("\n")[0].strip()
        )
        energies = ast.literal_eval(
            content.split("List of Energies:")[1].split("\n")[0].strip()
        )
        flexo_amps = ast.literal_eval(
            content.split("List of Flexo Amplitudes:")[1].split("\n")[0].strip()
        )

        # Parse tensors
        flexo_tensors = parse_tensors(content)

    return amplitudes, energies, flexo_amps, flexo_tensors


def plot_energy(amplitudes, energies, ax=None):
    """Plot energy vs amplitude curve."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(amplitudes, energies, "b.-", markersize=8, linewidth=1.5)
    ax.set(
        xlabel="Amplitude (bohr)",
        ylabel="Energy (Ha)",
        title="Energy vs Displacement Amplitude",
    )
    ax.grid(True, alpha=0.3)
    return ax


def get_component_label(component_index):
    """Get proper physical component label for a flattened tensor index"""
    row_idx = component_index // 6
    col_idx = component_index % 6
    return f"μ_{TENSOR_ROW_LABELS[row_idx]}{TENSOR_COL_LABELS[col_idx]}"


# Modified plotting functions
def plot_flexo_components(flexo_amps, flexo_tensors, ax=None):
    """Plot all flexoelectric components with proper labels"""
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 6))

    for comp_idx in range(54):
        components = [tensor.flatten()[comp_idx] for tensor in flexo_tensors]
        ax.plot(
            flexo_amps,
            components,
            alpha=0.5,
            linewidth=0.8,
            label=get_component_label(comp_idx),
        )

    ax.set(
        xlabel="Amplitude (bohr)",
        ylabel="Component Value (nC/m)",
        title="Flexoelectric Tensor Components",
    )
    ax.grid(True, alpha=0.2)
    ax.legend(ncol=4, fontsize=6, bbox_to_anchor=(1.02, 1), loc="upper left")
    return ax


def plot_flexo_grid(flexo_amps, flexo_tensors, figsize=(30, 25)):
    """Create 9x6 grid with proper physical labels"""
    fig, axs = plt.subplots(9, 6, figsize=figsize, sharex=True, sharey="row")
    plt.subplots_adjust(hspace=0.4, wspace=0.3)

    for comp_idx in range(54):
        row_idx = comp_idx // 6
        col_idx = comp_idx % 6
        ax = axs[row_idx, col_idx]
        components = [tensor.flatten()[comp_idx] for tensor in flexo_tensors]

        ax.plot(flexo_amps, components, "b.-", markersize=4, linewidth=1)
        ax.grid(True, alpha=0.3)
        ax.set_title(get_component_label(comp_idx), fontsize=10, pad=3)

        if row_idx == 8:
            ax.set_xlabel("Amplitude (bohr)", fontsize=8)
        if col_idx == 0:
            ax.set_ylabel("nC/m", fontsize=8)

    fig.text(0.5, 0.08, "Amplitude (bohr)", ha="center", va="center", fontsize=12)
    fig.text(
        0.08,
        0.5,
        "Component Value (nC/m)",
        ha="center",
        va="center",
        rotation="vertical",
        fontsize=12,
    )
    fig.suptitle("Flexoelectric Tensor Component Analysis", y=0.98, fontsize=14)
    return fig


def plot_varying_components(
    flexo_amps, flexo_tensors, threshold="auto", figsize=(15, 10)
):
    """Plot varying components with proper physical labels"""
    variations = np.std([t.flatten() for t in flexo_tensors], axis=0)

    if threshold == "auto":
        threshold = 0.01 * np.max(variations)

    varying_mask = variations > threshold
    varying_indices = np.where(varying_mask)[0]
    n_varying = len(varying_indices)

    if n_varying == 0:
        raise ValueError("No components vary above threshold")

    max_cols = 4
    n_cols = min(max_cols, n_varying)
    n_rows = int(np.ceil(n_varying / n_cols))

    fig, axs = plt.subplots(n_rows, n_cols, figsize=figsize)
    axs = np.array(axs).flatten()

    for idx, comp_idx in enumerate(varying_indices):
        ax = axs[idx]
        components = [t.flatten()[comp_idx] for t in flexo_tensors]

        ax.plot(flexo_amps, components, "b.-", markersize=6, linewidth=1)
        ax.set_title(
            f"{get_component_label(comp_idx)}\n(σ={variations[comp_idx]:.2e})",
            fontsize=10,
        )
        ax.grid(True, alpha=0.3)
        ax.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2))

        if idx < (n_varying - n_cols):
            ax.set_xticklabels([])

    for idx in range(n_varying, len(axs)):
        axs[idx].axis("off")

    fig.text(0.5, 0.01, "Amplitude (bohr)", ha="center", va="center", fontsize=12)
    fig.text(
        0.08,
        0.5,
        "Component Value (nC/m)",
        ha="center",
        va="center",
        rotation="vertical",
        fontsize=12,
    )
    fig.suptitle(f"Varying Components (σ > {threshold:.1e} nC/m)", y=0.97, fontsize=14)

    plt.tight_layout(pad=2.0)
    return fig


# Usage example
if __name__ == "__main__":
    # Load data
    filename = "../../results2025RitzLab/catio3_Pm3m/GM4-/catio3_GM4-_9.txt"
    amps, energies, flexo_amps, flexo_tensors = load_flexo_data(filename)

    # Create individual plots
    fig1 = plt.figure(figsize=(12, 6))
    # plot_combined_analysis(amps, energies, flexo_amps, flexo_tensors)
    plot_varying_components(flexo_amps, flexo_tensors)

    fig2 = plt.figure(figsize=(14, 8))
    plot_flexo_components(flexo_amps, flexo_tensors)

    # Create grid plot
    grid_fig = plot_flexo_grid(flexo_amps, flexo_tensors)

    plt.show()
