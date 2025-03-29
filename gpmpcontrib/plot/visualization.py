import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import gpmp as gp

# In Matplotlib, interactive mode is a feature that allows for
# real-time updates to plots. When interactive mode is on, Matplotlib
# will automatically update the plot window after every plotting
# command without needing to call plt.show() explicitly.
# To turn interactive mode globally.

# plt.ion()

# Set interactive mode for plotting (set to True if interactive
# plotting is desired)
interactive = False


def plot_1d(xt, zt, xi, zi, zpm, zpv, zpsim=None, xnew=None, title=None):
    """
    Visualize the results of the predictions and the dataset.

    Parameters:
    xt (ndarray): Test points
    zt (ndarray): True values at test points
    xi (ndarray): Input data points
    zi (ndarray): Output values at input data points
    zpm (ndarray): Posterior mean values
    zpv (ndarray): Posterior variances
    zpsim (ndarray, optional): Conditional sample paths
    xnew (ndarray, optional): New data point being added
    title (str, optional): Title for the plot
    """
    fig = gp.misc.plotutils.Figure(isinteractive=interactive)

    # Plot zt if it is provided
    if zt is not None:
        fig.plot(xt, zt, "k", linewidth=1, linestyle=(0, (5, 5)), label="truth")

    # Plot conditional sample paths only if zpsim is provided
    if zpsim is not None:
        fig.plot(xt, zpsim[:, 0], "k", linewidth=0.5, label="conditional sample paths")
        fig.plot(xt, zpsim[:, 1:], "k", linewidth=0.5)

    # Plot data points
    fig.plotdata(xi, zi)

    # Plot GP mean and variance
    fig.plotgp(xt, zpm, zpv, colorscheme="simple")

    # Plot new evaluation point if provided
    if xnew is not None:
        fig.plot(np.repeat(xnew, 2), fig.ylim(), color="tab:gray", linewidth=3)
        if title is None:
            fig.title("New Evaluation")

    # Add title if it is provided
    if title is not None:
        fig.title(title)

    # Set labels and show plot
    fig.xylabels("$x$", "$z$")
    fig.show(grid=True, xlim=[-1.0, 1.0], legend=True, legend_fontsize=9)


def show_truth_vs_prediction(zt, zpm):
    """
    Visualize the predictions vs truth
    """
    num_outputs = zt.shape[1]
    fig, axs = plt.subplots(1, num_outputs, figsize=(6 * num_outputs, 5))

    for i in range(num_outputs):
        ax = axs[i] if num_outputs > 1 else axs
        ax.scatter(zt[:, i], zpm[:, i])
        ax.plot(
            [zt[:, i].min(), zt[:, i].max()], [zt[:, i].min(), zt[:, i].max()], "k--"
        )
        ax.set_xlabel("True Values")
        ax.set_ylabel("Predicted Values")
        ax.set_title(f"Output {i+1}")

    plt.tight_layout()
    plt.show()


def show_loo_errors(zi, zloom, zloov):
    """
    Visualize leave-one-out errors for each output dimension.
    """
    num_outputs = zi.shape[1]
    fig, axs = plt.subplots(1, num_outputs, figsize=(6 * num_outputs, 5), squeeze=False)

    for i in range(num_outputs):
        ax = axs[0, i]
        ax.errorbar(
            zi[:, i], zloom[:, i], yerr=1.96 * np.sqrt(zloov[:, i]), fmt="ko", ls="None"
        )
        ax.set_xlabel("True Values")
        ax.set_ylabel("LOO Predicted")
        ax.set_title(f"Output {i + 1} - LOO predictions with 95% CI")

        # Add identity line
        min_val = min(ax.get_xlim()[0], ax.get_ylim()[0])
        max_val = max(ax.get_xlim()[1], ax.get_ylim()[1])
        ax.plot([min_val, max_val], [min_val, max_val], "k--")

        ax.grid(True, "major", linestyle=(0, (1, 5)), linewidth=0.5)

    plt.tight_layout()
    plt.show()


def plotmatrix(data, colors=None):
    """
    Generates a matrix scatter plot from the given 2D numerical numpy array or matrix,
    optionally colorizing the scatter points based on an n x 1 ndarray.

    Parameters:
    data (2D numpy array): A matrix of numerical data.
    colors (1D numpy array, optional): An array of values to color the scatter points.
    """
    num_vars = data.shape[1]

    # Create a grid of subplots
    fig, axes = plt.subplots(nrows=num_vars, ncols=num_vars, figsize=(10, 10))

    # Initialize the scatter plot object (to be used for colorbar if needed)
    sc = None

    # Iterate over all variable pairs
    for i, j in itertools.product(range(num_vars), range(num_vars)):
        # Off-diagonal scatter plot with optional colorization
        if i != j:
            if colors is not None:
                sc = axes[i, j].scatter(
                    data[:, j], data[:, i], alpha=0.5, s=10, c=colors, cmap="viridis"
                )
            else:
                sc = axes[i, j].scatter(data[:, j], data[:, i], alpha=0.5, s=10)
        # Diagonal: plot the histograms
        else:
            axes[i, j].hist(data[:, i], bins=20, alpha=0.7)

        # Set labels on the outer edge
        if i == num_vars - 1:
            axes[i, j].set_xlabel(f"Var {j+1}")
        if j == 0:
            axes[i, j].set_ylabel(f"Var {i+1}")

    # If colorization is applied, add a colorbar outside the plots
    if colors is not None:
        # Create a new axis for the colorbar
        # [left, bottom, width, height]
        cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
        fig.colorbar(sc, cax=cbar_ax)

    # Adjust layout for spacing
    # Adjust layout to leave space for colorbar
    plt.tight_layout(rect=[0, 0, 0.9, 1])
    plt.show()


def parallel_coordinates_plot(
    x, z, p=None, show_p=False, xi=None, zi=None, ci=None, show_type=False
):
    """
    Creates an interactive parallel coordinates plot using go.Parcoords, with optional
    custom categories for the datasets and a flag to control the display of the type variable.

    Parameters:
    - x: numpy array, shape (n_samples, n_dimensions)
         The input dataset for parallel coordinates.
    - z: numpy array, shape (n_samples, n_targets)
         The target or latent data.
    - p: numpy array, shape (n_samples,), optional
         The latent variable used for colorization (if provided).
    - show_p: bool, optional (default: False)
         If True, p will be displayed as one of the dimensions in the plot.
    - xi: numpy array, shape (n_samples, n_dimensions), optional
         Additional dataset to be displayed as type 1 (optional).
    - zi: numpy array, shape (n_samples, n_targets), optional
         Additional target data for type 1 (optional).
    - ci: numpy array, shape (n_samples,), optional
         Custom type values for the xi, zi dataset (overrides the default categories).
    - show_type: bool, optional (default: False)
         If True, the type variable will be shown as the first dimension.

    Returns:
    - Plotly figure object (go.Figure)
    """
    import plotly as px
    import plotly.graph_objects as go

    # Ensure x and z are 2D arrays, p is 1D array (if provided)
    assert x.ndim == 2, "x must be a 2D numpy array"
    assert z.ndim == 2, "z must be a 2D numpy array"
    if p is not None:
        assert p.ndim == 1, "p must be a 1D numpy array if provided"
        assert p.shape[0] == x.shape[0], "p must have the same number of samples as x"

    # Ensure that xi and zi are provided together and are 2D arrays if present
    if xi is not None or zi is not None:
        assert xi is not None and zi is not None, "Both xi and zi must be provided"
        assert xi.ndim == 2 and zi.ndim == 2, "xi and zi must be 2D numpy arrays"
        assert (
            xi.shape[0] == zi.shape[0]
        ), "xi and zi must have the same number of samples"
        assert (
            xi.shape[1] == x.shape[1]
        ), "xi must have the same number of dimensions as x"
        assert zi.shape[1] == z.shape[1], "zi must have the same number of targets as z"

    # Create a DataFrame from x and z
    df_x = pd.DataFrame(x, columns=[f"var_{i}" for i in range(x.shape[1])])
    df_z = pd.DataFrame(z, columns=[f"z_{i}" for i in range(z.shape[1])])

    # Concatenate x and z into one DataFrame
    df = pd.concat([df_x, df_z], axis=1)

    # Add a default type column with value 0 for the main dataset
    df["type"] = 0

    # If xi and zi are provided, concatenate them and assign type 1
    if xi is not None and zi is not None:
        df_xi = pd.DataFrame(xi, columns=[f"var_{i}" for i in range(xi.shape[1])])
        df_zi = pd.DataFrame(zi, columns=[f"z_{i}" for i in range(zi.shape[1])])
        df_highlight = pd.concat([df_xi, df_zi], axis=1)

        if ci is not None:
            # If custom categories are provided for xi and zi, use ci
            assert (
                ci.shape[0] == xi.shape[0]
            ), "ci must have the same number of samples as xi and zi"
            df_highlight["type"] = ci
        else:
            # Assign default type 1 to the xi, zi dataset
            df_highlight["type"] = 1

        # Concatenate the main and highlighted datasets
        df = pd.concat([df, df_highlight], ignore_index=True)

    # If p is provided, add it as a column for colorization
    if p is not None:
        # Concatenate p and create a corresponding p for xi, if provided
        if xi is not None and zi is not None:
            # Set p as NaN for xi/zi (optional)
            p_highlight = np.nan * np.ones(xi.shape[0])
            df["p"] = np.concatenate([p, p_highlight])
        else:
            df["p"] = p

    # Set up the dimensions list for the plot
    dimensions = []

    # Add the type variable as the first axis if show_type is True
    if show_type and xi is not None and zi is not None:
        dimensions.append(
            dict(
                range=[df["type"].min(), df["type"].max()],
                tickvals=np.unique(df["type"]),
                ticktext=[f"Type {int(val)}" for val in np.unique(df["type"])],
                label="Type",
                values=df["type"],
            )
        )

    # Add x variables to the dimensions list
    for i in range(x.shape[1]):
        dimensions.append(
            dict(
                range=[np.min(df[f"var_{i}"]), np.max(df[f"var_{i}"])],
                label=f"var_{i}",
                values=df[f"var_{i}"],
            )
        )

    # Add z variables to the dimensions list
    for j in range(z.shape[1]):
        dimensions.append(
            dict(
                range=[np.min(df[f"z_{j}"]), np.max(df[f"z_{j}"])],
                label=f"z_{j}",
                values=df[f"z_{j}"],
            )
        )

    # Optionally add p as a dimension, based on show_p
    if p is not None and show_p:
        dimensions.append(
            dict(
                range=[np.nanmin(df["p"]), np.nanmax(df["p"])],
                label="p (Latent Variable)",
                values=df["p"],
            )
        )

    # Create the main parallel coordinates plot
    fig = go.Figure()

    # Add the dataset with colorization
    fig.add_trace(
        go.Parcoords(
            line=dict(
                color=df["p"] if p is not None else df_z.iloc[:, 0],
                # 'Sunsetdark',  # Color scale can be customized
                colorscale=px.colors.diverging.Tealrose,
                showscale=True,  # Show the color bar
            ),
            dimensions=dimensions,
        )
    )

    # Customize the layout (optional)
    fig.update_layout(
        title="Interactive Parallel Coordinates Plot with Categories",
        plot_bgcolor="white",
        paper_bgcolor="white",
    )

    # Display the plot
    fig.show()
