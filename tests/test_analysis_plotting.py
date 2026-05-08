import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from bacteria_analysis.plotting import create_rdm_panel_figure, render_prepared_rdm_panels


def test_public_rdm_panel_helpers_render_prepared_frames():
    matrix = pd.DataFrame(
        [
            [0.0, 1.0, 2.0],
            [1.0, 0.0, 3.0],
            [2.0, 3.0, 0.0],
        ],
        index=["a", "b", "c"],
        columns=["a", "b", "c"],
    )

    figure, axes, colorbar_axes = create_rdm_panel_figure(nrows=1, figsize=(6.0, 3.0))
    render_prepared_rdm_panels(
        figure,
        axes,
        colorbar_axes,
        [
            (0, 0, matrix, "Left", "missing"),
            (0, 1, matrix, "Right", "missing"),
        ],
    )

    assert isinstance(figure, Figure)
    assert axes[0, 0].get_title() == "Left"
    assert axes[0, 1].get_title() == "Right"
    assert colorbar_axes[0, 0].get_visible()
    assert colorbar_axes[0, 1].get_visible()
    plt.close(figure)
