"""
Helper functions for generating formatted reports
"""

import base64
import io

import matplotlib.figure as mfigure


def base64ify(fig: mfigure.Figure) -> str:
    """
    Saves a Matplotlib figure as date url.

    Data urls let you directly embed images in markdown (and html) files.

    For example ::

       fragment = f'![caption][{base64ify(fig)}]'

    will generate a markdown image tag that has the image directly
    embedded as a base64 string.

    This will make your markdown documents large and hard to edit by hand, but
    the trade off is that they are fully self-contained.

    Parameters
    ----------
    fig: matplotlib.figure.Figure

    Returns
    -------
    str
    """
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png")
    buffer.seek(0)

    return f"data:image/png;base64,{base64.b64encode(buffer.read()).decode()}"
