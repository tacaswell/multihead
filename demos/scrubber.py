"""
Interactive image scrubber UI for HRD detector data.

This module provides a matplotlib-based interactive interface for exploring
raw detector data with frame selection capabilities.
"""

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from matplotlib.widgets import Button, RadioButtons, SpanSelector

from multihead.config import DetectorROIs
from multihead.file_io import HRDRawBase


class ImageScrubber:
    """
    Interactive image scrubber for HRD detector data.

    Provides a two-panel interface with:
    - Top: 2D image showing sum of selected frame range for current detector
    - Bottom: 1D line plot of ROI sums vs arm tth with span selector
    - Radio buttons to switch between detectors
    - Next/Back buttons to move the frame selection
    """

    def __init__(self, raw: HRDRawBase, detector_rois: DetectorROIs | None = None):
        """
        Initialize the image scrubber.

        Parameters
        ----------
        raw : HRDRawBase
            The raw data object containing detector images and metadata
        detector_rois : DetectorROIs, optional
            ROI definitions for each detector. If None, full detector area is used.
        """
        self.raw = raw
        self.detector_rois = detector_rois

        # Get available detectors from the raw data mapping
        self.detector_numbers = sorted(self.raw._detector_map.keys())
        self.current_detector = self.detector_numbers[0]

        # Get arm tth data and frame information
        self.arm_tth = self.raw.get_arm_tth()
        self.n_frames = len(self.arm_tth)

        # Initialize frame selection (start with first 10% of frames)
        initial_width = max(1, self.n_frames // 10)
        self.frame_start: int = 0
        self.frame_end: int = initial_width

        # Cache detector data to avoid repeated loading
        self._detector_cache = {}
        self._roi_cache = {}

        # Create the UI
        self._setup_figure()
        self._setup_widgets()

        # Initialize matplotlib objects that will be updated
        self._init_plot_objects()
        self._update_display()

    def _setup_figure(self):
        """Setup the matplotlib figure and axes."""
        self.fig = plt.figure(figsize=(12, 10))
        self.fig.suptitle("HRD Image Scrubber", fontsize=14)

        # Create main axes
        gs = self.fig.add_gridspec(
            3, 4, height_ratios=[2, 1, 0.3], width_ratios=[3, 3, 1, 0.5]
        )

        # Top: 2D image
        self.image_ax = self.fig.add_subplot(gs[0, :2])
        self.image_ax.set_title("Detector Image (Frame Sum)")
        self.image_ax.set_xlabel("Column")
        self.image_ax.set_ylabel("Row")

        # Bottom: 1D line plot
        self.line_ax = self.fig.add_subplot(gs[1, :2])
        self.line_ax.set_title("ROI Sum vs Arm 2θ")
        self.line_ax.set_xlabel("Arm 2θ (degrees)")
        self.line_ax.set_ylabel("ROI Sum")

        # Secondary x-axis for frame numbers
        self.frame_ax = self.line_ax.twiny()
        self.frame_ax.set_xlabel("Frame Number")

        # Radio buttons for detector selection
        self.radio_ax = self.fig.add_subplot(gs[0, 2])

        # Navigation buttons
        self.nav_ax = self.fig.add_subplot(gs[2, :2])

        plt.tight_layout()

    def _setup_widgets(self):
        """Setup interactive widgets."""
        # Radio buttons for detector selection
        detector_labels = [f"Det {n}" for n in self.detector_numbers]
        self.radio = RadioButtons(self.radio_ax, detector_labels)
        self.radio.on_clicked(self._on_detector_change)

        # Span selector for frame range selection
        self.span_selector = SpanSelector(
            self.line_ax,
            self._on_span_select,
            direction="horizontal",
            interactive=True,
            useblit=True,
            props=dict(alpha=0.3, facecolor="red"),
            drag_from_anywhere=True,
        )
        self.span_selector.extents = (
            self.arm_tth[self.frame_start],
            self.arm_tth[self.frame_end],
        )

        # Navigation buttons
        self.nav_ax.set_xlim(0, 1)
        self.nav_ax.set_ylim(0, 1)
        self.nav_ax.axis("off")

        # Back button
        back_ax = plt.axes((0.1, 0.05, 0.1, 0.04))
        self.back_button = Button(back_ax, "Back")
        self.back_button.on_clicked(self._on_back)

        # Next button
        next_ax = plt.axes((0.22, 0.05, 0.1, 0.04))
        self.next_button = Button(next_ax, "Next")
        self.next_button.on_clicked(self._on_next)

        # Width info text
        self.width_text = self.nav_ax.text(
            0.5, 0.5, "", ha="center", va="center", transform=self.nav_ax.transAxes
        )

    def _init_plot_objects(self):
        """Initialize matplotlib objects that will be updated during interaction."""
        # Initialize image display with dummy data
        dummy_image = np.zeros((256, 256))
        self.image_obj = self.image_ax.imshow(
            dummy_image, origin="lower", aspect="auto"
        )
        self.image_ax.set_xlabel("Column")
        self.image_ax.set_ylabel("Row")

        # Initialize colorbar
        self._colorbar = self.fig.colorbar(self.image_obj, ax=self.image_ax, shrink=0.6)

        # Initialize ROI rectangle (will be shown/hidden as needed)
        from matplotlib.patches import Rectangle

        self.roi_rect = Rectangle(
            (0, 0), 1, 1, linewidth=2, edgecolor="red", facecolor="none", visible=False
        )
        self.image_ax.add_patch(self.roi_rect)

        # Initialize line plot with dummy data
        dummy_roi_sums = np.zeros_like(self.arm_tth)
        (self.line_obj,) = self.line_ax.plot(
            self.arm_tth, dummy_roi_sums, "b-", alpha=0.7
        )
        self.line_ax.set_xlabel("Arm 2θ (degrees)")
        self.line_ax.set_ylabel("ROI Sum")
        self.line_ax.grid(True, alpha=0.3)

        # Setup secondary axis
        self.frame_ax.set_xlabel("Frame Number")

    def _get_detector_data(self, detector_num: int) -> npt.NDArray[np.uint16]:
        """Get detector data with caching."""
        if detector_num not in self._detector_cache:
            self._detector_cache[detector_num] = self.raw.get_detector(detector_num)
        return self._detector_cache[detector_num]

    def _get_roi_sum(self, detector_num: int) -> npt.NDArray:
        """Get ROI sum for a detector with caching."""
        if detector_num not in self._roi_cache:
            detector_data = self._get_detector_data(detector_num)

            if self.detector_rois and detector_num in self.detector_rois.rois:
                roi = self.detector_rois.rois[detector_num]
                rslc, cslc = roi.to_slices()
                roi_data = detector_data[:, rslc, cslc]
            else:
                # Use full detector if no ROI defined
                roi_data = detector_data

            # Sum over spatial dimensions for each frame
            self._roi_cache[detector_num] = roi_data.sum(axis=(1, 2))

        return self._roi_cache[detector_num]

    def _get_frame_sum_image(
        self, detector_num: int, start_frame: int, end_frame: int
    ) -> npt.NDArray:
        """Get summed image for the specified frame range."""
        detector_data = self._get_detector_data(detector_num)
        return detector_data[start_frame : end_frame + 1].sum(axis=0)

    def _update_display(self):
        """Update both the image and line plot displays."""
        self._update_image()
        self._update_line_plot()
        self._update_info_text()
        self.fig.canvas.draw_idle()

    def _update_image(self):
        """Update the 2D image display."""
        # Get summed image for current frame range
        summed_image = self._get_frame_sum_image(
            self.current_detector, self.frame_start, self.frame_end
        )

        # Update existing image object
        self.image_obj.set_array(summed_image)
        self.image_obj.set_clim(vmin=summed_image.min(), vmax=summed_image.max())

        # Update title
        self.image_ax.set_title(
            f"Detector {self.current_detector} - Frames {self.frame_start}-{self.frame_end}"
        )

        # Update colorbar
        self._colorbar.update_normal(self.image_obj)

        # Update ROI rectangle if available
        if self.detector_rois and self.current_detector in self.detector_rois.rois:
            roi = self.detector_rois.rois[self.current_detector]
            self.roi_rect.set_xy((roi.cslc.start, roi.rslc.start))
            self.roi_rect.set_width(roi.cslc.stop - roi.cslc.start)
            self.roi_rect.set_height(roi.rslc.stop - roi.rslc.start)
            self.roi_rect.set_visible(True)
        else:
            self.roi_rect.set_visible(False)

    def _update_line_plot(self):
        """Update the 1D line plot display."""
        # Get ROI sum data
        roi_sums = self._get_roi_sum(self.current_detector)

        # Update existing line object
        self.line_obj.set_ydata(roi_sums)

        # Update axis limits and title
        self.line_ax.set_ylim(roi_sums.min() * 0.95, roi_sums.max() * 1.05)
        self.line_ax.set_title(f"Detector {self.current_detector} ROI Sum vs Arm 2θ")

        # Update secondary axis (frame numbers)
        self.frame_ax.set_xlim(self.line_ax.get_xlim())
        tth_min, tth_max = self.line_ax.get_xlim()
        frame_min = int(np.searchsorted(self.arm_tth, tth_min))
        frame_max = int(np.searchsorted(self.arm_tth, tth_max))

        # Set secondary axis ticks
        n_ticks = 5
        frame_ticks = np.linspace(frame_min, frame_max - 1, n_ticks, dtype=int)
        tth_ticks = self.arm_tth[frame_ticks]
        self.frame_ax.set_xticks(tth_ticks)
        self.frame_ax.set_xticklabels([str(f) for f in frame_ticks])

    def _update_info_text(self):
        """Update the information text."""
        width = self.frame_end - self.frame_start + 1
        self.width_text.set_text(
            f"Selection: frames {self.frame_start}-{self.frame_end} (width: {width})"
        )

    def _on_detector_change(self, label: str | None):
        """Handle detector radio button change."""
        if label is None:
            return
        # Extract detector number from label (e.g., "Det 1" -> 1)
        detector_num = int(label.split()[1])
        self.current_detector = detector_num
        self._update_display()

    def _on_span_select(self, xmin: float, xmax: float):
        """Handle span selector change."""
        # Convert tth values to frame indices
        start_frame = int(np.searchsorted(self.arm_tth, xmin))
        end_frame = int(np.searchsorted(self.arm_tth, xmax))

        # Ensure valid range
        start_frame = max(0, min(start_frame, self.n_frames - 1))
        end_frame = max(start_frame, min(end_frame, self.n_frames - 1))

        self.frame_start = start_frame
        self.frame_end = end_frame

        # Update only the image and fill region (more efficient than full update)
        self._update_image()
        self._update_info_text()
        self.fig.canvas.draw_idle()

    def _on_back(self, event):  # noqa: ARG002
        """Handle back button click."""
        width = self.frame_end - self.frame_start
        new_start = max(0, self.frame_start - width)
        new_end = new_start + width

        tth_start = self.arm_tth[new_start]
        tth_end = self.arm_tth[new_end]
        self.span_selector.extents = (tth_start, tth_end)
        self.span_selector.onselect(tth_start, tth_end)

    def _on_next(self, event):  # noqa: ARG002
        """Handle next button click."""
        width = self.frame_end - self.frame_start
        new_start = min(self.n_frames - width - 1, self.frame_start + width)
        new_end = new_start + width

        tth_start = self.arm_tth[new_start]
        tth_end = self.arm_tth[new_end]
        self.span_selector.extents = (tth_start, tth_end)
        self.span_selector.onselect(tth_start, tth_end)

    def show(self):
        """Display the scrubber interface."""
        plt.show()


def main():
    """Example usage of the ImageScrubber."""
    import argparse
    from pathlib import Path

    from multihead.file_io import open_data
    from multihead.raw_proc import compute_rois

    parser = argparse.ArgumentParser(description="Launch image scrubber for HRD data")
    parser.add_argument("filename", help="Path to the data file")
    parser.add_argument(
        "--ver",
        type=int,
        default=2,
        choices=[1, 2],
        help="Data format version (default: 2)",
    )
    parser.add_argument("--roi-config", type=str, help="Path to ROI configuration file")

    args = parser.parse_args()

    # Load raw data
    raw = open_data(args.filename, args.ver)

    # Load or compute ROIs
    detector_rois = None
    if args.roi_config:
        roi_path = Path(args.roi_config)
        if roi_path.exists():
            detector_rois = DetectorROIs.from_yaml(roi_path)
            print(f"Loaded ROI configuration from {roi_path}")
        else:
            print(f"ROI config file not found: {roi_path}")

    if detector_rois is None:
        print("Computing ROIs from detector sums...")
        detector_rois = compute_rois(raw.get_detector_sums())

    # Create and show scrubber
    scrubber = ImageScrubber(raw, detector_rois)
    scrubber.show()


if __name__ == "__main__":
    main()
