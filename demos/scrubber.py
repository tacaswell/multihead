"""
Interactive image scrubber UI for HRD detector data.

This module provides a matplotlib-based interactive interface for exploring
raw detector data with frame selection capabilities.
"""

import functools
import matplotlib
matplotlib.use('qtagg')
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from matplotlib.widgets import Button, RadioButtons, RectangleSelector, SpanSelector
from matplotlib.backends.qt_compat import QtWidgets

from multihead.config import CrystalROI, DetectorROIs, SimpleSliceTuple
from multihead.file_io import HRDRawBase
from multihead.raw_proc import find_crystal_range


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
            ROI definitions for each detector. If None, center 240x240 pixel ROIs are used.
        """
        self.raw = raw

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

        # Initialize ROIs for all detectors - merge static/dynamic into single structure
        self._detector_rois: dict[int, CrystalROI] = {}
        self._initialize_detector_rois(detector_rois)

        # Create the UI
        self._setup_figure()
        self._setup_widgets()

        # Initialize matplotlib objects that will be updated
        self._init_plot_objects()
        self._update_display()

    def _initialize_detector_rois(self, detector_rois: DetectorROIs | None):
        """Initialize ROIs for all detectors, using provided config or defaults."""
        for detector_num in self.detector_numbers:
            if detector_rois and detector_num in detector_rois.rois:
                # Use provided ROI
                self._detector_rois[detector_num] = detector_rois.rois[detector_num]
            else:
                # Create default ROI: 6-250 in each direction for 256x256 detectors
                self._detector_rois[detector_num] = CrystalROI(
                    rslc=SimpleSliceTuple(6, 250),
                    cslc=SimpleSliceTuple(6, 250)
                )

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

        # Rectangle selector for dynamic ROI selection
        self.rect_selector = RectangleSelector(
            self.image_ax,
            self._on_roi_select,
            useblit=True,
            button=[1],  # Only left mouse button
            minspanx=5,
            minspany=5,
            spancoords='pixels',
            interactive=True,
            props=dict(alpha=0.3, facecolor="None", edgecolor="blue", linewidth=2),
        )

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
        back_ax = plt.axes((0.02, 0.05, 0.06, 0.04))
        self.back_button = Button(back_ax, "Back")
        self.back_button.on_clicked(self._on_back)

        # Next button
        next_ax = plt.axes((0.09, 0.05, 0.06, 0.04))
        self.next_button = Button(next_ax, "Next")
        self.next_button.on_clicked(self._on_next)

        # Save data button
        save_data_ax = plt.axes((0.16, 0.05, 0.08, 0.04))
        self.save_data_button = Button(save_data_ax, "Save Data")
        self.save_data_button.on_clicked(self._on_save_data)

        # Save ROI button
        save_roi_ax = plt.axes((0.25, 0.05, 0.08, 0.04))
        self.save_roi_button = Button(save_roi_ax, "Save ROI")
        self.save_roi_button.on_clicked(self._on_save_roi)

        # Auto ROI button
        auto_roi_ax = plt.axes((0.34, 0.05, 0.08, 0.04))
        self.auto_roi_button = Button(auto_roi_ax, "Auto ROI")
        self.auto_roi_button.on_clicked(self._on_auto_roi)

        # Width info text - moved to right side to avoid button overlap
        self.width_text = self.nav_ax.text(
            0.85, 0.5, "", ha="center", va="center", transform=self.nav_ax.transAxes
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

        # Initialize dynamic ROI rectangle (blue, for RectangleSelector feedback)
        self.dynamic_roi_rect = Rectangle(
            (0, 0), 1, 1, linewidth=2, edgecolor="blue", facecolor="none", alpha=0.7, visible=False
        )
        self.image_ax.add_patch(self.dynamic_roi_rect)

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

    @functools.lru_cache(maxsize=3)
    def _get_detector_data(self, detector_num: int) -> npt.NDArray[np.uint16]:
        """Get detector data with caching (max 3 detectors)."""
        return self.raw.get_detector(detector_num)

    def _get_roi_sum(self, detector_num: int) -> npt.NDArray:
        """Get ROI sum for a detector."""
        detector_data = self._get_detector_data(detector_num)
        current_roi = self._detector_rois[detector_num]

        rslc, cslc = current_roi.to_slices()
        roi_data = detector_data[:, rslc, cslc]

        # Sum over spatial dimensions for each frame
        return roi_data.sum(axis=(1, 2))

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

        # Update ROI rectangle and rectangle selector - now always visible
        current_roi = self._detector_rois[self.current_detector]
        self.roi_rect.set_xy((current_roi.cslc.start, current_roi.rslc.start))
        self.roi_rect.set_width(current_roi.cslc.stop - current_roi.cslc.start)
        self.roi_rect.set_height(current_roi.rslc.stop - current_roi.rslc.start)
        self.roi_rect.set_visible(True)

        # Update rectangle selector to match current ROI
        x1, y1 = current_roi.cslc.start, current_roi.rslc.start
        x2, y2 = current_roi.cslc.stop, current_roi.rslc.stop
        self.rect_selector.extents = (x1, x2, y1, y2)

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

    def _on_roi_select(self, eclick, erelease):
        """Handle ROI rectangle selection."""
        # Get rectangle coordinates (matplotlib uses bottom-left origin)
        x1, y1 = int(eclick.xdata), int(eclick.ydata)
        x2, y2 = int(erelease.xdata), int(erelease.ydata)

        # Ensure proper ordering
        xmin, xmax = min(x1, x2), max(x1, x2)
        ymin, ymax = min(y1, y2), max(y1, y2)

        # Create new ROI and update the unified structure
        new_roi = CrystalROI(
            rslc=SimpleSliceTuple(ymin, ymax),
            cslc=SimpleSliceTuple(xmin, xmax)
        )

        self._detector_rois[self.current_detector] = new_roi

        # Update displays
        self._update_line_plot()
        self._update_image()
        self.fig.canvas.draw_idle()

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

    def _on_save_data(self, event):  # noqa: ARG002
        """Handle save data button click - export ROI sum vs angle data."""
        # Get current ROI data
        roi_sums = self._get_roi_sum(self.current_detector)
        current_roi = self._detector_rois[self.current_detector]

        # ROI description for filename suggestion
        roi_desc = f"roi_{current_roi.rslc.start}-{current_roi.rslc.stop}_{current_roi.cslc.start}-{current_roi.cslc.stop}"

        # Suggest filename
        default_filename = f"detector_{self.current_detector}_{roi_desc}_frames_{self.frame_start}-{self.frame_end}.xy"

        # Open file dialog using Qt
        filename, _ = QtWidgets.QFileDialog.getSaveFileName(
            None,
            "Save ROI Sum vs Angle Data",
            default_filename,
            "XY data files (*.xy);;Tab-separated values (*.tsv);;Comma-separated values (*.csv);;Text files (*.txt);;All files (*.*)"
        )

        if filename:
            try:
                # Determine separator based on file extension
                separator = "\t" if filename.endswith((".tsv", ".txt", ".xy")) else ","

                # Create header with metadata
                header_lines = [
                    f"# HRD Image Scrubber Export",
                    f"# Detector: {self.current_detector}",
                    f"# Frame range: {self.frame_start}-{self.frame_end}",
                    f"# Total frames in range: {self.frame_end - self.frame_start + 1}",
                    f"# ROI rows: {current_roi.rslc.start}-{current_roi.rslc.stop}",
                    f"# ROI columns: {current_roi.cslc.start}-{current_roi.cslc.stop}",
                    f"# Data columns: Arm_2theta_degrees{separator}ROI_Sum",
                    ""  # Empty line before data
                ]

                # Write data
                with open(filename, 'w') as f:
                    # Write header
                    for line in header_lines:
                        f.write(line + "\n")

                    # Write column headers
                    f.write(f"Arm_2theta_degrees{separator}ROI_Sum\n")

                    # Write data
                    for angle, intensity in zip(self.arm_tth, roi_sums):
                        f.write(f"{angle:.6f}{separator}{intensity}\n")

                print(f"Data saved to: {filename}")

            except Exception as e:
                print(f"Error saving file: {e}")

    def _on_save_roi(self, event):  # noqa: ARG002
        """Handle save ROI button click - export ROI configuration."""
        if not self._detector_rois:
            print("No ROIs to save.")
            return

        # Suggest filename
        default_filename = "detector_rois.yaml"

        # Open file dialog using Qt
        filename, _ = QtWidgets.QFileDialog.getSaveFileName(
            None,
            "Save ROI Configuration",
            default_filename,
            "YAML files (*.yaml *.yml);;All files (*.*)"
        )

        if filename:
            try:
                # Create DetectorROIs object with current ROIs
                detector_rois = DetectorROIs(
                    rois=self._detector_rois.copy(),
                    software={
                        "name": "HRD Image Scrubber",
                        "version": "1.0",
                        "module": "multihead.demos.scrubber"
                    },
                    parameters={
                        "total_detectors": len(self.detector_numbers),
                        "source": "interactive_selection",
                        "frame_range": f"{self.frame_start}-{self.frame_end}",
                        "rois_count": len(self._detector_rois)
                    }
                )

                # Save to YAML
                detector_rois.to_yaml(filename)
                print(f"ROI configuration saved to: {filename}")
                print(f"  Total ROIs: {len(self._detector_rois)}")

            except Exception as e:
                print(f"Error saving ROI file: {e}")

    # Rename the old _on_save method to maintain compatibility
    def _on_save(self, event):  # noqa: ARG002
        """Legacy save method - redirect to save data."""
        self._on_save_data(event)

    def _on_auto_roi(self, event):  # noqa: ARG002
        """Handle auto ROI button click - automatically detect ROI using find_crystal_range."""
        try:
            # Get current summed image
            summed_image = self._get_frame_sum_image(
                self.current_detector, self.frame_start, self.frame_end
            )

            threshold = 15
            photon_mask = summed_image > threshold

            # Use find_crystal_range to detect ROI
            mask, detected_roi = find_crystal_range(
                photon_mask, opening_radius=5, closing_radius=10
            )

            # Update the ROI for current detector
            self._detector_rois[self.current_detector] = detected_roi

            # Update displays
            self._update_line_plot()
            self._update_image()
            self.fig.canvas.draw_idle()

            print(f"Auto-detected ROI for detector {self.current_detector}: "
                  f"rows {detected_roi.rslc.start}-{detected_roi.rslc.stop}, "
                  f"cols {detected_roi.cslc.start}-{detected_roi.cslc.stop}")

        except Exception as e:
            print(f"Error in auto ROI detection: {e}")
            print("Try adjusting the frame range or manually selecting an ROI")

    def show(self):
        """Display the scrubber interface."""
        plt.show()


def main():
    """Example usage of the ImageScrubber."""
    import argparse
    from pathlib import Path

    from multihead.file_io import open_data

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

    # Load ROIs if specified
    detector_rois = None
    if args.roi_config:
        roi_path = Path(args.roi_config)
        if roi_path.exists():
            detector_rois = DetectorROIs.from_yaml(roi_path)
            print(f"Loaded ROI configuration from {roi_path}")
        else:
            print(f"ROI config file not found: {roi_path}")
            print("Using full detector area as ROI")
    else:
        print("No ROI configuration specified, using full detector area as ROI")

    # Create and show scrubber
    scrubber = ImageScrubber(raw, detector_rois)
    scrubber.show()


if __name__ == "__main__":
    main()
