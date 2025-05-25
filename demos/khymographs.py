
def extract_khymo(
    detector_series: npt.NDArray, roi: CrystalROI | None = None
) -> npt.NDArray:
    """
    Extract the khymograph for a single detector sequence.

    Parameters
    ----------
    detector_series : array
       Expected order is (time, row, col)

    roi: CrystalROI
       The ROI to consider.  If None, sum whole detector

    Returns
    -------

    """
    if roi is None:
        rslc = cslc = slice(-1)
    else:
        rslc, cslc = roi.to_slices()

    return cast(npt.NDArray, np.sum(detector_series[:, rslc, cslc], axis=1))


# %%
def get_arm_tth(self) -> npt.NDArray[np.float64]:
    sc = self._mda.scan_config
    (steps_per_bin,) = sc["MCS prescale"].value
    (step_size,) = sc["encoder resolution"].value

    bin_size: float = steps_per_bin * step_size

    (Npts,) = sc["NPTS"].value
    start_tth: float
    (start_tth,) = sc["start_tth_rbk"].value

    return start_tth + bin_size * np.arange(Npts, dtype=float)


def get_monitor(self):
    return {_.desc: _.data for _ in self._mda.scan.d}


# %%
def get_khymograph(
    raw: RawHRPD11BM, detector: int, roi: CrystalROI | None = None
) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.uint16]]:
    """
    Extract the khymograph for a single detector sequence.

    Parameters
    ----------
    raw :
       Expected order is (time, row, col)

    roi: CrystalROI
       The ROI to consider.  If None, sum whole detector

    """
    series = raw.get_detector(detector)
    print(roi)
    khymo = extract_khymo(series, roi)

    tth = get_arm_tth(raw)

    return tth, khymo


# %%


def all_khymographs(
    raw: RawHRPD11BM, rois: dict[int, CrystalROI]
) -> dict[int, tuple[npt.NDArray[np.floating], npt.NDArray[np.uint16]]]:
    return {det: get_khymograph(raw, det, roi) for det, roi in rois.items()}


# %%


def integrate_simple(tth: npt.NDArray[np.floating], khymo: npt.NDArray):
    return tth, khymo.sum(axis=1)


# %%

rois2 = compute_rois(sums)
all_khymos = all_khymographs(t, rois2)

# %%

flats = {det: integrate_simple(tth, khymo) for det, (tth, khymo) in all_khymos.items()}

# %%


def estimate_crystal_offsets_ref(
    raw: RawHRPD11BM,
    flats: dict[int, tuple[npt.NDArray[np.floating], npt.NDArray[np.uint16]]],
    ref_indx: int = 0,
) -> dict[int, float]:
    sc = raw._mda.scan_config
    (steps_per_bin,) = sc["MCS prescale"].value
    (step_size,) = sc["encoder resolution"].value

    bin_size: float = steps_per_bin * step_size
    ref = flats[ref_indx][1]
    (Npts,) = ref.shape
    out: dict[int, float] = {}
    for det, (_, I) in flats.items():
        offset = np.argmax(np.correlate(ref, I, mode="full")) - Npts
        out[det] = offset * bin_size

    return out


def estimate_crystal_offsets(
    raw: RawHRPD11BM,
    flats: dict[int, tuple[npt.NDArray[np.floating], npt.NDArray[np.uint16]]],
) -> dict[int, float]:
    sc = raw._mda.scan_config
    (steps_per_bin,) = sc["MCS prescale"].value
    (step_size,) = sc["encoder resolution"].value

    bin_size: float = steps_per_bin * step_size

    out: dict[int, float] = {}
    iterator = iter(flats.items())
    det, (_, ref) = next(iterator)
    (Npts,) = ref.shape
    out[det] = cum_offset = 0.0

    for det, (_, I) in iterator:
        offset = np.argmax(np.correlate(ref, I, mode="full")) - Npts
        cum_offset += offset * bin_size
        out[det] = cum_offset
        ref = I

    return out


# %%

offsets = estimate_crystal_offsets(t, flats)

# %%

fig, ax = plt.subplots(layout="constrained")

{
    ax.plot(tth + offsets[d], I + d * 200, label=str(d))[0]
    for d, (tth, I) in flats.items()
}
ax.legend()
ax.set_xlabel("tth")
ax.set_ylabel("I")

plt.show()
