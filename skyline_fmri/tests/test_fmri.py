import json

import nibabel as nib
import numpy as np
import numpy.testing as npt
import pytest

from skyline_fmri.fmri import (
    epi_mask,
    fit_glm,
    load_events,
    normalize_timeseries,
    percent_change,
    power_spectrum,
    repetition_time,
    seed_correlation,
    spm_hrf,
    task_regressors,
    threshold_clusters,
    voxel_from_world,
)

TR = 2.0
N_SCANS = 120
BLOCKS = {"task": np.array([[t, 20.0] for t in range(20, 240, 40)], dtype=float)}


def _write(path, text):
    path.write_text(text)
    return path


def test_spm_hrf_peaks_near_five_seconds_with_an_undershoot():
    dt = 0.1
    hrf = spm_hrf(dt)

    npt.assert_allclose(hrf.sum(), 1.0)
    assert 4.5 <= np.argmax(hrf) * dt <= 5.5
    undershoot = np.argmin(hrf) * dt
    assert 12 <= undershoot <= 18
    assert hrf.min() < 0


def test_load_events_groups_conditions_and_skips_missing_rows(tmp_path):
    path = _write(
        tmp_path / "events.tsv",
        "onset\tduration\ttrial_type\n"
        "30\t10\tfaces\n"
        "0\t10\thouses\n"
        "10\t10\tfaces\n"
        "n/a\t10\tfaces\n"
        "50\t5\tn/a\n",
    )

    events = load_events(path)

    assert list(events) == ["faces", "houses", "task"]
    npt.assert_array_equal(events["faces"], [[10, 10], [30, 10]])
    npt.assert_array_equal(events["houses"], [[0, 10]])
    npt.assert_array_equal(events["task"], [[50, 5]])


def test_load_events_reads_csv_without_trial_type(tmp_path):
    path = _write(tmp_path / "events.csv", "onset,duration\n5,2\n1,2\n")

    events = load_events(path)

    npt.assert_array_equal(events["task"], [[1, 2], [5, 2]])


def test_load_events_requires_onset_and_duration(tmp_path):
    path = _write(tmp_path / "events.tsv", "start\tlength\n1\t2\n")

    with pytest.raises(ValueError, match="onset"):
        load_events(path)


def test_repetition_time_prefers_the_bids_sidecar(tmp_path):
    img = nib.Nifti1Image(np.zeros((2, 2, 2, 3), dtype=np.float32), np.eye(4))
    img.header.set_zooms((1, 1, 1, 3.0))
    path = tmp_path / "bold.nii.gz"
    nib.save(img, path)
    (tmp_path / "bold.json").write_text(json.dumps({"RepetitionTime": 0.8}))

    assert repetition_time(fname=path, img=img) == pytest.approx(0.8)


def test_repetition_time_converts_header_milliseconds():
    img = nib.Nifti1Image(np.zeros((2, 2, 2, 3), dtype=np.float32), np.eye(4))
    img.header.set_zooms((1, 1, 1, 2500.0))
    img.header.set_xyzt_units("mm", "msec")

    assert repetition_time(img=img) == pytest.approx(2.5)


def test_repetition_time_is_none_for_3d_images():
    img = nib.Nifti1Image(np.zeros((2, 2, 2), dtype=np.float32), np.eye(4))

    assert repetition_time(img=img) is None


def _synthetic_run(seed=0):
    """Block-design run where voxel (1, 1, 1) responds and the rest is noise."""
    rng = np.random.default_rng(seed)
    _, regressors = task_regressors(BLOCKS, N_SCANS, TR)
    data = 100 + rng.normal(0, 1, (4, 4, 4, N_SCANS))
    data[1, 1, 1] += 5 * regressors[:, 0]
    data[2, 2, 2] += np.linspace(0, 10, N_SCANS)  # slow drift only
    return data.astype(np.float32)


def test_fit_glm_detects_the_responsive_voxel_and_ignores_drift():
    data = _synthetic_run()

    t_map = fit_glm(data, BLOCKS, TR)["task"]

    assert np.unravel_index(np.argmax(t_map), t_map.shape) == (1, 1, 1)
    assert t_map[1, 1, 1] > 10
    assert abs(t_map[2, 2, 2]) < 4
    others = np.delete(t_map.ravel(), np.ravel_multi_index((1, 1, 1), t_map.shape))
    assert np.abs(others).max() < 5


def test_fit_glm_leaves_voxels_outside_the_mask_at_zero():
    data = _synthetic_run()
    mask = np.zeros(data.shape[:3], dtype=bool)
    mask[1, 1, 1] = True

    t_map = fit_glm(data, BLOCKS, TR, mask=mask)["task"]

    assert t_map[1, 1, 1] > 10
    assert np.count_nonzero(t_map) == 1


def test_fit_glm_requires_conditions():
    with pytest.raises(ValueError, match="condition"):
        fit_glm(_synthetic_run(), {}, TR)


def test_seed_correlation_matches_pearson_and_zeroes_constant_voxels():
    rng = np.random.default_rng(1)
    seed = rng.normal(size=50)
    data = rng.normal(size=(3, 1, 1, 50))
    data[0, 0, 0] = 3 * seed + 7
    data[1, 0, 0] = -seed
    data[2, 0, 0] = 4.0

    r_map = seed_correlation(data, seed)

    npt.assert_allclose(r_map[:, 0, 0], [1.0, -1.0, 0.0], atol=1e-6)


def test_threshold_clusters_drops_small_clusters_and_reports_peaks():
    stat = np.zeros((10, 10, 10), dtype=np.float32)
    stat[1:4, 1:4, 1:4] = 5.0
    stat[2, 2, 2] = 9.0
    stat[7, 7, 7] = -6.0  # single-voxel negative cluster
    affine = np.diag([2.0, 2.0, 2.0, 1.0])

    kept, clusters, labels = threshold_clusters(stat, 3.0, min_size=2, affine=affine)

    assert len(clusters) == 1
    assert clusters[0]["size"] == 27
    assert clusters[0]["peak"] == pytest.approx(9.0)
    npt.assert_allclose(clusters[0]["peak_world"], [4.0, 4.0, 4.0])
    assert kept[7, 7, 7] == 0
    assert kept[0, 0, 0] == 0
    npt.assert_array_equal(kept[1:4, 1:4, 1:4], stat[1:4, 1:4, 1:4])


def test_threshold_clusters_keeps_only_the_requested_tail():
    stat = np.zeros((5, 5, 5), dtype=np.float32)
    stat[0, 0, 0] = 4.0
    stat[4, 4, 4] = -4.0

    _, negative, _ = threshold_clusters(stat, 3.0, sign="negative")
    _, both, labels = threshold_clusters(stat, 3.0)

    assert [c["peak"] for c in negative] == [-4.0]
    assert sorted(c["peak"] for c in both) == [-4.0, 4.0]
    # Labels are unique across tails and match each cluster's peak voxel.
    assert {labels[c["peak_index"]] for c in both} == {1, 2}
    for cluster in both:
        assert labels[cluster["peak_index"]] == cluster["label"]
    with pytest.raises(ValueError, match="sign"):
        threshold_clusters(stat, 3.0, sign="up")


def test_normalize_timeseries_scales_and_detrends():
    ts = 200 + np.linspace(0, 20, 21)

    percent = normalize_timeseries(ts, mode="percent")
    zscore = normalize_timeseries(ts, mode="zscore")
    detrended = normalize_timeseries(ts, mode="raw", detrend=True)

    npt.assert_allclose(percent[[0, -1]], [-100 * 10 / 210, 100 * 10 / 210])
    npt.assert_allclose([zscore.mean(), zscore.std()], [0.0, 1.0], atol=1e-12)
    npt.assert_allclose(detrended, np.full(21, ts.mean()))
    with pytest.raises(ValueError, match="mode"):
        normalize_timeseries(ts, mode="log")


def test_power_spectrum_peaks_at_the_signal_frequency():
    t = np.arange(200) * TR
    freqs, power = power_spectrum(np.sin(2 * np.pi * 0.05 * t), TR)

    assert freqs[np.argmax(power)] == pytest.approx(0.05)
    assert freqs.max() == pytest.approx(0.5 / TR)


def test_voxel_from_world_rounds_and_rejects_outside_points():
    affine = np.array(
        [[-2.0, 0, 0, 10], [0, 2.0, 0, -10], [0, 0, 2.0, 0], [0, 0, 0, 1]]
    )

    assert voxel_from_world([6.2, -6.1, 4.1], affine, (5, 5, 5)) == (2, 2, 2)
    assert voxel_from_world([12.0, 0, 0], affine, (5, 5, 5)) is None


def test_epi_mask_keeps_the_support_of_premasked_runs():
    data = np.zeros((8, 8, 8, 5), dtype=np.float32)
    data[2:6, 2:6, 2:6] = 100.0
    data[0, 0, 0] = 100.0  # isolated speck outside the brain

    mask = epi_mask(data)

    assert mask.sum() == 64
    assert mask[3, 3, 3] and not mask[0, 0, 0]


def test_percent_change_removes_drift_and_scales_by_the_mean():
    n_scans = 100
    response = np.zeros(n_scans)
    response[40:50] = 3.0  # about 2.6% of the run's mean of ~115
    data = np.zeros((2, 1, 1, n_scans), dtype=np.float32)
    data[0, 0, 0] = 100 + response + np.linspace(0, 30, n_scans)  # strong drift
    mask = np.array([True, False]).reshape(2, 1, 1)

    change = percent_change(data, TR, mask=mask)

    series = change[0, 0, 0]
    assert series[40:50].mean() - np.delete(series, range(40, 50)).mean() > 1.5
    assert abs(series[:30].mean()) < 0.5  # drift is gone
    npt.assert_array_equal(change[1], 0)
