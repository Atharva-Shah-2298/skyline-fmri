"""Command line entry point for Skyline fMRI."""

from dipy.utils.optpkg import optional_package
from dipy.workflows.flow_runner import run_flow
from dipy.workflows.workflow import Workflow

fury_pckg_msg = (
    "Skyline requires FURY version 2.1.0 or higher. "
    "Please install or upgrade FURY using pip install -U fury."
)
fury, has_fury, _ = optional_package(
    "fury", trip_msg=fury_pckg_msg, min_version="2.1.0"
)
if has_fury:
    from skyline_fmri import skyline_from_files
else:
    skyline_from_files = fury


class SkylineFMRIFlow(Workflow):
    @classmethod
    def get_short_name(cls):
        return "skyline_fmri"

    def run(
        self,
        input_files,
        *,
        rois=None,
        peaks=None,
        odfs=None,
        bold=None,
        stat_maps=None,
        events=None,
        tr=0.0,
        layout="3d",
        sh_basis="descoteaux07",
        cluster=False,
        performance_version=False,
        glass_brain=False,
        bg_color=None,
        tract_colors=None,
        cluster_thr=15.0,
        cluster_size_thr=-1,
        cluster_length_thr=-1,
        buan_pvals=None,
        stealth=False,
        rgb=None,
        out_dir="",
        out_stealth_png="out_skyline.png",
    ):
        """Launch Skyline GUI.

        If you want to load only odfs or rois.
        use dipy_skyline run --odfs <Your ODF files> or
        dipy_skyline run --rois <Your ROI files> respectively. You can also load
        peaks with dipy_skyline run --peaks <Your peak files>.

        For fMRI, pass BOLD runs with --bold, activation maps with --stat_maps
        and a BIDS events table with --events, next to any anatomical or
        diffusion input files.

        These options should be used in stealth mode. For GUI mode, the files can be
        loaded through the file dialog.

        Parameters
        ----------
        input_files : variable string or Path
            Tuple of path for each image, peak, surface or tractogram to be added to
            the Skyline viewer.
        rois : variable str, optional
            Tuple of path for each ROI to be added to the Skyline viewer.
        peaks : variable str, optional
            Tuple of path for each peaks file (.pam5, or NIfTI with shape
            (X, Y, Z, 3*N) or (X, Y, Z, N, 3)) to be added to the Skyline viewer.
        odfs : variable str, optional
            Tuple of path for each ODF file (.pam5, or 4D NIfTI of SH
            coefficients) to be added to the Skyline viewer.
        bold : variable str, optional
            Tuple of path for each 4D BOLD run (.nii, .nii.gz) to show with
            time navigation and a time-series panel.
        stat_maps : variable str, optional
            Tuple of path for each 3D statistical map (t, z, correlation, ...)
            to overlay as a thresholded activation map.
        events : str, optional
            BIDS events table (.tsv or .csv) attached to every BOLD run, used
            to shade the time series and fit a task GLM.
        tr : float, optional
            Repetition time in seconds for every BOLD run. 0 reads it from the
            BIDS sidecar (.json) or the NIfTI header.
        layout : str, optional
            Initial view layout: '3d' for the 3D view or 'triplanar' for 2D
            axial, sagittal and coronal views next to the 3D view.
        sh_basis : str, optional
            Input convention for PAM and NIfTI ODFs: 'descoteaux07' (latest),
            'descoteaux07_legacy' (old DIPY), 'tournier07' (MRtrix 0.2),
            or 'tournier19' (MRtrix3).
            Unsupported values are logged and use latest Descoteaux07.
        cluster : bool, optional
            Whether to cluster the tractograms.
        performance_version : bool, optional
            Whether to use the performance version of the tractogram rendering.
            This will render tractograms as lines instead of tubes,
            which can improve performance for large tractograms.
        glass_brain : bool, optional
            Glass brain view: a translucent shell of the first image, activation
            clusters as 3D surfaces and hidden slices. The background defaults
            to white unless set explicitly.
        bg_color : variable float, optional
            Define the background color of the scene. Colors can be defined with
            3 values and should be between [0-1].
            For example, a value of (0, 0, 0) would mean the black color.
        tract_colors : str, optional
            Define the colors of the tractograms. Colors can be defined with
            3 values and should be between [0-1].
            String options are 'random' for random colors for each tractogram,
            'direction'  for directionally colored streamlines.
            For example, a value of (1, 0, 0) would mean the red color.
        cluster_thr : float, optional
            Distance threshold used for clustering. Default value 15.0 for
            small animal brains you may need to use something smaller such
            as 2.0. The distance is in mm. For this parameter to be active
            ``cluster`` should be enabled.
        cluster_size_thr : int, optional
            Clusters with size less than ``cluster_size_thr`` will be hidden.
            If -1, it will show all cluster above the 50th percentile of the cluster
            size distribution.
        cluster_length_thr : float, optional
            Clusters with average length less than ``cluster_length_thr`` in mm will be
            hidden. If -1, it will show all cluster above the 25th percentile of the
            cluster length distribution.
        buan_pvals : variable str, optional
            File path for BUAN p-values to be used for BUAN-based coloring of
            tractograms.
        stealth : bool, optional
            Capture a figure instead of opening an interactive window.
            ``DIPY_OFFSCREEN=1`` or ``true`` (case-insensitive) also captures when False.
        rgb : bool or None, optional
            ``None``: auto-detect from structured NIfTI ``DT_RGB24`` dtype;
            ``--rgb`` forces RGB; ``--no-rgb`` disables it entirely.
        out_dir : str or Path, optional
            Output directory for offscreen figures.
        out_stealth_png : str, optional
            Filename of the offscreen figure.

        Returns
        -------
        Skyline
            The viewer, returned once its window is closed or the offscreen
            figure is saved.
        """
        super().__init__(force=True)
        skyline_input_files = []
        start_gui = input_files is not None and input_files[0] in (
            "run",
            "start",
            "launch",
            "initialize",
        )
        if not start_gui:
            io_it = self.get_io_iterator()
            skyline_input_files.extend(input_output[0] for input_output in io_it)

        if cluster_length_thr == -1:
            # set default as None is not allowed in int
            # Further down it will be set to
            # 25th percentile of the cluster length distribution
            cluster_length_thr = None
        if cluster_size_thr == -1:
            # set default as None is not allowed in int
            # Further down it will be set to
            # 50th percentile of the cluster size distribution
            cluster_size_thr = None

        return skyline_from_files(
            skyline_input_files,
            rois=rois,
            peaks=peaks,
            shm_coeffs=odfs,
            functional=bold,
            stat_maps=stat_maps,
            events=events,
            tr=tr if tr and tr > 0 else None,
            sh_basis=sh_basis,
            is_cluster=cluster,
            is_light_version=performance_version,
            glass_brain=glass_brain,
            bg_color=bg_color,
            tract_colors=tract_colors,
            cluster_thr=cluster_thr,
            cluster_size_thr=cluster_size_thr,
            cluster_length_thr=cluster_length_thr,
            buan_pvals=buan_pvals,
            stealth=stealth,
            rgb=rgb,
            layout=layout,
            out_dir=out_dir,
            out_stealth_png=out_stealth_png,
        )


def main():
    """Run the ``skyline_fmri`` command line interface."""
    run_flow(SkylineFMRIFlow())
