==================
Using Skyline fMRI
==================

Skyline fMRI is an interactive viewer for diffusion and functional MRI data. It
shows images, tractograms, peaks, ODFs, ROIs, surfaces, BOLD runs and
activation maps together in one 3D scene and is started from the command line
with ``skyline_fmri``.

To follow this tutorial, install Skyline fMRI (see the README). ::

  pip install git+https://github.com/Atharva-Shah-2298/skyline-fmri.git

Let's explore the options that Skyline provides. ::

  skyline_fmri --help


------------------
The Skyline window
------------------

Every command in this tutorial opens the same window: the 3D scene on the
right and a control panel on the left. The top of the panel has three buttons:

* **Add Visualization** opens a menu to load *3D/4D Images*, *Functional
  (BOLD) Runs*, *Activation / Statistical Maps*, *Task Events*, *Peaks*,
  *ODFs*, *Surfaces*, *Tractograms* or *ROIs* through a file dialog. For
  *ODFs*, pick the *Input SH basis* before *Select files...*.
* **Change Background Color** sets the scene background.
* **Take Snapshot** saves the current view as a PNG file.

Below them, the **Global Slices** block moves the X, Y and Z slices of every
dataset whose *Synchronize Slices* toggle is on, and the toggle next to each
axis shows or hides that axis for all datasets at once. The dataset that
defines the slice coordinates is shown as *Reference*.

Loaded datasets are grouped as *Images*, *Functional (BOLD)*, *Activation
Maps*, *Tractograms*, *Peaks*, *ODFs*, *ROIs* and *Surfaces*. Each group can be
collapsed or hidden, and each dataset opens its own controls and can be removed
from the scene. Clicking a dataset's title brings its slices to the front.

You can also start with an empty window and load everything from the menu,
which opens automatically. ::

  skyline_fmri run


-------------------------
Visualize 3D Brain Images
-------------------------

Using a terminal, let's download a dataset called ``mni_template``. You can
skip this step if you already have the dataset downloaded. ::

  dipy_fetch mni_template

To see more details about ``dipy_fetch`` you can refer to :ref:`data_fetch`.

This command will download the data in your ``.dipy`` folder placed in your
home directory.

Let's load the T1 and T2 templates together.

**For macOS and Linux** ::

  skyline_fmri ~/.dipy/mni_template/mni_icbm152_t1_tal_nlin_asym_09a.nii ~/.dipy/mni_template/mni_icbm152_t2_tal_nlin_asym_09a.nii

**For Windows (cmd)** ::

  skyline_fmri %USERPROFILE%/.dipy/mni_template/mni_icbm152_t1_tal_nlin_asym_09a.nii %USERPROFILE%/.dipy/mni_template/mni_icbm152_t2_tal_nlin_asym_09a.nii

**For Windows (PowerShell)** ::

  skyline_fmri "\$home/.dipy/mni_template/mni_icbm152_t1_tal_nlin_asym_09a.nii" "\$home/.dipy/mni_template/mni_icbm152_t2_tal_nlin_asym_09a.nii"

Skyline reads ``.nii`` and ``.nii.gz`` images. Each image has these controls:

* *Synchronize Slices* links the image to the Global Slices.
* *Opacity* and the *Slice* sliders for the X, Y and Z planes, each with a
  visibility toggle.
* *Intensities* sets the displayed intensity range by percentiles.
* *Colormap* picks one of Gray, Inferno, Magma, Plasma, Viridis, Cool, Hot,
  Bone, Copper, Pink, Spring, Summer, Autumn, Winter, Jet, Cividis, Distinct
  or Divergent.
* *Interpolation* switches between Linear and Nearest.

Instead of listing files one by one, you can pass a quoted glob pattern or a
folder. Skyline then loads every matching file and skips the formats it does
not support. ::

  skyline_fmri "\$HOME/.dipy/mni_template/*.nii"
  skyline_fmri ~/.dipy/mni_template

On Windows, use ``%USERPROFILE%`` (cmd) or ``\$home`` (PowerShell) instead of
``\$HOME`` and ``~``, as above. The remaining examples show the macOS and Linux
form only.


------------------------
Visualize 4D Brain Image
------------------------

Let's download a diffusion dataset called ``stanford_hardi``. ::

  dipy_fetch stanford_hardi

This dataset has a NIfTI file with 160 volumes.

**For macOS and Linux** ::

  skyline_fmri ~/.dipy/stanford_hardi/HARDI150.nii.gz

**For Windows (cmd)** ::

  skyline_fmri %USERPROFILE%/.dipy/stanford_hardi/HARDI150.nii.gz

**For Windows (PowerShell)** ::

  skyline_fmri "\$home/.dipy/stanford_hardi/HARDI150.nii.gz"

For 4D images, the *Directions* slider selects which volume is shown.

Color images, such as the DEC map that ``dipy_fit_dti`` saves as
``rgb.nii.gz``, are shown in RGB automatically when the file uses the NIfTI
RGB data type. For other 4D images whose last dimension is 3 (RGB) or 4
(RGBA), an *RGB* toggle appears in the image controls. You can also choose
from the command line: ``--rgb`` forces RGB mode and ``--no-rgb`` turns it
off. ::

  skyline_fmri rgb.nii.gz --rgb


---------------------
Visualize Brain ROIs
---------------------

Binary masks given with ``--rois`` are drawn as 3D contours over the anatomy.
The ``mni_template`` dataset includes a brain mask for the 2009c template. ::

  skyline_fmri ~/.dipy/mni_template/mni_icbm152_t1_tal_nlin_asym_09c.nii --rois ~/.dipy/mni_template/mni_icbm152_t1_tal_nlin_asym_09c_mask.nii

Each ROI has an *Opacity* slider and a color picker. ROIs must be ``.nii`` or
``.nii.gz`` files.


--------------------------
Visualize Brain Tractogram
--------------------------

Let's download a dataset called ``bundle_atlas_hcp842``. ::

  dipy_fetch bundle_atlas_hcp842

Skyline supports the following tractogram formats.

* .trk
* .trx
* .dpy
* .tck
* .vtk
* .vtp
* .fib

``.trk`` and ``.trx`` files carry their own spatial reference. The other
formats are placed using the header of the MNI ICBM 2009c template.

Let's load the whole-brain atlas and group its streamlines into clusters.

**For macOS and Linux** ::

  skyline_fmri ~/.dipy/bundle_atlas_hcp842/Atlas_80_Bundles/whole_brain/whole_brain_MNI.trk --cluster

**For Windows (cmd)** ::

  skyline_fmri %USERPROFILE%/.dipy/bundle_atlas_hcp842/Atlas_80_Bundles/whole_brain/whole_brain_MNI.trk --cluster

**For Windows (PowerShell)** ::

  skyline_fmri "\$home/.dipy/bundle_atlas_hcp842/Atlas_80_Bundles/whole_brain/whole_brain_MNI.trk" --cluster

With ``--cluster``, the streamlines are grouped with QuickBundlesX and each
cluster is drawn as its centroid. These options control the clustering:

* ``--cluster_thr`` is the distance threshold in mm (default 15.0). For small
  animal brains you may need something smaller, such as 2.0.
* ``--cluster_size_thr`` hides clusters with fewer streamlines than this
  value. By default it is the 50th percentile of the cluster sizes.
* ``--cluster_length_thr`` hides clusters whose average length in mm is below
  this value. By default it is the 25th percentile of the cluster lengths.

The same values can be changed in the tractogram controls (*Threshold*,
*Size* and *Length*). Click a centroid to select or deselect its cluster,
then use these keys:

* ``e`` expands the selected clusters into their streamlines.
* ``c`` collapses them back into centroids.
* ``a`` selects all clusters and ``d`` deselects all.
* ``h`` hides the deselected clusters and ``s`` shows all clusters again.

The download button in the cluster controls saves the selected clusters as a
new ``.trx`` tractogram.

Without ``--cluster``, every streamline is drawn. Opening a large tractogram
this way can be computationally expensive. The *Cluster* toggle in the
tractogram controls switches between both modes at any time.

**Colors.** By default, streamlines are colored by their local direction. Use
``--tract_colors random`` to give each tractogram its own distinguishable
color, or three numbers between 0 and 1 for a single RGB color. ::

  skyline_fmri ~/.dipy/bundle_atlas_hcp842/Atlas_80_Bundles/bundles/AF_L.trk ~/.dipy/bundle_atlas_hcp842/Atlas_80_Bundles/bundles/CST_L.trk --tract_colors random
  skyline_fmri ~/.dipy/bundle_atlas_hcp842/Atlas_80_Bundles/bundles/AF_L.trk --tract_colors "1 0 0"

The color picker in the tractogram controls changes the color afterwards.

**Lines or tubes.** Streamlines are drawn as tubes by default. For large
tractograms, ``--performance_version`` draws them as lines, which is much
faster. The *Line Type* switch in the tractogram controls changes this at
any time, and *Opacity* and *Thickness* adjust the lines.

**Tractograms with anatomy.** The atlas is in MNI space, so it lines up with
the MNI template. ::

  skyline_fmri ~/.dipy/mni_template/mni_icbm152_t1_tal_nlin_asym_09a.nii ~/.dipy/bundle_atlas_hcp842/Atlas_80_Bundles/whole_brain/whole_brain_MNI.trk --cluster

**BUAN p-values.** Pass the p-values file saved by ``dipy_buan_lmm`` with
``--buan_pvals`` to color a bundle by the significance of group differences
along its length. ::

  skyline_fmri AF_L.trk --buan_pvals AF_L_fa_pvalues.npy

The *Hue*, *Saturation* and *Value* sliders tune the colors. *Upload BUAN P
values* in the tractogram controls loads a p-values file into a running
window. See :ref:`buan_flow` for how these files are produced.


-------------------------
Visualize Peaks and ODFs
-------------------------

Peaks and spherical harmonic ODFs come from reconstruction workflows. Here we
use the ``recons_csd_output/peaks.pam5`` file saved by ``dipy_fit_csd`` in
:ref:`reconstruction_flow`.

A ``.pam5`` file given as an input file is shown as peaks. ::

  skyline_fmri recons_csd_output/peaks.pam5

To show its ODF glyphs, pass the same file to ``--odfs``. Use ``run`` as the
input file when you only load files through options. ::

  skyline_fmri run --odfs recons_csd_output/peaks.pam5

Peaks and ODFs can also come from NIfTI files. ``--peaks`` accepts volumes of
shape (X, Y, Z, 3*N) or (X, Y, Z, N, 3), and ``--odfs`` accepts 4D volumes of
SH coefficients, such as the ``peaks_dirs.nii.gz`` and ``shm.nii.gz`` files
that ``dipy_fit_csd --extract_pam_values`` writes. ::

  skyline_fmri run --peaks recons_csd_output/peaks_dirs.nii.gz --odfs recons_csd_output/shm.nii.gz

``--sh_basis`` declares the SH convention of the ODF files:
``descoteaux07`` (the default, current DIPY), ``descoteaux07_legacy`` (older
DIPY), ``tournier07`` (MRtrix 0.2) or ``tournier19`` (MRtrix3). ::

  skyline_fmri run --odfs wm_fod.nii.gz --sh_basis tournier19

Peaks and ODFs have *Synchronize Slices*, *Scale*, *Opacity* and *Slice*
controls, so you can step through them together with an image such as FA.


-----------------------
Visualize Brain Surface
-----------------------

Skyline reads FreeSurfer surfaces (``.pial``) and GIFTI surfaces (``.gii``,
``.gii.gz``).

Using terminal, let's download a brain surface.

**For macOS and Linux** ::

  wget https://github.com/maharshi-gor/dipy_data/raw/surface_data/surfaces/lh.pial

**For Windows (PowerShell)** ::

  wget https://github.com/maharshi-gor/dipy_data/raw/surface_data/surfaces/lh.pial -O lh.pial

**For macOS users**, if you do not have ``wget`` on your terminal you can set
it up with the following command. ::

  brew install wget

If you are still getting an error you can download the surface by clicking
`here <https://github.com/maharshi-gor/dipy_data/raw/surface_data/surfaces/lh.pial>`_.

The previous step downloads the file into your current directory. To load the
surface, ::

  skyline_fmri lh.pial

Each surface has an *Opacity* slider. ``--glass_brain`` draws surfaces as dark
translucent shells on a white background, which is useful for showing
tractograms inside the brain. ::

  skyline_fmri lh.pial --glass_brain


----------------------
Visualize fMRI Data
----------------------

Skyline shows functional data next to structural and diffusion data: BOLD
runs with time navigation, activation maps as thresholded overlays, and the
BOLD time series of any voxel in a panel under the 3D view. Every layer is
placed in world coordinates, so a 2 mm BOLD run, a 1 mm T1, FA maps and
tractograms in the same space line up without resampling.

**Activation maps.** Pass t, z or correlation maps with ``--stat_maps``.
Voxels below the threshold are transparent; positive values use a red to
yellow ramp and negative values a blue to cyan ramp. The overlay is always
drawn above the anatomy. ::

  skyline_fmri T1w.nii.gz --stat_maps zstat1.nii.gz

Each map has these controls:

* *Threshold* and *Max* set the smallest shown magnitude and where colors
  saturate. Maps whose values reach 5 start at 3.1; weaker maps, such as
  correlations, start at 40% of their maximum.
* *Show* picks both tails, only positive or only negative values.
* *Min cluster* hides clusters (26-connected) with fewer voxels.
* *Colormap* offers Cold-Hot, Red-Yellow, Blue-Cyan, Viridis, Plasma and
  Inferno.
* *Clusters* lists the strongest clusters with their size, peak value and peak
  coordinate in mm. Click a row to center the slices on that peak.

A color scale for each visible map is drawn in the corner of the 3D view.

**BOLD runs.** Pass 4D runs with ``--bold``. The repetition time is read from
the BIDS sidecar (``bold.json``) or the NIfTI header; ``--tr`` overrides it,
and the *TR (s)* field in the run's controls changes it later. ::

  skyline_fmri T1w.nii.gz --bold sub-01_task-motor_bold.nii.gz --tr 2.0

The run's controls add a *Volume* slider and a *Show* switch:

* *Volume* shows the raw intensities of the current volume.
* *Change* shows each volume's percent signal change, with slow drift
  removed and a light 1-voxel smoothing, as a hot/cold overlay on top of the
  anatomy. This is the mode to use while playing a run: raw BOLD only changes
  by a few percent, so activity is invisible in *Volume*, but here active
  regions light up and fade with the task. *Show above* and *Max* set the
  transparency threshold and saturation; they start at the 95th and 99.5th
  percentiles of the in-brain changes.
* *Mean* and *tSNR* show the temporal mean and temporal SNR.

**Time series.** Click a slice to pick a voxel: an orange cross marks it and
the panel under the view plots its BOLD signal for every loaded run.
Double-click a slice to also move every synchronized slice through that
point. In the panel:

* The play, previous and next buttons, or Space and the left and right arrow
  keys, move through the run; *fps* sets the playback speed.
* Drag the orange line, or double-click the plot, to jump in time.
* *Scale* shows the raw signal, percent signal change or z-scores, and
  *Detrend* removes a linear trend.
* *Spectrum* switches to the power spectrum.
* *Seed map* correlates every voxel of the run with the series under the
  cursor and shows the result as an activation map named ``Seed r · <run>``.
  Seeding again updates the same map.
* *CSV* saves the plotted series.

**Task events.** Pass a BIDS events table (``onset``, ``duration`` and an
optional ``trial_type`` column, in seconds) with ``--events``, or use *Load
events* in a run's controls. Events are shaded behind the time series and
*Model fit* overlays the fit of the HRF-convolved conditions. *Fit task GLM*
fits every voxel with the canonical double-gamma HRF and a 128 s cosine
high-pass filter, and adds one t-map per condition named
``t · <condition> · <run>``. ::

  skyline_fmri T1w.nii.gz --bold sub-01_task-auditory_bold.nii --events sub-01_task-auditory_events.tsv

**Tracts through activation.** Load a bundle atlas next to an activation map
to see which tracts pass through each cluster. Pass the atlas folder as an
input; every bundle becomes a tractogram named after its file, such as
``AF_L.trk`` (arcuate fasciculus, left). With the ``mni_template`` and
``bundle_atlas_hcp842`` datasets and a group map in MNI space: ::

  skyline_fmri ~/.dipy/mni_template/mni_icbm152_t1_tal_nlin_asym_09c.nii ~/.dipy/bundle_atlas_hcp842/Atlas_30_Bundles/bundles --stat_maps group_zstat.nii.gz --tract_colors random

The activation map's controls then show:

* a *Tracts* column in the cluster table, with the tracts whose streamlines
  cross each cluster (hover it for the full names and streamline counts);
* a *Tracts through activation* table with every crossing tract, how many of
  its streamlines cross the thresholded clusters and what fraction of the
  tract that is. The filter icon adds a layer, named for example
  ``AF_L in group_zstat``, with only the streamlines that cross the
  activation.

Clicking a slice shows a readout in the corner of the view with the cursor
position, the value of each activation map there and the full names of the
tracts passing within 2 mm. The tables follow the current threshold and
minimum cluster size.

The same data can be passed from Python with ``skyline(functional=...,
stat_maps=..., events=..., tr=..., tractograms=...)``.


------------------
2D Triplanar Views
------------------

The layout button next to the camera in the sidebar switches between the 3D
view and a triplanar layout: axial, sagittal and coronal 2D views with the
3D view in the fourth quarter. ``--layout triplanar`` starts in it. ::

  skyline_fmri T1w.nii.gz --stat_maps zstat1.nii.gz --layout triplanar

The 2D views follow the neurological convention (the subject's left on the
left) and label each edge with its direction. Drag with the left button to
pan and use the wheel to zoom. Clicking a 2D view moves every synchronized
slice through the clicked point, like the crosshair of other viewers.
Activation maps are drawn above the anatomy in every view. Tracts, surfaces
and ROIs are cut to a 10 mm slab around each 2D slice, so geometry in front
of the slice does not hide it. Tracts drawn as tubes appear in the 3D view
only; switch their *Line Type* to *Line* to see them in the 2D views too
(``--performance_version`` draws every tractogram as lines).


---------------
Capture Figures
---------------

The camera button opens *Capture Figure*:

* *Resolution* renders at 1x, 2x or 4x the size of the views on screen.
* *Background* keeps the scene color or uses white, black or a transparent
  background (*Clear*).
* *Color scale* draws a labelled color bar for each visible activation map
  and BOLD *Change* overlay.
* *Cursor* keeps or removes the orange cursor cross.
* *Separate views* (triplanar layout) also saves each view as its own file,
  ``figure_axial.png``, ``figure_sagittal.png``, ``figure_coronal.png`` and
  ``figure_3d.png``.

The views are rendered again offscreen, so the sidebar and the time-series
panel are not in the image. From Python, ``viewer.capture_figure("fig.png",
scale=4, background="white", panels=True)`` does the same.



-----------------------------
Save Figures Without a Window
-----------------------------

Use ``--stealth`` to render the scene offscreen and save it as a PNG instead
of opening a window. ``--out_dir`` and ``--out_stealth_png`` choose where the
image goes. ::

  skyline_fmri ~/.dipy/mni_template/mni_icbm152_t1_tal_nlin_asym_09a.nii --stealth --out_dir figures --out_stealth_png t1.png

Setting the environment variable ``DIPY_OFFSCREEN`` to ``1`` or ``true`` has
the same effect without changing the command, which is convenient on servers
and in scripts.

``--bg_color`` sets the background as three numbers between 0 and 1, for
example ``--bg_color 1 1 1`` for white. In an open window, the *Take Snapshot*
button saves the current view.
