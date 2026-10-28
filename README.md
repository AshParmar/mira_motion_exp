# Motion Experiment

This project detects motion in a selected region of a recorded video. It compares
the current ROI with the ROI from a short time earlier, measures the fraction of
changed pixels, and reports `MOVING`/`STILL` events using hysteresis.

The script does not record video or publish MQTT messages. It writes analysis
files to the output directory so that thresholds can be tuned from real data.

## Requirements

- Python 3.10 or newer
- OpenCV
- NumPy
- Matplotlib (required for the plot output)

Install the dependencies from PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install opencv-python numpy matplotlib
```

If PowerShell prevents activation, run the script with the virtual environment
Python directly:

```powershell
.\.venv\Scripts\python.exe motion_experiment.py .\clip.mp4
```

## Basic usage

Use a video file as the first argument. Without `--roi`, the script opens a
window where you can click and drag the region of interest, then press Enter or
Space:

```powershell
python motion_experiment.py .\clip.mp4
```

To skip the ROI selection window, provide `x,y,width,height` in original video
pixels:

```powershell
python motion_experiment.py .\clip.mp4 --roi 400,300,600,200
```

The script creates an `out` directory containing:

- `<video>_samples.csv` - timestamped motion measurements
- `<video>_plot.png` - changed-pixel fraction and estimated horizontal shift
- `<video>_annotated.mp4` - optional annotated video when `--annotate` is used

Example:

```powershell
python motion_experiment.py .\clip.mp4 `
  --roi 400,300,600,200 `
  --annotate `
  --out .\out
```

## Finding an ROI

Save a frame with a coordinate grid:

```powershell
python motion_experiment.py .\clip.mp4 --grid-frame --roi-frame 100
```

Or let the script suggest a region around areas that change throughout the
video:

```powershell
python motion_experiment.py .\clip.mp4 --suggest-roi
```

The suggested ROI should be checked manually, especially if other objects or
camera movement are present.

## Motion thresholds

The default settings are:

| Option | Default | Meaning |
| --- | ---: | --- |
| `--diff-thresh` | `18` | Minimum grayscale change for a pixel to count as changed |
| `--on` | `0.04` | More than 4% changed ROI pixels starts motion detection |
| `--confirm` | `3` | Consecutive samples required before reporting `MOVING` |
| `--off` | `0.01` | Below 1% changed ROI pixels counts as inactive |
| `--hold` | `2.0` | Seconds below `--off` before reporting `STILL` |
| `--lag` | `0.8` | How far back the reference frame is taken |
| `--sample-hz` | `10.0` | Analysis sampling rate |

For a more sensitive detector:

```powershell
python motion_experiment.py .\clip.mp4 --diff-thresh 10 --on 0.02 --confirm 2
```

For a less sensitive detector:

```powershell
python motion_experiment.py .\clip.mp4 --diff-thresh 25 --on 0.08 --confirm 5
```

The `--on` and `--off` values are fractions, not percentages. For example,
`0.04` means 4% of the ROI pixels.

## Output interpretation

The CSV columns are:

- `ts`: frame timestamp in seconds
- `changed_frac`: fraction of ROI pixels that changed
- `shift_px_s`: estimated coherent horizontal movement in original pixels/second
- `response`: phase-correlation confidence

The console prints events in this form:

```text
MOVING event_ts=  12.30s  decided_at=  12.50s
STILL  event_ts=  18.10s  decided_at=  20.10s
```

`event_ts` is back-dated to when the movement likely started or stopped;
`decided_at` is when enough samples were available to confirm it.

## License

No license has been specified yet.
