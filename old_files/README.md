# Archived workflows

These files are retained for reference and reproducibility but are not part of
the current fixed-plane green-ball catcher.

- `rolling/` contains the rolling-ball, table-plane, and red-can experiments.
- `red_ball/` contains the earlier red-ball tracking, pickup, catching, and
  Viam Vision-service workflows.
- `tests/` contains tests for those archived workflows.

Run archived modules from the repository root using their full module path, for
example:

```bash
python -m old_files.red_ball.track_red
python -m old_files.rolling.rolling_preview
```

Run the archived tests separately:

```bash
python -m unittest discover -s old_files/tests -v
```

Shared helpers still needed by the active catcher or calibration tools were
extracted into `motion/`; archived modules import those maintained helpers.
