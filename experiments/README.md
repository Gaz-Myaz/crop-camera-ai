# experiments

Bench hardware bring-up work: probes, calibration rigs and the reverse-engineering
notes that established how a particular camera behaves. Kept because the findings
are hard-won and not documented anywhere else, but not part of the running stack —
nothing here is on the critical path of a deployment.

| Folder | What it established |
|---|---|
| `decxin-sm-2930v1/` | A cheap USB board turned out to be a binocular stereo camera; probed frame format, resolutions, 64.8 mm baseline, and a disparity-offset correction that brings measured distances within 1.5% of truth. |

Scripts here reach `../common` through a `sys.path` insert, so they still run
directly from this directory.
