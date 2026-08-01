# Written by Alvin Yadav

"""
AEM Source Designer: Solver adapter.

Bridges the designer model to ATSimulation. Exposes solve_scene (blocking),
collect_artifacts (locate output files), and bundle_results (zip for download).
"""
from __future__ import annotations

import json
import multiprocessing
import os
import sys
import tempfile
import time
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

from at_config import ATConfiguration
from at_simulation import ATSimulation

# Restore the "fork" multiprocessing start method on macOS/Linux.
#
# at_simulation.conc_array parallelises the grid evaluation with a Pool and was
# written for the fork context ("fast worker startup, shared memory with the
# parent" — see its docstring). Python 3.8+ silently changed the macOS default
# to "spawn", which re-imports the entire module stack (matplotlib included) in
# every worker on every solve. For the small grids this tool produces that
# overhead dominates — a one-element solve spent ~2.6 s of ~3.8 s just spawning
# workers. Forcing fork drops that to ~0.06 s and restores the intended speed.
if sys.platform != "win32":
    try:
        multiprocessing.set_start_method("fork", force=True)
    except (RuntimeError, ValueError):
        pass


@dataclass
class SolveResult:
    xaxis: object
    yaxis: object
    result: object
    L_max: object
    interface_length: object
    run_dir: str
    artifacts: dict
    config_dict: dict
    wall_time: float = 0.0


def solve_scene(scene) -> SolveResult:
    """
    Run the full AEM simulation for the given Scene.

    Converts the Scene to a config dict, passes it directly to
    ATConfiguration.from_dict (no temp file needed), runs the simulation,
    and returns a SolveResult with all post-run attributes populated.

    Call this from a background thread — sim.run() is CPU-intensive.
    """
    t0 = time.time()
    cfg_dict = scene.to_config_dict()
    config = ATConfiguration.from_dict(cfg_dict)
    sim = ATSimulation(config)
    sim.run()
    wall_time = time.time() - t0
    return SolveResult(
        xaxis=sim.xaxis,
        yaxis=sim.yaxis,
        result=sim.result,
        L_max=sim.L_max,
        interface_length=sim.interface_length,
        run_dir=sim.run_dir,
        artifacts=collect_artifacts(sim.run_dir),
        config_dict=cfg_dict,
        wall_time=wall_time,
    )


def collect_artifacts(run_dir: str) -> dict[str, str | None]:
    """
    Locate the artifacts that sim.run() wrote under run_dir.

    Returns a dict mapping role → absolute path (or None if missing).
    Uses glob so filenames are robust to suffix changes.
    """
    d = Path(run_dir)

    def first(pattern: str) -> str | None:
        hits = sorted(d.glob(pattern))
        return str(hits[0]) if hits else None

    return {
        "result_plot": first("*_plot_*.pdf"),
        "input_plot":  first("*_input_*.pdf"),
        "error_plot":  first("*_error_*.pdf"),
        "stats_file":  first("*.txt"),
    }


def bundle_results(run_dir: str, scene_config: dict | None = None) -> str:
    """
    Zip all artifacts from run_dir (plus the config JSON) into a single file.

    Returns the path to the created .zip in the system temp directory.
    Calling this twice with the same run_dir produces the same path and
    overwrites the previous zip — re-downloading is always safe.
    """
    arts = collect_artifacts(run_dir)
    zip_path = os.path.join(
        tempfile.gettempdir(),
        f"aem_results_{Path(run_dir).name}.zip",
    )
    missing: list[str] = []
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for role, p in arts.items():
            if p and os.path.exists(p):
                z.write(p, arcname=os.path.basename(p))
            else:
                missing.append(role)
        if scene_config is not None:
            z.writestr(
                "simulation_config.json",
                json.dumps(scene_config, indent=4),
            )
        if missing:
            z.writestr(
                "MISSING.txt",
                "The following artifacts were not found:\n" + "\n".join(missing),
            )
    return zip_path
