# Written by Alvin Yadav
# Based on code from Willi Kappler and Anton Köhler

# Python std library
import argparse
import json
import logging
import os

# Choose mode: "simulate" to run transport sim, "inverse" to run alpha-finder
MODE = "simulate"  # "simulate" or "inverse"

# When MODE == "simulate": forward model's exclusive config
CONFIG_PATH = "simulation_config.json"

# When MODE == "inverse": inverse model's exclusive config.
# All inverse run-level options, search bounds, fixed dispersivities and
# domain / numerical settings live in this file (the inverse model no longer
# reads simulation_config.json). The per-case inputs come from the input file
# referenced inside inverse_config.json ("input_file").
INVERSE_CONFIG_PATH = "inverse_config.json"


def _wants_interactive_plots(path):
    """
    Peek at "show_plots" in the config without building an ATConfiguration.

    Needed because matplotlib's backend has to be fixed before pyplot is
    imported, and importing at_config/at_simulation to read the flag would
    import pyplot first. A missing or malformed config isn't reported here —
    the real loader below raises with a better message.
    """
    try:
        with open(path) as f:
            return bool(json.load(f).get("show_plots", False))
    except (OSError, json.JSONDecodeError):
        return False


# Backend selection, which must happen before the local imports below: they
# pull in pyplot transitively (at_simulation directly, at_inverse_model via
# at_simulation), and at_simulation forces the non-interactive Agg backend
# unless AEM_INTERACTIVE is set. Translating "show_plots" into that variable
# here means the config flag alone is enough — no second switch to remember.
# setdefault so an explicit AEM_INTERACTIVE=0 from the environment still wins.
if MODE == "simulate" and _wants_interactive_plots(CONFIG_PATH):
    os.environ.setdefault("AEM_INTERACTIVE", "1")

# Local imports (deliberately after the backend decision above):
from at_config import ATConfiguration                            # noqa: E402
from at_simulation import ATSimulation                           # noqa: E402
from at_inverse_config import InverseConfiguration               # noqa: E402
from at_inverse_model import process_input_file_with_logging     # noqa: E402

result = None

def setup_logging():
    log_file = "aem_transport_simulation.log"
    log_format = "%(asctime)s - %(levelname)s - %(name)s - %(message)s"
    logging.basicConfig(filename=log_file,
                        level=logging.DEBUG,
                        format=log_format)


def run_simulation(export_overrides=None):
    # Load configuration and run the full transport simulation
    config = ATConfiguration.from_json(CONFIG_PATH)

    # CLI overrides for the grid export, applied on top of the config file so a
    # one-off run can request an export (or a different destination/format)
    # without editing the JSON. None means "leave the config value alone".
    if export_overrides:
        if export_overrides.get("concentration_output") is not None:
            config.concentration_output = export_overrides["concentration_output"]
        if export_overrides.get("export_step") is not None:
            config.export_step = export_overrides["export_step"]
        if export_overrides.get("export_path") is not None:
            config.export_path = export_overrides["export_path"]
        if export_overrides.get("export_npz_float32") is not None:
            config.export_npz_float32 = export_overrides["export_npz_float32"]

    # Optional: size the domain from the learned predictor instead of the
    # config's dom_xmax. Only saves the cost of a domain-extension pass —
    # the extension loop in ATSimulation.run() still guarantees the plume
    # isn't truncated, and an untrained/out-of-range model silently falls
    # back to the legacy "furthest source point + 150" rule.
    if config.use_xmax_predictor:
        from xmax_predictor import suggest_dom_xmax
        suggested = suggest_dom_xmax(config)
        print(f"xmax predictor: dom_xmax {config.dom_xmax:.0f} -> {suggested:.0f} m")
        config.dom_xmax = suggested

    sim = ATSimulation(config)
    sim.run()

    # Return or process sim.result_tuple as needed
    return sim.result_tuple

def run_at_inverse(inv_cfg):
    # Run the parameter-finder in batch mode with logging
    process_input_file_with_logging(inv_cfg)

def parse_args(argv=None):
    """
    Parse the optional CLI overrides. With no arguments the returned values
    are all None, so behaviour is identical to running without argparse.
    """
    parser = argparse.ArgumentParser(
        description="Run the AEM transport simulation (see MODE / config paths "
                    "at the top of this file for the rest of the configuration).")
    parser.add_argument("--format", dest="concentration_output", default=None,
                        choices=["none", "csv", "npz", "both"],
                        help="Which grid file(s) to write after the run "
                             "(overrides concentration_output in the config).")
    parser.add_argument("--step", dest="export_step", type=int, default=None,
                        metavar="N",
                        help="Keep every N-th grid point on both axes (1 = all). "
                             "Implies --format both if no format is given.")
    parser.add_argument("--path", dest="export_path", type=str, default=None,
                        metavar="STEM",
                        help="Output path stem (no extension; .csv/.npz are added). "
                             "Default is 'grid' inside the run directory. "
                             "Implies --format both if no format is given.")
    parser.add_argument("--npz-float32", dest="export_npz_float32", action="store_true",
                        default=None,
                        help="Store the .npz concentration as float32 (~half the "
                             "bytes). Implies an npz export if no format is given.")
    args = parser.parse_args(argv)

    # A step/path/float32 flag with no explicit --format only makes sense with
    # an export on, so pick a sensible default rather than silently ignoring
    # them: --npz-float32 implies npz, the others imply both.
    if args.concentration_output is None:
        if args.export_npz_float32 is not None:
            args.concentration_output = "npz"
        elif args.export_step is not None or args.export_path is not None:
            args.concentration_output = "both"
    return args


def main():
    global result
    setup_logging()

    args = parse_args()
    export_overrides = {
        "concentration_output": args.concentration_output,
        "export_step": args.export_step,
        "export_path": args.export_path,
        "export_npz_float32": args.export_npz_float32,
    }

    if MODE == "simulate":
        result = run_simulation(export_overrides)
        print("Simulation completed. Result tuple returned.")
    elif MODE == "inverse":
        inv_cfg = InverseConfiguration.from_json(INVERSE_CONFIG_PATH)
        run_at_inverse(inv_cfg)
        print(f"Inverse Dispersivity Finder completed. Results in '{inv_cfg.output_file}'. Element Statistics in '{inv_cfg.stats_file}'. Console output logged to 'findalpha_log.txt'.")
    else:
        raise ValueError(f"Unknown MODE '{MODE}'. Use 'simulate' or 'inverse'.")


if __name__ == "__main__":
    main()
