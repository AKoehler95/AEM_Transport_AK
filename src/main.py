# Written by Willi Kappler, willi.kappler@uni-tuebingen.de
# Based on code from Anton Köhler

# Python std library
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


def run_simulation():
    # Load configuration and run the full transport simulation
    config = ATConfiguration.from_json(CONFIG_PATH)

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

def main():
    global result
    setup_logging()

    if MODE == "simulate":
        result = run_simulation()
        print("Simulation completed. Result tuple returned.")
    elif MODE == "inverse":
        inv_cfg = InverseConfiguration.from_json(INVERSE_CONFIG_PATH)
        run_at_inverse(inv_cfg)
        print(f"Inverse Dispersivity Finder completed. Results in '{inv_cfg.output_file}'. Element Statistics in '{inv_cfg.stats_file}'. Console output logged to 'findalpha_log.txt'.")
    else:
        raise ValueError(f"Unknown MODE '{MODE}'. Use 'simulate' or 'inverse'.")


if __name__ == "__main__":
    main()
