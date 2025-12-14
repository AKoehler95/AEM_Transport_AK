# Written by Willi Kappler, willi.kappler@uni-tuebingen.de
# Based on code from Anton Köhler

# Python std library
import logging

# Local imports
from at_config import ATConfiguration
from at_simulation import ATSimulation
from at_inverse_model import process_input_file_with_logging

# Choose mode: "simulate" to run transport sim, "inverse" to run alpha-finder
MODE = "inverse"  # "simulate" or "inverse"

# When MODE == "simulate":
CONFIG_PATH = "simulation_config.json"

# When MODE == "inverse":
INPUT_FILE       = "inverse_model_files/input_values_new.txt"
OUTPUT_FILE      = "inverse_model_files/output_values_new.txt"
STATS_FILE       = "stats_values.txt"
ORIENTATION      = "vertical"   # "horizontal" or "vertical"
ELEM_KIND        = "Circle"       #Circle or Line
SOURCE_THICKNESS_MODIFIER = 0.5   # set between 0.0 and 1.0 to determine what percentage of
                                   # aquifer thickness is radius of source thickness

# New: which parameter to estimate and optional bounds
# Options: "alpha_t", "alpha_l", "r", "C0", "ca", "gamma"
ESTIMATE_PARAM   = "r"

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
    sim = ATSimulation(config)
    sim.run()

    # Return or process sim.result_tuple as needed
    return sim.result_tuple

def run_inverse(
    input_file,
    output_file,
    stats_file,
    orientation,
    elem_kind,
    source_thickness_modifier,
    estimate_param,
):
    # Run the parameter-finder in batch mode with logging
    process_input_file_with_logging(
        input_file=input_file,
        output_file=output_file,
        statsfile=stats_file,
        orientation=orientation,
        elem_kind=elem_kind,
        source_thickness_modifier=source_thickness_modifier,
        estimate_param=estimate_param,
    )

def main():
    global result
    setup_logging()

    if MODE == "simulate":
        result = run_simulation()
        print("Simulation completed. Result tuple returned.")
    elif MODE == "inverse":
        run_inverse(
            input_file=INPUT_FILE,
            output_file=OUTPUT_FILE,
            stats_file=STATS_FILE,
            orientation=ORIENTATION,
            elem_kind=ELEM_KIND,
            source_thickness_modifier=SOURCE_THICKNESS_MODIFIER,
            estimate_param=ESTIMATE_PARAM,
        )
        print(f"Inverse Dispersivity Finder completed. Results in '{OUTPUT_FILE}'. Element Statistics in '{STATS_FILE}'. Console output logged to 'findalpha_log.txt'.")
    else:
        raise ValueError(f"Unknown MODE '{MODE}'. Use 'simulate' or 'inverse'.")


if __name__ == "__main__":
    main()