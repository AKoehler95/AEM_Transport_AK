# Written by Willi Kappler, willi.kappler@uni-tuebingen.de
# Based on code from Anton Köhler

# Python std library
import logging

# Local imports
from at_config import ATConfiguration
from at_simulation import ATSimulation
from at_inverse_config import InverseConfiguration
from at_inverse_model import process_input_file_with_logging

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
