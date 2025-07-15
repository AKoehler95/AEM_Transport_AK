# Written by Willi Kappler, willi.kappler@uni-tuebingen.de
# Based on code from Anton Köhler

# Python std library
import logging

# Local imports
import at_config
import at_simulation

def main():
    # Set up logging
    log_file_name = "aem_transport_simulation.log"
    log_format = "%(asctime)s - %(levelname)s - %(name)s - %(message)s"
    logging.basicConfig(filename=log_file_name, level=logging.DEBUG, format=log_format)

    # Load configuration
    filename = "simulation_config.json"
    config = at_config.ATConfiguration.from_json(filename)
    inc = config.dom_inc

    sim = at_simulation.ATSimulation(config)
    sim.run()
    result = sim.result_tuple

    return result

if __name__ == "__main__":
    result = main()

