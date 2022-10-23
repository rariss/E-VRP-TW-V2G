import time
import sys
import os
import logging.config
import traceback
import json

import pandas as pd

from datetime import datetime
from pathlib import Path
from milp.evrptwv2g_base import EVRPTWV2G
from utils.utilities import generate_stats, create_json_inputs, create_json_out, NpEncoder
from utils.plot import plot_evrptwv2g
from config.LOCAL_CONFIG import DIR_INSTANCES, DIR_OUTPUT

_HERE = os.path.dirname(__file__)
_CONFIG = os.path.abspath(os.path.join(_HERE, 'config/loggingconfig.ini'))
logging.config.fileConfig(_CONFIG)


def run_evrptwv2g(instance: str, problem_type: str, dist_type: str, save_folder: str=None):
    """
    Allows us to call the script from the command line
    :param process: filename of main process
    """
    fpath = f'{DIR_INSTANCES}/{instance}.csv'

    m = EVRPTWV2G(problem_type=problem_type, dist_type=dist_type)
    m.full_solve(fpath)

    print(m.results)

    x, xp, traces, routes = plot_evrptwv2g(m, save=True, save_folder=save_folder, add_basemap=True)

    m_stats = generate_stats(m)

    return m, m_stats, x, xp, traces, routes


def main(fpath_instances: str):
    """
    Runs multiple instances in sequence and outputs results in new columns for instances CSV
    :param fpath_instances: filepath to instances CSV
    :return: updates instances CSV with appended results
    """
    logging.info(f'Reading instances from: {fpath_instances}')
    # Read in CSV of instances to run in parallel
    instances = pd.read_csv(fpath_instances,
                            dtype={'dir': str, 'instance': str, 'problem_type': str, 'dist_type': str},
                            na_filter=False)
    dist_type = []
    for instance_dist_type in instances['dist_type']:
        if '.csv' in instance_dist_type:
            dist_type.append('csv')
        else:
            dist_type.append(instance_dist_type)
    dist_type = pd.Series(dist_type)
    instances = instances.set_index(instances['dir']+'_'+instances['instance']+'_'+instances['problem_type']+'_'+dist_type)

    # Filter to just instances to run
    run_instances = instances[instances['run']]  # == True

    # Iterate through scenarios
    logging.info(f'Running a batch of {len(run_instances)} instances')

    results = {}
    stats_results = []
    for name, instance in run_instances.iterrows():
        logging.info(f"Running {name}")
        tic = time.time()
        try:
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

            # Make results output folder
            results_folder = f'{DIR_OUTPUT}/results/{timestamp}_{name}'
            Path(results_folder).mkdir(parents=True, exist_ok=True)

            m, m_stats, x, xp, traces, routes = run_evrptwv2g(f"{instance['dir']}/{instance['instance']}",
                                                              instance['problem_type'],
                                                              instance['dist_type'],
                                                              save_folder=results_folder)
            results[name] = {
                'm_stats': m_stats,
                'x': x,
                'xp': xp,
                'traces': traces,
                'routes': routes
            }

            stats_results.append({
                'instance': name,
                'results_folder': results_folder,
                **m_stats
            })

            # Save model inputs/outputs
            create_json_inputs(m.p, f'{results_folder}/inputs')
            create_json_out(m.instance, m.results, f'{results_folder}/output')

            # Save stats and plot results
            for key, result in results[name].items():
                with open(f'{results_folder}/{key}.json', 'w') as fp:
                    if isinstance(result, pd.DataFrame):
                        json.dump(result.to_json(), fp)
                    else:
                        json.dump(result, fp, cls=NpEncoder, sort_keys=True, indent=4, separators=(',', ': '))

            logging.info(f"Success after {round(time.time()-tic, 4)} s")
        except:
            logging.warning(f"Failed after {round(time.time()-tic, 4)} s")
            logging.error(traceback.format_exc())

    stats = pd.DataFrame(stats_results).set_index('instance')  # generate dataframe from list of stats dicts

    instances_stats = instances.merge(stats, how='outer', left_index=True, right_index=True)  # combine stats dataframes into original instances dataframe for output

    # Export instances results to CSV
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    instances_stats.to_csv(f'{os.path.dirname(fpath_instances)}/{timestamp}_{os.path.basename(fpath_instances)}')
    logging.info("Output instance statistics to CSV")
    logging.info("Done")

    return instances


if __name__ == "__main__":
    fpath_instances = sys.argv[1]  # filepath to instances CSV

    main(fpath_instances)
