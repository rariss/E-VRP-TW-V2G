import time
import sys
import os
import logging.config
import traceback
import json

import pandas as pd

from multiprocessing import Pool
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
    try:
        fpath = f'{DIR_INSTANCES}/{instance}.csv'

        m = EVRPTWV2G(problem_type=problem_type, dist_type=dist_type)
        m.full_solve(fpath)

        print(m.results)

        x, xp, traces, routes = plot_evrptwv2g(m, save=True, save_folder=save_folder)

        m_stats = generate_stats(m)  # For parallelizing, need to generate stats so model with <locals> not returned (fails to pickle for pool)

        return m, m_stats, x, xp, traces, routes
    except:
        logging.info(f'Failed running {instance} {problem_type} {dist_type}')
        logging.error(traceback.format_exc())
        return None

def main(fpath_instances: str, n_processes: int=2):
    """
    Runs multiple instances in parallel and outputs results in new columns for instances CSV
    :param fpath_instances: filepath to instances CSV
    :param n_processes: number of parallel processes
    :return: updates instances CSV with appended results
    """
    logging.info(f'Reading instances from: {fpath_instances}')
    # Read in CSV of instances to run in parallel
    instances = pd.read_csv(fpath_instances,
                            dtype={'dir': str, 'instance': str, 'problem_type': str, 'dist_type': str},
                            na_filter=False)
    instances = instances.set_index(instances['dir'] + '_' + instances['instance'] + '_' +
                                    instances['problem_type'] + '_' + instances['dist_type'])

    # Filter to just instances to run
    run_instances = instances[instances['run']]  # == True

    # Create a pooled multiprocess
    pool = Pool(processes=n_processes)
    logging.info(f'Created a pool with {n_processes} process to run {len(run_instances)} instances')

    pool_results = pool.starmap(run_evrptwv2g,
                           list(zip(run_instances['dir']+'/'+run_instances['instance'],
                                    run_instances['problem_type'], run_instances['dist_type'])))
    pool.close()
    pool.join()

    # Iterate through scenarios
    logging.info(f'Compiling results for {len(run_instances)} instances')

    results = {}
    stats_results = []
    for i, (name, instance) in enumerate(run_instances.iterrows()):
        logging.info(f"{i} Compiling {name}")
        tic = time.time()
        try:
            m = pool_results[i][0]
            results[name] = {
                'm_stats': pool_results[i][1],
                'x': pool_results[i][2],
                'xp': pool_results[i][3],
                'traces': pool_results[i][4],
                'routes': pool_results[i][5]
            }

            stats_results.append({
                'instance': name,
                **pool_results[i][1]
            })

            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

            # Make results output folder
            results_folder = f'{DIR_OUTPUT}/results/{timestamp}_{name}'
            Path(results_folder).mkdir(parents=True, exist_ok=True)

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

            logging.info(f"({round(time.time() - tic, 4)} s) Success")
        except:
            logging.warning(f"({round(time.time() - tic, 4)} s) Failed")
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

    if len(sys.argv) > 2:
        n_processes = int(sys.argv[2])
    else:
        n_processes = 2

    main(fpath_instances, n_processes)
