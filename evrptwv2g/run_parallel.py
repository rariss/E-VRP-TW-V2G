import time
import sys
import os
import logging.config

import pandas as pd

from multiprocessing import Pool
from datetime import datetime
from milp.evrptwv2g_base import EVRPTWV2G
from utils.utilities import generate_stats
from utils.plot import plot_evrptwv2g
from config.LOCAL_CONFIG import DIR_INSTANCES

_HERE = os.path.dirname(__file__)
_CONFIG = os.path.abspath(os.path.join(_HERE, 'config/loggingconfig.ini'))
logging.config.fileConfig(_CONFIG)

def run_evrptwv2g(instance: str, problem_type: str, dist_type: str):
    """
    Allows us to call the script from the command line
    :param process: filename of main process
    """
    fpath = f'{DIR_INSTANCES}/{instance}.csv'

    m = EVRPTWV2G(problem_type=problem_type, dist_type=dist_type)
    m.full_solve(fpath)

    print(m.results)

    x, xp, traces, routes = plot_evrptwv2g(m, save=True)

    m_stats = generate_stats(m)  # For parallelizing, need to generate stats so model with <locals> not returned (fails to pickle for pool)

    return m_stats, x, xp, traces, routes

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
    run_instances = instances[instances['run'] == True]

    # Create a pooled multiprocess
    pool = Pool(processes=n_processes)
    logging.info(f'Created a pool with {n_processes} process to run {len(run_instances)} instances')

    results = pool.starmap(run_evrptwv2g,
                           list(zip(run_instances['dir']+'/'+run_instances['instance'],
                                    run_instances['problem_type'], run_instances['dist_type'])))
    pool.close()
    pool.join()

    # Generate statistics
    stats_dicts = []
    for i, r in enumerate(results):
        d = {'instance': run_instances.index[i]}  # unique instance name
        stat = r[0]  # dict of statistics given model
        d.update(stat)
        stats_dicts.append(d)  # append dict of instance stats to list

    stats = pd.DataFrame(stats_dicts).set_index('instance')  # generate dataframe from list of stats dicts

    instances_stats = instances.merge(stats, how='outer', left_index=True, right_index=True)  # combine stats dataframes into original instances dataframe for output

    # Export instances results to CSV
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    instances_stats.to_csv(f'{os.path.dirname(fpath_instances)}/{timestamp}_{os.path.basename(fpath_instances)}')

    return instances


if __name__ == "__main__":
    fpath_instances = sys.argv[1]  # filepath to instances CSV

    if len(sys.argv) > 1:
        n_processes = int(sys.argv[2])
    else:
        n_processes = 2

    main(fpath_instances, n_processes)
