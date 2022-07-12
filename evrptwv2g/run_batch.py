import time
import sys
import os
import logging.config

import pandas as pd

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

    m_stats = generate_stats(m)

    return m_stats, x, xp, traces, routes

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
    instances = instances.set_index(instances['dir'] + '_' + instances['instance'] + '_' +
                                    instances['problem_type'] + '_' + instances['dist_type'])

    # Filter to just instances to run
    run_instances = instances[instances['run']]  # == True

    # Iterate through scenarios
    logging.info(f'Running a batch of {len(run_instances)} instances')

    results = {}
    for instance in run_instances:
        # TODO: Try/Catch for error handling
        m_stats, x, xp, traces, routes = run_evrptwv2g(instance['dir']+'/'+instance['instance'],
                                                       instance['problem_type'],
                                                       instance['dist_type'])
        results[instance.index] = {
            'm_stats': m_stats,
            'x': x,
            'xp': xp,
            'traces': traces,
            'routes': routes
        }

        # TODO: Pickle results

    # Generate statistics
    stats_dicts = []
    for key, result in enumerate(results):
        d = {'instance': key}  # unique instance name
        d.update(result['m_stats'])
        stats_dicts.append(d)  # append dict of instance stats to list

    stats = pd.DataFrame(stats_dicts).set_index('instance')  # generate dataframe from list of stats dicts

    instances_stats = instances.merge(stats, how='outer', left_index=True, right_index=True)  # combine stats dataframes into original instances dataframe for output

    # Export instances results to CSV
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    instances_stats.to_csv(f'{os.path.dirname(fpath_instances)}/{timestamp}_{os.path.basename(fpath_instances)}')

    return instances


if __name__ == "__main__":
    fpath_instances = sys.argv[1]  # filepath to instances CSV

    main(fpath_instances)
