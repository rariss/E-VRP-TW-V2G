import sys
import os
import logging.config

from milp.evrptwv2g_base import EVRPTWV2G
from utils.plot import plot_evrptwv2g
from config.LOCAL_CONFIG import DIR_INSTANCES

_HERE = os.path.dirname(__file__)
_CONFIG = os.path.abspath(os.path.join(_HERE, 'config/loggingconfig.ini'))
logging.config.fileConfig(_CONFIG)


def main(instance: str, problem_type: str, dist_type: str):
    """ Run a full solve """
    fpath = f'{DIR_INSTANCES}/{instance}.csv'

    m = EVRPTWV2G(problem_type=problem_type, dist_type=dist_type)
    m.full_solve(fpath)

    print(m.results)

    x, xp, traces, routes = plot_evrptwv2g(m, save=True)


if __name__ == "__main__":
    instance = sys.argv[1]
    problem_type = sys.argv[2]

    if len(sys.argv) > 3:
        dist_type = sys.argv[3]
    else:
        dist_type = 'scipy'

    main(instance, problem_type, dist_type)
