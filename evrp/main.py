import sys

from milp.evrptw import EVRPTW
from utils.plot import plot_evrptwv2g


def main(instance, problem_type):
    """ Run a full solve """
    fpath = 'config/test_instances/' + instance + '.csv'

    m = EVRPTW(problem_type=problem_type)
    m.full_solve(fpath)

    print(m.results)

    x, xp, traces, routes = plot_evrptwv2g(m, save=True)


if __name__ == "__main__":
    instance = sys.argv[1]
    problem_type = sys.argv[2]

    main(instance, problem_type)
