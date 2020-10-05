from pyomo.environ import *
from evrp.config.default_params import create_params
import numpy as np
import logging

#TODO: move changes to model.py

class TSP:
    """ Builds pyomo Mixed Integer Linear Program (MILP) model for exact-form solution of Traveling Salesman Problem(TSP).\n
    Author: Rami Ariss
    """

    def __init__(self, instance_filepath: str):
        self.instance_name = instance_filepath.split('/')[-1].split('.')[0]
        logging.info('Building TSP MILP for instance: {}'.format(self.instance_name))

        # Create parameters
        self.params = create_params(instance_filepath)

        # Instantiate pyomo Abstract Model
        self.m = AbstractModel()
        logging.info('Building model')
        self.build_model()

    def build_model(self):
        logging.info('Defining parameters and sets')
        self.define_parameters_and_sets()
        logging.info('Defining variables')
        self.define_variables()
        logging.info('Defining constraints')
        self.define_constraints()
        logging.info('Defining objective')
        self.define_objective()
        # logging.info('Constructing model')
        # self.m.construct()
        logging.info('Done building model')

    def define_parameters_and_sets(self):
        # Defining sets
        self.m.V01 = Set(doc='Graph nodes')
        self.m.V = Set(doc='Graph nodes')
        self.m.V0 = Set(doc='Graph nodes without terminal')
        self.m.V1 = Set(doc='Graph nodes without start')
        self.m.E = Set(within=self.m.V01 * self.m.V01, doc='Graph edges')

        # Defining fixed parameters
        self.m.d = Param(self.m.E)

    def define_variables(self):
        # Defining variables
        self.m.xgamma = Var(self.m.E, within=Boolean)

        # Dummy variable ui
        self.m.u = Var(self.m.V1, within=NonNegativeIntegers, bounds=(0, 15 - 1))

    def define_constraints(self):
        # Defining routing constraints

        # Defining constraint functions and dependent functions
        def constraint_in_arcs(m, j):
            return sum([self.m.xgamma[i, j] for i in self.m.V01 - j]) == 1

        # Create constraint
        self.m.constraint_in_arcs = Constraint(self.m.V01, rule=constraint_in_arcs)

        def constraint_out_arcs(m, i):
            return sum([self.m.xgamma[i, j] for j in self.m.V01 - i]) == 1

        # Create constraint
        self.m.constraint_out_arcs = Constraint(self.m.V01, rule=constraint_out_arcs)

        def constraint_single_subtour(m, i, j):
            if i != j:
                return self.m.u[i] - self.m.u[j] + self.m.xgamma[i, j] * 15 <= 15 - 1
            else:
                return Constraint.Skip

        self.m.constraint_single_subtour = Constraint(self.m.V1, self.m.V1, rule=constraint_single_subtour)

    def define_objective(self):
        # Defining objective function and dependent functions
        def objective_distance(m=self.m):
            """Objective: Calculate net amortized profit across fleet"""
            return sum(m.xgamma[e] * m.d[e] for e in m.E)

        # Create objective function in Abstract
        self.m.obj = Objective(rule=objective_distance, sense=minimize)

    def solve(self):
        # Specify solver
        opt = SolverFactory('gurobi') #, solver_io="python"

        # Create an instance of the AbstractModel passing in parameters
        logging.info('Creating instance')
        if self.m.is_constructed():
            instance = self.m
        else:
            instance = self.m.create_instance(self.params)

        # Solver options
        solver_options = {'threads': 4}  # 'keep': True, 'LocRes': True, 'results': True,

        # Solve instance
        logging.info('Solving instance...')
        results = opt.solve(instance, tee=True, options=solver_options)
        logging.info('Done')
        # instance.pprint()
        return results

        # solve()
