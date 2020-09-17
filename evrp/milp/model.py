from pyomo.environ import *
# from utils.utilities import
import pandas as pd
import numpy as np


class MILP:
    """ Electric Vehicle Routing Problem with Time Windows, Vehicle-to-Grid, and Heterogeneous Fleet Design (E-VRP-TWV2GHFD).
    Builds pyomo Mixed Integer Linear Program (MILP) model for exact-form solutions.
    Author: Rami Ariss
    """

    def __init__(self, instance):

        # Instantiate pyomo Abstract Model
        self.m = AbstractModel()


    #     self. =
    #
    # def parse_instance(self):
