from utils.utilities import import_instance


def create_data_dictionary(data):  # TODO: get hardcoded values from instance file

    p = {None: {
        'Mq': {None: data['W'].loc[:, 'QMAX'].mean()},  # Large value for big M payload constraints
        'Mt': {None: data['Parameters'].loc['t_T', 'value']},  # Large value for big M service time constraints
        'Me': {None: data['W'].loc[:, 'EMAX'].mean()},  # Large value for big M energy constraints
        'cW': {None: data['W'].loc[:, 'cc'].mean()},
        # Amortized capital cost for purchasing a vehicle of type w
        'QMAX': {None: data['W'].loc[:, 'QMAX'].mean()},  # Maximum payload limit for all vehicles
        'EMAX': {None: data['W'].loc[:, 'EMAX'].mean()},  # Maximum EV battery SOE limit for all EVs
        'V01_': {None: data['V'].index.values},  # All nodes extended np.arange(len(nodes)))
        'F_': {None: data['S'].index.values},  # All charging station nodes in the extended graph
        'V': {None: data['M'].index.values},  # All customer nodes in the extended graph
        'd': data['d'].stack().to_dict(),  # Distance of edge (i;j) between nodes i;j (km)
        'q': data['V']['q'].to_dict(),  # Delivery demand at each customer
        'tS': data['V']['tS'].to_dict(),  # Fixed service time for a vehicle at node i
        'tA': data['V']['tA'].to_dict(),  # Delivery demand at each customer
        'tB': data['V']['tB'].to_dict(),  # Delivery demand at each customer
        'start_node': {None: data['start_node']},
        'end_node': {None: data['end_node']},
        'v': {None: data['W'].loc[:, 'v'].mean()},
        'rE': {None: data['W'].loc[:, 'rE'].mean()},  # Electric consumption per unit distance for EV
        'rC': {None: data['W'].loc[:, 'rC'].mean()}

    }}

    return p
