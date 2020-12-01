from utils.utilities import import_instance


def create_data_dictionary(data):  # TODO: get hardcoded values from instance file

    p = {None: {
        'Mq': {None: int(data['W'].loc[:, 'QMAX'].mean())},  # Large value for big M payload constraints
        'Mt': {None: data['Parameters'].loc['t_T', 'value']},  # Large value for big M service time constraints
        'Me': {None: 60.63},  # Large value for big M energy constraints
        'cW': {None: data['W'].loc[:, 'cc'].mean()},
        # Amortized capital cost for purchasing a vehicle of type w
        'QMAX': {None: int(data['W'].loc[:, 'QMAX'].mean())},  # Maximum payload limit for all vehicles
        'EMAX': {None: 60.63},  # Maximum EV battery SOE limit for all EVs
        'V01_': {None: data['V'].index.values},  # All nodes extended np.arange(len(nodes)))
        'F': {None: data['S'].index.values},  # All charging station nodes in the extended graph
        'V': {None: data['M'].index.values},  # All customer nodes in the extended graph
        'd': data['d'].stack().to_dict(),  # Distance of edge (i;j) between nodes i;j (km)
        'q': data['V']['q'].to_dict(),  # Delivery demand at each customer
        'tS': data['V']['tS'].to_dict(),  # Fixed service time for a vehicle at node i
        'tA': data['V']['tA'].to_dict(),  # Delivery demand at each customer
        'tB': data['V']['tB'].to_dict(),  # Delivery demand at each customer
        'start_node': {None: data['start_node']},
        'end_node': {None: data['end_node']},
        # 'v': {None: 1},
        'rE': {None: 1}  # Electric consumption per unit distance for EV

    }}

    return p
