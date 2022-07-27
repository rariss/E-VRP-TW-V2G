# Node colors in hexadecimals and RGBA
node_colors_hex = {'m': '#f0a9ab',
                   'd': '#bff3b4',
                   's': '#a7d7f1'
                   }

node_colors_rgba = {'m': 'rgba(240, 169, 171, 0.8)', #alpha=0.4
                    'd': 'rgba(191, 243, 180, 0.8)',
                    's': 'rgba(167, 215, 241, 0.8)'
                    }

node_colors_rgba_tuple = {'M': (240/255, 169/255, 171/255, 0.8), #alpha=0.4
                          'D': (191/255, 243/255, 180/255, 0.8),
                          'S': (167/255, 215/255, 241/255, 0.8)
                          }

SOLVER_TYPE = 'gurobi'  # gurobi_direct, gurobi_persistent (not fully supported)
