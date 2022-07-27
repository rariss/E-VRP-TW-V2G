from os import listdir
from milp.evrptwv2g_base import EVRPTWV2G
from milp.evrptwv2g_gdp import EVRPTWV2G as EVRPTWV2G_gdp
from milp.evrptwv2g_gdp_nested_all import EVRPTWV2G as EVRPTWV2G_gdp_nested_all
from milp.evrptwv2g_gdp_nested_station import EVRPTWV2G as EVRPTWV2G_gdp_nested_station
from config.LOCAL_CONFIG import DIR_INSTANCES, DIR_OUTPUT


def save_solution(sol_tag, m):

    try:
        obj_breakdown = [m.total_distance(m.instance)(),
                         m.C_fleet_capital_cost(m.instance)(),
                         m.O_delivery_operating_cost(m.instance)() + m.O_maintenance_operating_cost(m.instance)(),
                         m.R_delivery_revenue(m.instance)(),
                         m.R_energy_arbitrage_revenue(m.instance)(),
                         m.R_peak_shaving_revenue(m.instance)(),
                         m.cycle_cost(m.instance)()]

        gap = m.results['Problem'][0]
        gap = round((gap['Upper bound'] - gap['Lower bound']) / gap['Upper bound'] * 100, 2)

        sol_path = f'{DIR_OUTPUT}/model_compare/{sol_tag}'
        f = open(sol_path,'w+')

        solve_time = m.results['Solver'][0]['Time'] if hasattr(m.results['Solver'][0], 'Time') else m.results['Solver'][0]['Wallclock time']  # SOLVER_TYPE = 'gurobi' or 'gurobi_direct'
        f.write('Objective: {:.2f}, \nGap: {:.2f}%, Solver time: {:.1f}s\n'.format(m.instance.obj.expr(), gap, solve_time))
        f.write('Dist.: {:.2f}, CapEx: {:.2f}, OpEx: {:.2f}, Delivery: {:.2f}, EA: {:.2f}, DCM: {:.2f}, Cycle: {:.2f}'.format(*obj_breakdown))
        f.write('\nBuild duration: {:.2f}s'.format(m.model_build_duration))
        try:
            f.write('\nXfrm duration: {:.2f}s'.format(m.model_xfrm_duration))
        except:
            pass # splitxp model is hardcoded, no XFRM time
        f.write('\nTotal solve duration: {:.2f}s'.format(m.model_solve_duration))
        f.write('\n\n\n--------------------------------------------\n')
        f.write(str(m.results))

        f.close()

    except Exception as e:
        sol_path = f'{DIR_OUTPUT}/model_compare/{sol_tag}'
        logf = open(sol_path,'w')
        logf.write(str(e))
        logf.close()



def main():

    """ Run a full solve """

    fpath = f'{DIR_INSTANCES}/model_compare/'
    instances = listdir(fpath)

    model_type = {'splitxp' : EVRPTWV2G,
                  'gdp_nested_station' : EVRPTWV2G_gdp_nested_station,
                  'gdp' : EVRPTWV2G_gdp,
                  'gdp_nested_all' : EVRPTWV2G_gdp_nested_all}


    xfrm_type = ['bigm'] #, 'cuttingplane' , 'hull'
    # xfrm_type = ['bigm', 'hull', 'cuttingplane', 'gdpopt']
    # xfrm_type = ['gdpopt']

    problem_type_generic = 'OpEx CapEx Cycle EA DCM Delivery splitxp'

    for instance in instances:

        instance_filepath = fpath + instance

        for model in model_type:

            if model == 'splitxp':

                sol_tag = instance[:-4] + model + '.txt'
                problem_type_ = problem_type_generic
         
                m = model_type[model](problem_type=problem_type_)

                m.import_instance(instance_filepath)
                m.make_instance()
                m.make_solver(solve_options={'TimeLimit': 60 * 2})

                try:
                    m.solve()
                    save_solution(sol_tag, m)

                except Exception as e:
                    sol_path = f'{DIR_OUTPUT}/model_compare/{sol_tag}'
                    logf = open(sol_path,'w')
                    logf.write(str(e))
                    logf.close()


            else:
                problem_type_ = problem_type_generic + ' ' + model

                for xfrm in xfrm_type:

                    sol_tag = instance[:-4] + model + '_' + xfrm + '.txt'
                    problem_type_xfrm = problem_type_ + ' ' + xfrm

                    m = model_type[model](problem_type=problem_type_xfrm)

                    m.import_instance(instance_filepath)
                    m.make_instance()
                    m.make_solver(solve_options={'TimeLimit': 60 * 2})

                    try:
                        m.solve()
                        save_solution(sol_tag, m)
                    except Exception as e:
                        sol_path = f'{DIR_OUTPUT}/model_compare/{sol_tag}'
                        logf = open(sol_path,'w')
                        logf.write(str(e))
                        logf.close()


if __name__ == "__main__":
    main()
