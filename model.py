import utilies
from ped import Ped
from floor import Floor
from building import Building
import argparse


MAX_STEPS = 1000
steps  = range(MAX_STEPS)

cellsize = 0.4
vmax =  1.2
dt = cellsize / vmax

from_x, to_x = 1, 63  # todo parse this too
from_y, to_y = 1, 63  # todo parse this too
DEFAULT_BOX = [from_x, to_x, from_y, to_y]
del from_x, to_x, from_y, to_y


def get_parser_args():
    parser = argparse.ArgumentParser(
        description='Cellular Automaton. Floor Field Model [Burstedde2001] Simulation of pedestrian'
                    'dynamics using a two-dimensional cellular automaton Physica A, 295, 507-525, 2001')
    parser.add_argument('-s', '--ks', type=float, default=2,
                        help='sensitivity parameter for the  Static Floor Field (default 2)')
    parser.add_argument('-d', '--kd', type=float, default=1,
                        help='sensitivity parameter for the  Dynamic Floor Field (default 1)')
    parser.add_argument('-n', '--numPeds', type=int, default=10, help='Number of agents (default 10)')
    parser.add_argument('-p', '--plotS', action='store_const', const=True, default=False,
                        help='plot Static Floor Field')
    parser.add_argument('--plotD', action='store_const', const=True, default=False,
                        help='plot Dynamic Floor Field')
    parser.add_argument('--plotAvgD', action='store_const', const = True, default=False,
                        help='plot average Dynamic Floor Field')
    parser.add_argument('-P', '--plotP', action='store_const', const=True, default=False,
                        help='plot Pedestrians')
    parser.add_argument('-r', '--shuffle', action='store_const', const=True, default=True,
                        help='random shuffle')
    parser.add_argument('-v', '--reverse', action='store_const', const=True, default=False,
                        help='reverse sequential update')
    parser.add_argument('-l', '--log', type=argparse.FileType('w'), default='log.dat',
                        help='log file (default log.dat)')
    parser.add_argument('--decay', type=float, default=0.3,
                        help='the decay probability of the Dynamic Floor Field (default 0.2')
    parser.add_argument('--diffusion', type=float, default=0.1,
                        help='the diffusion probability of the Dynamic Floor Field (default 0.2)')
    parser.add_argument('-W', '--width', type=float, default=4.0,
                        help='the width of the simulation area in meter, excluding walls')
    parser.add_argument('-H', '--height', type=float, default=4.0,
                        help='the height of the simulation room in meter, excluding walls')

    parser.add_argument('-c', '--clean', action='store_const', const=True, default=False,
                        help='remove files from directories dff/ sff/ and peds/')

    parser.add_argument('-N', '--nruns', type=int, default=1,
                        help='repeat the simulation N times')

    parser.add_argument('--parallel', action='store_const', const=True, default=False,
                        help='use multithreading')
    parser.add_argument('--moore', action='store_const', const=True, default=False,
                        help='use moore neighborhood. Default= Von Neumann')

    parser.add_argument('--box', type=int, nargs=4, default=DEFAULT_BOX,
                        help='Rectangular box, initially populated with agents: from_x, to_x, from_y, to_y. Default: The whole room')

    _args = parser.parse_args()
    return _args


def simulate(building:Building):
    pass


if __name__ == "__main__":

    building = Building(1)

    f1 = Floor(id=1, width=4.0, height=4.0, cellsize=0.4, z=0.0)
    f1.set_exit_cells()
    f1.init_wall()
    f1.init_sff()
    f1.init_peds(10)
    
    


