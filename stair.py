import os
from ped import Ped
from cell2 import * 
from field import Field

class Stair(Field):
    def __init__(self, x_range: tuple, y_range: tuple, id: int = 1, kappaD: int = 1, kappaS: int = 2, delta: float = 0.3, alpha: float = 0.1, cellsize: float = 0.4) -> None:
        super().__init__(x_range, y_range, id, kappaD, kappaS, delta, alpha, cellsize)
        self.stair_in = []

    def set_stair_in(self, stari_in_loc:list) :
        for loc in stari_in_loc:
            if self.check_loc(loc=loc):
                self.set(loc=loc, cell=Free)
                self.stair_in.append(loc)

if __name__ == "__main__":
    exit_loc = [[10, 10], [11,10]]
    s = Stair(x_range=[6,13], y_range=[10, 16])
    s_in = [[7,10], [8,10]]
    s.init_walls()
    s.init_exit(exit_cells=exit_loc)
    s.set_stair_in(s_in)
    s.init_sff()
    s.show_obst()
    s.init_peds(n_peds=3)
    s.sim(1000)