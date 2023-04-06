from ped import Ped
from cell import *
import utilies
import numpy as np
import copy
import random

class Field(object):
    @staticmethod
    def init_cells(x_range:tuple, y_range:tuple) -> dict:
        cells = {}
        for i in range(x_range[0],x_range[1]):
            tmp = []
            for j in range(y_range[0], y_range[1]):
                cell = Free(x=i,y=j)
                tmp.append(cell)
            cells[i] = tmp
        return cells
    
    def __init__(self, x_range:tuple, y_range:tuple) -> None:
        assert x_range[0] < x_range[1] and y_range[0] < y_range[1], f"Wrong coordinate range"
        self.__x_range = x_range
        self.__y_range = y_range
        self.cells = Field.init_cells(x_range, y_range)
    
    @property
    def x_start(self): return self.__x_range[0]

    @property
    def x_end(self): return self.__x_range[1]

    def x_len(self): return self.x_end - self.x_start

    @property
    def y_start(self): return self.__y_range[0]

    @property
    def y_end(self): return self.__y_range[1]

    def y_len(self): return self.y_end - self.y_start

    def at(self, loc:tuple|list):
        return self.cells[loc[0]][loc[1]]
    
    def set(self, loc:tuple|list, cell:type = Cell|Wall|Free|Exit):
        # 考虑越界
        x, y = loc
        assert self.x_start <= x <= self.x_end - 1 and self.y_start <= y <= self.y_end - 1, f"Loc:{loc} is out of range of Filed Size:{self.__x_range, self.__y_range}"
        self.cells[x][y] = cell(x, y)

    def get_x_range(self):return self.__x_range
    
    def get_y_range(self):return self.__y_range
    
    def init_walls(self) -> None:
        # 第一行与最后一行
        for y in range(self.y_start, self.y_end):
            cell = (self.x_start, y)   # 第一行
            self.set(loc=cell, cell=Wall)
            cell = (self.x_end - 1, y)  # 最后一行
            self.set(loc=cell, cell=Wall)
        # 第一列与最后一列
        for x in range(self.x_start + 1, self.x_end):
            cell = (x, self.y_start)   # 第一列
            self.set(loc=cell, cell=Wall)
            cell = (x, self.y_end - 1)   # 最后一列
            self.set(loc=cell, cell=Wall)

    def show_obst(self):
        res = ''
        for row in self.cells.values():
            for cell in row:
                if cell.get_classname() == "Wall": res += '-1, '
                elif cell.get_classname() == "Free": res += ' 0, '
                elif cell.get_classname() == "Exit": res += ' 0, '
                elif cell.get_classname() == "Cell": res += '-2, '
                else:raise ValueError(f"Wrong cell {cell}, type={cell.get_classname()} in the filed")
            res += '\n'
        print(res)
    
    def init_exit(self, exit_cells = None):
        # 如果没有传入具体的出口坐标 
        if exit_cells is None:
            exit_cells = (
                (self.x_end // 2, self.y_start),
                (self.x_end // 2, self.y_end - 1),
                (self.x_start, self.y_end // 2),
                (self.x_end - 1, self.y_end // 2)
            )
        for i in exit_cells:
            self.set(loc=i, cell=Exit)
        
    def __str__(self) -> str:
        return f"Field x_range:{self.__x_range}, y_range:{self.__y_range}"
    
    def count(self):
        total = 0
        res = {
            "Wall":0,
            "Exit":0,
            "Free":0,
            "Cell":0
        }
        for row in self.cells.values():
            total += len(row)
            for cell in row:
                classname = cell.get_classname()
                res[classname] += 1
        return total, res
    
    def ped_capacity(self):
        _, res = self.count()
        return res['Free'] + res["Exit"]

    def get_walkable_cells(self) -> list[Free]:
        free_cells = []
        for row in self.cells.values():
            for cell in row:
                if cell.get_classname() == "Free": free_cells.append(cell)
                else:pass
        return free_cells

    def init_peds(self, n_peds:int = 10):
        # 首先检查传入的人数是否能被容纳
        PED_CAPACITY = self.ped_capacity()
        assert n_peds <= PED_CAPACITY, f"Too many Peds ({n_peds}) to be initized in the field(capacity:{PED_CAPACITY})"
        # 获取场内的free cells
        walkable_cells = self.get_walkable_cells()
        random.shuffle(walkable_cells)
        ped_index = 0
        for cell in walkable_cells:
            cell.init_ped(ped_id=ped_index)
            ped_index += 1
        


    
    def sim(self,):
     pass
    
    

f = Field((0,18), (0,12))
print(f)
f.init_walls()
f.show_obst()
f.init_exit()
f.show_obst()
f.init_peds(20)