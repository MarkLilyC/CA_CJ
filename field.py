from ped import Ped
from cell2 import *
import utilies
import copy
import random
import math

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
    
    def __init__(self, x_range:tuple, y_range:tuple, kappaD: int = 1, kappaS:int = 2, delta: float = 0.3, alpha: float = 0.1) -> None:
        assert x_range[0] < x_range[1] and y_range[0] < y_range[1], f"Wrong coordinate range"
        self.__x_range = x_range
        self.__y_range = y_range
        self.kappaD = kappaD
        self.kappaS = kappaS
        self.delta = delta
        self.alpha = alpha
        self.cells = Field.init_cells(x_range, y_range)
        self.ped_n = 0
        self.ped = []
        self.walls = []
        self.exits = []
        self.frees = []
    
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
    
    def check_loc(self, loc:tuple|list):
        x, y = loc
        return self.x_start <= x < self.x_end and self.y_start <= y < self.y_end

    def at(self, loc:tuple|list):
        return self.cells[loc[0]][loc[1]]
    
    def set(self, loc:tuple|list, cell:type = Cell|Wall|Free|Exit, sff:float = 0, dff:float = 0, ped:Ped = None):
        # 考虑越界
        x, y = loc
        assert self.x_start <= x <= self.x_end - 1 and self.y_start <= y <= self.y_end - 1, f"Loc:{loc} is out of range of Filed Size:{self.__x_range, self.__y_range}"
        if cell is not Wall:
            self.cells[x][y] = cell(x, y, sff = sff, dff = dff, ped = ped)
        else:
            self.cells[x][y] = cell(x, y, sff = sff, dff = dff)

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
                if cell.cname == "Wall": res += '-1, '
                elif cell.cname == "Free": 
                    res += f' {0 if cell.free else 1}, '  # 如果free元胞被占据则赋值为1 未被占据则赋值为0
                elif cell.cname == "Exit": 
                    res += f' {0 if cell.free else 1}, '  # 如果free元胞被占据则赋值为1 未被占据则赋值为0
                elif cell.cname == "Cell": res += '-2, '
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
            self.set(loc=i, cell=Exit, sff = 0)
    
    def init_free(self):
        for row in self.cells:
            for cell in row:
                if cell.cname == 'Free':self.frees.append(cell.loc)
                else:pass
    
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
                classname = cell.cname
                res[classname] += 1
        return total, res
    
    def ped_capacity(self):
        _, res = self.count()
        return res['Free'] + res["Exit"]

    def get_walkable_cells(self) -> list[Free]:
        free_cells = []
        for row in self.cells.values():
            for cell in row:
                if cell.cname == "Free": free_cells.append(cell)
                else:pass
        return free_cells

    def init_peds(self, n_peds:int = 10):
        # 首先检查传入的人数是否能被容纳
        PED_CAPACITY = self.ped_capacity()
        assert n_peds <= PED_CAPACITY, f"Too many Peds ({n_peds}) to be initized in the field(capacity:{PED_CAPACITY})"
        # 获取场内的free cells
        walkable_cells = self.get_walkable_cells()
        # 随机打乱
        random.shuffle(walkable_cells)
        for i in range(n_peds):
            walkable_cells[i].init_ped(ped_id=i)
            self.ped.append(walkable_cells[i].ped)
        self.ped_n = n_peds
    
    def classify_cells(self):
        '''
        reture: walls, exits, frees
        '''
        # 用于对field中的cell进行分类 需要建立在init_wall 和 init_exit的基础上
        for row in self.cells.values(): 
            for cell in row:
                if isinstance(cell, Wall):self.walls.append(cell)
                elif isinstance(cell, Exit):self.exits.append(cell)
                elif isinstance(cell, Free):self.frees.append(cell)
                else:raise TypeError(f"Wrong type of cell in the filed {cell.__str__()}")
    
    def find_neighbor(self, cell) -> list[Cell]:
        if isinstance(cell, Cell):
            pass
        else:
            cell = self.at(cell)
        # 先找出当前元胞的邻居元胞该有的坐标
        neighbors = cell.neighbor_loc()
        neighbor_cells = []
        # 检查上述元胞坐标是否都在当前field中
        for nei in neighbors:
            if self.check_loc(nei) and isinstance(self.at(nei), Free): 
                neighbor_cells.append(self.at(nei))
            else:pass
        return neighbor_cells
    
    def init_sff(self):
        # 先对field中的cells进行分类
        self.classify_cells()
        # 先初始化wall元胞的sff
        wall_sff = math.ceil(math.sqrt(self.x_len() ** 2 + self.y_len() ** 2))
        
        for wall in self.walls:
            wall.sff = wall_sff
        
        for free in self.frees:
            free.sff = wall_sff
        # 接下来从exit开始向内逐步初始化静态场
        cells_initized = copy.deepcopy(self.exits)
        # 因为出口的场值为0 这是在init exit时就定义好的 所以exits属于已经经过初始化的cells
        # 以下更新规则为不停将当前的一级元胞取出来寻找其邻居 然后将其邻居当做下一次的一级元胞
        while(cells_initized):
            # 取出当前的一级元胞
            cell = cells_initized.pop(0)
            # 找到当前cell的可行进邻居元胞
            neis = self.find_neighbor(cell)
            # 迭代上述可行进元胞
            for nei in neis:
                # 将当前一级元胞的所有直连free元胞的sff值赋值为当前元胞sff值 + 1
                if cell.sff + 1 < nei.sff:
                    nei.sff = cell.sff + 1
                    cells_initized.append(nei)  
                
    def show_sff(self):
        res = ''
        for row in self.cells.values():
            for cell in row:
                res += f' {cell.sff}, '
            res += '\n'
        print(res)
    
    def show_dff(self):
        res = ''
        for row in self.cells.values():
            for cell in row:
                res += f' {cell.dff}, '
            res += '\n'
        print(res)
    
    def get_ped_cells(self):
        # 找出场景中的所有存在ped的cell
        # 直接迭代ped
        res = []
        for ped in self.ped:
            res.append(self.at(ped.get_loc))
        return res
    
    def seq_updata_cells(self, shuffle, reverse = None):
        
        # 把场内的非wall元胞打乱
        walkable = self.exits + self.frees
        if shuffle:random.shuffle(walkable)
        # 找出所有存在ped的cell
        ped_cells = self.get_ped_cells()
        for cell in ped_cells:
            pass
        '''
        for cell in walkable:
            # 如果当前元胞没有行人 则跳出本次循环
            if cell.free: continue
            
            # 如果当前为exit
            if isinstance(cell, Exit):
                cell.ped_exit()
                cell.dff_diff += 1
                continue
            
            # 如果当前为free元胞 且有人存在
            prob = 0
            probs = {}
            # loc = cell.loc
            for neighbor in self.find_neighbor(cell=cell):
                if neighbor.free:
                    probability = math.exp(self.kappaS * (cell.sff - neighbor.sff))#  * math.exp(self.kappaD * (neighbor.dff - cell.dff))
                    prob += probability
                    probs[neighbor]  = probability
            
            if prob == 0:continue
            
            r = random.random() * prob
            for nei, p in probs.items():
                r -= p
                if r <= 0:
                    cell.ped_moveout(nei)
                    cell.dff_diff += 1
                    self.show_obst()
                    # 将当前元胞弹出 这个step不再对其进行计算
                    break
            walkable.pop(0)
        '''
            
                # np.exp(kappaS * (sff[cell] - sff[neighbor])) * np.exp(kappaD  * (dff[neighbor] - dff[cell])) * (1 - tmp_peds[neighbor])
            
    
    def updata_dff(self):
        # 迭代所有可移动元胞
        for cell in self.exits:
            for _ in range(int(cell.dff)):
                if random.random() < self.delta:
                    cell.dff -= 1
                elif random.random() < self.alpha:
                    cell.dff -= 1
                    random.choice(self.find_neighbor(cell=cell)).dff += 1

        
    
    def sim(self,steps, shuffle):
        for t in range(steps):
            self.seq_updata_cells(shuffle=shuffle)
            self.show_obst()
            # self.updata_dff()
    
    

f = Field((0,8), (0,8))
print(f)
f.init_walls()
f.init_exit()
f.init_peds(3)
f.init_sff()
f.show_sff()
f.sim(1000, True)
