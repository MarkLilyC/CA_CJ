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
    
    def check_loc(self, loc:tuple|list) -> bool:
        x, y = loc
        x_flag = self.x_start <= x < self.x_end
        y_flag = self.y_start <= y < self.y_end
        return x_flag and y_flag

    def at(self, loc:tuple|list):
        '''返回在某一个坐标的元胞
        Args:
            loc (tuple | list): _description_

        Returns:
            _type_: _description_
        '''

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

    def get_free_cells(self) -> list[Free]:
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
        walkable_cells = self.get_free_cells()
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
    
    def find_all_neighbor(self, cell:Cell) -> list[Cell]:
        '''找到给定元胞的所有邻居元胞
            * 内部先拿到cell的计算所有坐标意义上的邻居坐标值
            * 然后对上述计算得到的邻居坐标值进行场内的判断 即该坐标是否位于当前场内
            * 找到只是所有邻居元胞 并没有对元胞的类型进行限制

        Args:
            cell (_type_): _description_

        Returns:
            list[Cell]: _description_
        '''
        neighbors_loc = cell.neighbor_loc()
        neighbor_cell = []
        for loc in neighbors_loc:
            # 要确定该位置是否存在于场内 这一步可能会出现该位置不存在于场内的正常情况 因此采用warn mode
            if self.check_loc(loc=loc): neighbor_cell.append(self.at(loc=loc))
        return neighbor_cell

    def find_walkable_neighbor(self, cell:Cell) -> list[Cell]:
        '''找到一个元胞周围的free exit两类元胞
            * 不判断是否可以行进
        Args:
            cell (Cell): _description_

        Returns:
            list[Cell]: _description_
        '''
        neighbor_loc = cell.neighbor_loc()
        neighbor_cell = []
        for loc in neighbor_loc:
            # 需要有两个条件必须满足 该位置位于当前场内 该位置的元胞是非Wall
            # 首先确定是否存在于场内
            if self.check_loc(loc=loc):
                cell = self.at(loc=loc)
                # 判断元胞的类型
                if cell.cname == 'Free' or cell.cname == "Exit":  
                    neighbor_cell.append(cell)
                elif cell.cname == "Wall":pass
                else: print(f"Got unexcpted Type of cell:{cell.__class__}")
        return neighbor_cell
    
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
            # 找到当前cell的Free类邻居元胞
            neis = self.find_walkable_neighbor(cell)
            # 迭代上述Free类邻居元胞
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
        
        # 直接迭代ped
        for ped in self.ped:
            loc = ped.get_loc   # 获取当前ped的坐标
            cell = self.at(loc=loc) # 根据loc取出该元胞
            # 判断cell的类型
            assert cell.cname != "Wall", f"Ped{ped.__str__()} unexpectedly move into a Wall cell"
            # 如果cell为exit 则直接调用exit的ped_exit
            if isinstance(cell, Exit):
                cell.ped_exit()
                self.ped.remove(ped)
                cell.dff_diff += 1
                self.show_obst()
            # 如果cell为free 则进行详细的移动概率计算
            elif isinstance(cell, Free):
                prob, probs = 0, {}  # 定义当前ped移动的总概率和各个邻居元胞的移动概率
                # 迭代当前ped所处元胞的所有的free exit邻居元胞
                for neighbor in self.find_walkable_neighbor(cell=cell):
                    # 判断元胞是否被占据
                    if neighbor.free:
                        # 计算概率
                        probability = math.exp(self.kappaS * (cell.sff - neighbor.sff))#  * math.exp(self.kappaD * (neighbor.dff - cell.dff))
                        prob += probability
                        probs[neighbor]  = probability
                    else: pass
                # 如果概率为0 则该ped不进行移动
                if prob == 0:continue
                else:
                    r = random.random() * prob  #   取一个随机数
                    # 迭代所有可移动元胞的概率列表
                    for nei, p in probs.items():
                        r -= p
                        if r <= 0:
                            cell.ped_moveout(nei)
                            cell.dff_diff += 1
                            self.show_obst()
                            # 只要ped进行了一次移动就应该弹出当前循环
                            break
            else: raise TypeError(f"{ped} located on a unexpexted type of Cell: {cell.__class__}")
                    
                            

            
                # np.exp(kappaS * (sff[cell] - sff[neighbor])) * np.exp(kappaD  * (dff[neighbor] - dff[cell])) * (1 - tmp_peds[neighbor])
            
    
    def updata_dff(self):
        # 迭代所有可移动元胞
        for cell in self.exits:
            for _ in range(int(cell.dff)):
                if random.random() < self.delta:
                    cell.dff -= 1
                elif random.random() < self.alpha:
                    cell.dff -= 1
                    random.choice(self.find_walkable_neighbor(cell=cell)).dff += 1

        
    
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
f.show_obst()
f.sim(1000, True)
