from ped import Ped
from cell2 import *
import utilies
import copy
import random
import math
import matplotlib.pyplot as plt
import numpy as np
import os
import imageio.v2 as imageio
from tqdm import tqdm
import collections

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
    
    def __init__(self,  x_range:tuple, y_range:tuple, id:int = 1, kappaD: int = 1, kappaS:int = 2, delta: float = 0.3, alpha: float = 0.1, cellsize:float = 0.4) -> None:
        assert x_range[0] < x_range[1] and y_range[0] < y_range[1], f"Wrong coordinate range"
        self.__x_range = x_range
        self.__y_range = y_range
        self.__id = id
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
        self.cellsize = cellsize
        self.total_sim_step = None
        self.sim_step = None
        self.op_path = os.getcwd() + f"//img//Field_{self.__id}_{self.__x_range, self.__y_range}_{self.kappaS}_{self.kappaD}_{self.delta}_{self.alpha}_{self.cellsize}//"
        if os.path.exists(self.op_path):pass
        else:os.makedirs(self.op_path)
        self.op_imgs_path = []
        self.stair = None
        self.field_connection = None
    
    def get_id(self): return self.__id
    
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

    def at(self, loc:tuple|list) -> Cell:
        '''返回在某一个坐标的元胞
        Args:
            loc (tuple | list): _description_

        Returns:
            _type_: _description_
        '''

        return self.cells[loc[0]][loc[1] - self.y_start]

    def set(self, loc:tuple|list, cell:type = Cell|Wall|Free|Exit, sff:float = 0, dff:float = 0, ped:Ped = None):
        # 考虑越界
        x, y = loc
        assert self.x_start <= x <= self.x_end - 1 and self.y_start <= y <= self.y_end - 1, f"Loc:{loc} is out of range of Filed Size:{self.__x_range, self.__y_range}"
        if cell is not Wall:
            self.cells[x][y-self.y_start] = cell(x, y, sff = sff, dff = dff, ped = ped)
        else:
            self.cells[x][y-self.y_start] = cell(x, y, sff = sff, dff = dff)

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
            
    def set_walls(self, wall_loc:list):
        for loc in wall_loc:
            if self.check_loc(loc):
                self.set(loc=loc, cell=Wall)        
    
    def show_obst(self):
        res = self.ped_info()
        print(res)
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

    def init_peds(self, n_peds:int = 10, loc: list = None):
        # 首先检查传入的人数是否能被容纳
        PED_CAPACITY = self.ped_capacity()
        assert n_peds <= PED_CAPACITY, f"Too many Peds ({n_peds}) to be initized in the field(capacity:{PED_CAPACITY})"
        if loc is None:
            # 获取场内的free cells
            walkable_cells = self.get_free_cells()
            # 随机打乱
            random.shuffle(walkable_cells)
            for i in range(n_peds):
                walkable_cells[i].init_ped(ped_id=i)
                self.ped.append(walkable_cells[i].ped)
            self.ped_n = n_peds
        else:
            assert n_peds == len(loc), f"Ped n unequal to ped loc"
            index = 0
            for l in loc:
                cell = self.at(loc=l)
                assert cell.cname != 'Wall', f"You cannot init ped into a wall cell"
                cell.init_ped(ped_id = index)
                self.ped.append(cell.ped)
                index += 1
            self.ped_n = n_peds

    def receive_ped(self, cell:Cell):
        pass

    def ped_tran_field(self, cell:Cell):
        '''
        1:只有在遇到exit cell时会调用本方法
        * 当前exit是上层楼层连接stair的cell: 此时直接应该将ped移动到所连接的stair
        * 当前exit是base楼层连接室外的cell: 此时直接应该将ped疏散到室外
        * 当前exit是stair连接field的cell: 此时应该将ped移动到所连接的field
        '''
        pass
            
    
    @property
    def get_n_ped(self):
        return len(self.ped)
    
    def get_ped_by_id(self, id:int) -> Ped:
        for ped in self.ped:
            if ped.get_id == id: return ped
            else:pass
        raise IndexError(f"Id is not in")
        
    def remove_ped_by_id(self, id):
        ped = self.get_ped_by_id(id=id)
        self.ped.remove(ped)
    
    def classify_cells(self) -> list[Cell]:
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
    
    def seq_updata_cells(self, shuffle):
        # 直接迭代ped
        # for ped in self.ped:
        # 创建一个临时ped列表用于迭代
        tmp_ped = copy.deepcopy(self.ped)
        while tmp_ped:
            ped = tmp_ped.pop(0)
            loc = ped.get_loc   # 获取当前ped的坐标
            cell = self.at(loc=loc) # 根据loc取出该元胞
            # 判断cell的类型
            assert cell.cname != "Wall", f"Ped{ped.__str__()} unexpectedly move into a Wall cell"
            # 如果cell为exit 则应该调用场的ped_tran_field方法
            if isinstance(cell, Exit):
                cell.ped_exit()
                self.remove_ped_by_id(id=ped.get_id)
                cell.dff_diff += 1
                # self.show_obst()
            elif isinstance(cell, Tran):
                # 如果当前cell为Tran 即下一次ped的运动会进入另一个场
                # 此时调用field类的ped_moveout方法
                self.ped_moveout(cell=cell)
            # 如果cell为free 则进行详细的移动概率计算
            elif isinstance(cell, Free):
                prob, probs = 0, {}  # 定义当前ped移动的总概率和各个邻居元胞的移动概率
                # 迭代当前ped所处元胞的所有的free exit邻居元胞
                for neighbor in self.find_walkable_neighbor(cell=cell):
                    # 判断元胞是否被占据
                    if neighbor.free:
                        # 计算概率
                        probability = math.pow(math.exp(self.kappaS * (cell.sff - neighbor.sff)), 2)#  * math.exp(self.kappaD * (neighbor.dff - cell.dff))
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
                            # self.show_obst()
                            # 只要ped进行了一次移动就应该弹出当前循环
                            break
            else: raise TypeError(f"{ped} located on a unexpexted type of Cell: {cell.__class__}")
            # np.exp(kappaS * (sff[cell] - sff[neighbor])) * np.exp(kappaD  * (dff[neighbor] - dff[cell])) * (1 - tmp_peds[neighbor])

    def seq_updata_cells_s(self):
        tmp_ped = copy.deepcopy(self.ped)
        cell_prob = collections.defaultdict(dict)
        while tmp_ped:
            ped = tmp_ped.pop(0)
            loc = ped.get_loc   # 获取当前ped的坐标
            cell = self.at(loc=loc) # 根据loc取出该元胞
            # 判断cell的类型
            assert cell.cname != "Wall", f"Ped{ped.__str__()} unexpectedly move into a Wall cell"
            # 如果cell为exit 则应该调用场的ped_tran_field方法
            if isinstance(cell, Exit):
                cell.ped_exit()
                self.remove_ped_by_id(id=ped.get_id)
                cell.dff_diff += 1
                # self.show_obst()
            
            # 如果cell为free 则进行详细的移动概率计算
            elif isinstance(cell, Free):
                prob, probs = 0, {}  # 定义当前ped移动的总概率和各个邻居元胞的移动概率
                # 迭代当前ped所处元胞的所有的free exit邻居元胞
                for neighbor in self.find_walkable_neighbor(cell=cell):
                    # 判断元胞是否被占据
                    if neighbor.free:
                        # 计算概率
                        probability = math.pow(math.exp(self.kappaS * (cell.sff - neighbor.sff)), 2)#  * math.exp(self.kappaD * (neighbor.dff - cell.dff))
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
                            cell_prob[nei][ped] = p
                            break
            else: raise TypeError(f"{ped} located on a unexpexted type of Cell: {cell.__class__}")
        # 迭代所有cell
        for cell, ped_probs in cell_prob.items():
            p_max = 0
            move_ped = None # 定义最大概率对应的ped
            for ped in ped_probs.keys():    # 迭代所有ped寻找概率最大的ped
                tmp = ped_probs[ped]
                if tmp >= p_max: 
                    p_max = tmp
                    move_ped = ped
                else:pass 
            # 找到概率最大的ped后 就应该移动该ped
            origin_cell = self.at(move_ped.get_loc)  # 找到概率最大的ped所在的cell
            origin_cell.ped_moveout(target_cell=cell)   # 调用ped所在的cell的moveout方法
            self.show_obst()
            # 其余ped保持不动 因此不需要额外操作
            
    def updata_dff(self):
        # 迭代所有可移动元胞
        for cell in self.exits:
            for _ in range(int(cell.dff)):
                if random.random() < self.delta:
                    cell.dff -= 1
                elif random.random() < self.alpha:
                    cell.dff -= 1
                    random.choice(self.find_walkable_neighbor(cell=cell)).dff += 1

    def log(self):
        print(f"Field: {self.__x_range, self.__y_range}, cellsize={self.cellsize}")
        
    def sim(self, steps = 100, shuffle:bool = False):
        self.total_sim_step = steps
        print(f"Field: {self.__x_range, self.__y_range}, cellsize={self.cellsize}, Walls={len(self.walls)}, Exits={len(self.exits)}, Frees={len(self.frees)}")
        print(f"Simulation: Total Steps={self.total_sim_step}, KappaS={self.kappaS}, KappaD={self.kappaD}")
        print(f"Origin cell: including wall, ped, free cell")
        self.show_obst()
        for t in range(steps):
            # 场景中没有ped则应该退出
            if len(self.ped) == 0:
                self.plot_ped(step=t)
                break
            else:
                print(f"Start step: {t}, ped: {len(self.ped)}")
                # self.show_obst()
                self.seq_updata_cells( shuffle = True)
                self.plot_ped(step=t)
                print(f"End step: {t}, ped: {len(self.ped)}")
            # self.updata_dff()
    
    def ped_info(self):
        res = ''
        for ped in self.ped:res += ped.__str__() + ' '
        return res
    
    def plot_ped(self, step = '-', ):
        ped = []
        for row in self.cells.values():
            tmp = []
            for cell in row:
                # 如果是wall
                if cell.cname == "Wall": tmp.append(-1)
                else:tmp.append(1 if cell.free else 2)
            ped.append(tmp)
        ped = np.array(ped)
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.cla()
        cmap = plt.get_cmap("gray")
        cmap.set_bad(color='b', alpha = 0.8)
        # 统计场景中的总人数
        n_ped = self.get_n_ped
        grid_x = np.arange(self.x_start + 1, self.x_end - 1, self.cellsize)
        grid_y = np.arange(self.y_start + 1, self.x_end - 1, self.cellsize)
        ax.imshow(ped, cmap = cmap, interpolation='nearest', vmin = -1, vmax = 2)
        plt.grid(True, color = 'k', alpha=0.3)
        plt.yticks(np.arange(1.5, ped.shape[0], 1))
        plt.xticks(np.arange(1.5, ped.shape[1], 1))
        plt.setp(ax.get_xticklabels(), visible=False)
        plt.setp(ax.get_yticklabels(), visible=False)
        ax.tick_params(axis='both', which='both', length=0)
        plt.title(f"Step: {str(step)}, Ped:{self.get_n_ped}")
        figure_name = self.op_path + f'//{step}.png'
        self.op_imgs_path.append(figure_name)
        plt.savefig(figure_name)
        plt.close()
        
    def generante_gif(self):
        print(f"Start ganerating GIF for {len(self.op_imgs_path)} imgs at {self.op_path}")
        with imageio.get_writer(uri=self.op_path + 'sim.gif', mode='I', fps=1) as writer:
            for img in tqdm(self.op_imgs_path):
                writer.append_data(imageio.imread(img))
    
    def ped_moveout(self, cell:Tran):
        # 找出当前cell所连接的stair
        stair = self.field_connection[cell]
        # 确保所连接的对象是stair
        if isinstance(stair, Stair):    
            # 调用stair的ped-movein方法
            # 具体判断逻辑存在于ped-movein中
            flag = stair.ped_movein(cell=cell)
            # 如果能够运动到stair中 则会将本cell的ped设置为none
            # 如果不能运动到stair中 则本cell不会做任何改动 本cell的ped也不会移动
            
    @staticmethod
    def base_floor_attach_stair(floor, stair:Stair):
        # 传入stair类的stair_in中的坐标在本场内必须都是边界
        for cell in stair.exits:    # 找到第一层楼梯的疏散出口
            loc = cell.loc  # 出口的位置
            assert floor.check_loc(loc=loc) # 该位置必须存在于第一层的场内
            assert isinstance(floor.at(loc=loc), Wall)  # 该位置的元胞在第一层必须是wall类
            floor.set(loc=loc, cell=Free)   # 将该位置的元胞设置为free
            stair.field_connection[cell] = floor
        floor.stair = stair # 
    
    @staticmethod
    def attach_stair(floor, stair:Stair):    # 本方法应在init_wall后调用
        for cell in stair.stair_in:  # 迭代所有的stair_in坐标
            loc = cell.loc
            assert floor.check_loc(loc=loc) # 检查该坐标必须存在于floor场内 
            assert isinstance(floor.at(loc=loc), Wall)  # 该位置必须为一个wall对象
            floor.set(loc=loc, cell=Tran)
            stair.field_connection[cell] = floor
            floor.field_connection[cell] = stair
        floor.stair = stair # g

class Stair(Field):
    def __init__(self, x_range: tuple, y_range: tuple, id: int = 1, kappaD: int = 1, kappaS: int = 2, delta: float = 0.3, alpha: float = 0.1, cellsize: float = 0.4) -> None:
        super().__init__(x_range, y_range, id, kappaD, kappaS, delta, alpha, cellsize)
        self.stair_in = []
        self.field_connection = {}

    def set_stair_in(self, stari_in_loc:list) :
        for loc in stari_in_loc:
            assert self.check_loc(loc=loc)
            self.set(loc=loc, cell=Free)
            self.stair_in.append(self.at(loc=loc))
    
    def ped_movein(self, cell:Cell):
        loc = cell.loc
        tran = self.at(loc = loc)
        assert isinstance(tran, Tran)
        # 只有当前tran cell为空时才能让ped进入
        if tran.free:
            tran.ped = cell.ped
            cell.ped = None
            return True
        else: return False
        
    def seq_updata_cells(self, shuffle):
        tmp_ped = copy.deepcopy(self.ped)
        while tmp_ped:
            ped = tmp_ped.pop(0)
            loc = ped.get_loc   # 获取当前ped的坐标
            cell = self.at(loc=loc) # 根据loc取出该元胞
            # 判断cell的类型
            assert cell.cname != "Wall", f"Ped{ped.__str__()} unexpectedly move into a Wall cell"
            # 如果cell为exit 则应该调用场的ped_tran_field方法
            if isinstance(cell, Exit):
                self.
                self.remove_ped_by_id(id=ped.get_id)
                cell.dff_diff += 1
                # self.show_obst()
            # 如果cell为free 则进行详细的移动概率计算
            elif isinstance(cell, Free):
                prob, probs = 0, {}  # 定义当前ped移动的总概率和各个邻居元胞的移动概率
                # 迭代当前ped所处元胞的所有的free exit邻居元胞
                for neighbor in self.find_walkable_neighbor(cell=cell):
                    # 判断元胞是否被占据
                    if neighbor.free:
                        # 计算概率
                        probability = math.pow(math.exp(self.kappaS * (cell.sff - neighbor.sff)), 2)#  * math.exp(self.kappaD * (neighbor.dff - cell.dff))
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
                            # self.show_obst()
                            # 只要ped进行了一次移动就应该弹出当前循环
                            break
            else: raise TypeError(f"{ped} located on a unexpexted type of Cell: {cell.__class__}")


if __name__ =="__main__":
    # ped_loc = [[1,1], [2,3], [6,2]]
    
    
    exit_loc = [[10, 15], [11,15]]
    s = Stair(x_range=[9,16], y_range=[15, 22])
    s_in = [[13,15], [14,15]]
    s.init_walls()
    s.set_walls(wall_loc=[[12,15],[12,16],[12,17],[12,18]])
    s.init_exit(exit_cells=exit_loc)
    s.set_stair_in(s_in)
    s.init_peds(n_peds=3)
    s.init_sff()
    s.show_sff()
    s.show_obst()
    
    
    exit_loc_f1 = [[6,0]]
    f1 = Field((0,16), (0,16))
    f1.init_walls()
    f1.init_exit(exit_cells=exit_loc_f1)
    Field.base_floor_attach_stair(f1, s)
    Field.attach_stair(f1, s)
    f1.init_peds(2, loc=[[5,1],[7,1]])
    f1.init_sff()
    f1.show_obst()
    
    
   
    
    
