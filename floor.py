import utilies
from utilies import cstr
import numpy as np
import random
import itertools as it
class Floor(object):

    def __init__(self, id, width, height, cellsize, z = 0.):
        self.__id = id
        self.__z = z
        self.__dim_x = int(height / cellsize + 2 + 0.00000001)  # x方向的格子数 包含两侧的墙体
        self.__dim_y = int(width / cellsize + 2 + 0.00000001)   # y方向的格子数
        self.__x_range = [1, self.__dim_x - 2]  # 元胞可移动的x轴范围
        self.__y_range = [1, self.__dim_y - 2]  # 元胞可移动的y轴范围
        self.__exci_cells = None    # 当前楼层的疏散出口
        self.__sff = None   #   当前楼层的静态场
        self.dff = None 
        self.ped = None
        self.__obst = np.ones((self.__dim_x, self.__dim_y), int)    # 初始化一个场 

        print(f"Floor initialized: ID: {self.__id}, dim_x: {self.__dim_x}, dim_y: {self.__dim_y}")

    @property
    def get_id(self): return self.__id
    def set_id(self, new_id): self.__id = new_id
    
    @property
    def get_z(self):return self.__z

    @property
    def get_x_range(self): return self.__x_range

    @property
    def get_y_range(self): return self.__y_range

    def get_sff(self): return self.__sff

    def get_exit_cells(self): return self.__exci_cells

    def set_exit_cells(self, exit_cells = None):
        if exit_cells is None:
            exit_cells = frozenset((
                            (self.__dim_x // 2, self.__dim_y - 1), (self.__dim_x // 2 + 1, self.__dim_y - 1),
                            (self.__dim_x - 1, self.__dim_y//2 + 1) , (self.__dim_x - 1, self.__dim_y//2),
                            (0, self.__dim_y//2 + 1) , (1, self.__dim_y//2),
                            (self.__dim_x//2 + 1, 0) , (self.__dim_x//2, 0)
         ))
        # 当前楼层的疏散出口位置一定是位于obst的边界上
        # 即cell的x 或 y坐标当中有一个必须是落在obst的边界上的
        '''
        for cell in exit_cells:
            x, y = cell
            flag1 = x == 0 or x == self.__dim_x - 1
            flag2 = y == 0 or y == self.__dim_y - 1
            if flag1 or flag2:pass
            else: print(cstr(f"the exit_cell must locate at the edge of obst, obst's x_range:0 and {self.__dim_x - 1}, obst's y_range:0 and {self.__dim_y - 1}, while given cell:{cell}",0))
        '''
        self.__exci_cells = exit_cells
        print(cstr(f"Floor {self.__id} set exit_cells finished", 1))

    def get_obst(self): return self.__obst

    def get_neighbors(self, cell) -> list:
        '''获取当前元胞的可前进元胞 判断的基础是初始化墙体后的obst

        Args:
            cell (_type_): _description_

        Returns:
            list: _description_
        '''
        neighbors = []
        # 分别对当前cell的上下左右四个元胞进行判断
        x, y = cell
        obst = self.__obst
        # 左侧元胞  当前元胞如果位于obst的左侧边界 或者当前元胞的左侧元胞是墙体 则不为当前元胞添加左侧邻居
        if x >= 1 and obst[(x - 1, y)] >= 0:neighbors.append((x - 1, y))
        # 右侧元胞  当前元胞如果位于obst的右侧边界 或者当前元胞的右侧元胞是墙体 则不添加右侧元胞
        if x < self.__dim_x - 1 and obst[(x + 1, y)] >= 0:neighbors.append((x + 1, y))
        # 下侧元胞  当前元胞如果位于obst的下侧边界 或者当前元胞的下侧元胞是墙体 则不添加下侧元胞
        if y >= 1 and obst[(x, y - 1)] >= 0:neighbors.append((x, y - 1))
        # 上侧元胞  当前元胞如果位于obst的上侧边界 或者当前元胞的上侧元胞是墙体 则不添加上册元胞
        if y < self.__dim_y - 1 and obst[(x, y + 1)] >= 0:neighbors.append((x, y + 1))
        random.shuffle(neighbors)
        return neighbors

    def get_ped_capacity(self):
        # x方向除去墙体外元胞个数
        n_cell_x = self.__x_range[1] - self.__x_range[0] + 1 
        n_cell_y = self.__y_range[1] - self.__y_range[0] + 1 
        # 除去墙体外的总元胞个数
        return n_cell_x * n_cell_y

    def check_ped(self, n_ped):
        n_cell_total = self.get_ped_capacity()
        if n_ped > n_cell_total: 
            print(f"Given number of Ped-{n_ped} is too large for the current floor's max Ped capacity-{n_cell_total}")
            return n_cell_total
        else: return n_ped


    def init_wall(self):
        '''初始化当前楼层的墙体 在楼层的obst上直接做修改

        Returns:
            _type_: _description_
        '''
        if self.__exci_cells is None:print(f"Floor {self.__id} haven't set any exit_cell", 2)
        # 当前obst中的元胞值都为1
        # 向将obst中的外层元胞赋值为-1 代表墙体
        self.__obst[0, :] = self.__obst[-1, :] = self.__obst[:, -1] = self.__obst[:, 0] = -1
        # 再根据当前楼层的exit——cells将墙体对应位置的元胞赋值为1
        for cell in self.__exci_cells:
            self.__obst[cell] = 1
        return self.__obst
    
    def init_sff(self):
        '''初始化当前楼层的静态场 static floor field
        '''
        # 定义一个初始的sff 大小obst一致
        sff = np.empty((self.__dim_x, self.__dim_y))
        # 初始化静态场, 此时静态场的值全部为sqrt(self.__dim_x ** 2 + self.__dim_y ** 2)
        sff[:] = np.sqrt(self.__dim_x ** 2 + self.__dim_y ** 2)
        # 定义一个临时变量存储exitcells
        tmp_exitcells = []
        for cell in self.__exci_cells:
            tmp_exitcells.append(cell)
            sff[cell] = 0   # 将静态场中出口对应位置的值修改为0
        # 迭代所有的出口元胞
        while tmp_exitcells:
            # 取出第一个元胞
            cell = tmp_exitcells.pop(0)
            # 找出当前元胞的邻居元胞
            neighbor_cells = self.get_neighbors(cell=cell)
            # 迭代当前的邻居元胞
            for neighbor in neighbor_cells:
                # 如果当前元胞的sff值+1 < 当前邻居的静态sff值 
                if sff[cell] + 1 < sff[neighbor]:
                    # 则将当前邻居的sff值修改为当前元胞的sff值+1 相当于将非墙体的元胞(ped可行进的元胞)的sff值修改为该元胞距离出口的距离
                    sff[neighbor] = sff[cell] + 1
                    # 并且将当前的邻居元胞添加到tmp列表中继续迭代 需按照当前邻居元胞的邻居元胞 并且修正其值
                    tmp_exitcells.append(neighbor)
        # sff值修改完成后 将sff赋值给当前的楼层
        self.__sff = sff
        return sff

    def init_peds(self, n_peds):
        from_x, to_x = self.__x_range
        from_y, to_y = self.__y_range
        n_x = to_x - from_x + 1
        n_y = to_y - from_y + 1
        ped_capacity = self.get_ped_capacity()
        n_peds = self.check_ped(n_peds)
        peds = np.ones(n_peds, int) #   生成一个n长的以为向量
        empty_cell_obst = np.zeros(ped_capacity - n_peds, int) # 剩余的可行进元胞个数
        peds = np.hstack((peds, empty_cell_obst))   # 将ped和剩余元胞拼接起来
        np.random.shuffle(peds)  # 打乱顺序
        peds = peds.reshape((n_x, n_y)) # 重新调整为obst的形状（除去墙体）
        total_cells = np.zeros((self.__dim_x, self.__dim_y), int)   # 新建一个obst形状的矩阵
        total_cells[from_x:to_x + 1, from_y:to_y + 1] = peds    # 将ped和空元胞放到对应位置
        self.ped = total_cells
        print(cstr(f"Init {n_peds} finished"))
    
    def init_dff(self):
        self.dff = np.zeros((self.__dim_x, self.__dim_y))
        
    def update_dff(self, dff_diff):
        self.dff += dff_diff
        # for i, j in it.chain(it.product(range(1)))
    
    def init_sim(self):
        self.set_exit_cells()
        self.init_wall()
        self.init_sff()
        self.init_peds(10)
        self.init_dff()
    
    def seq_updata_cells(self, kappaS, kappaD, shuffle, reverse):
        # 创建一个临时ped
        tmp_peds = np.empty_like(self.ped)
        # 拷贝赋值
        np.copyto(tmp_peds, self.ped)
        dff_diff = np.zeros(tmp_peds.shape)
        grid = list(it.product(range(1, self.__dim_x - 1), range(1, self.__dim_y - 1))) + list(self.__exci_cells)   # 所有的可行进元胞（内部元胞+出口元胞）
        if shuffle:  # sequential random update
            random.shuffle(grid)
        elif reverse:  # reversed sequential update
            grid.reverse()
        # 迭代所有可行进元胞
        for cell in grid:
            # 如果该位置没有行人 则跳出循环
            if self.ped[cell] == 0:continue
            # 以下逻辑建立在当前位置有行人的前提下
            # 如果当前位置为出口 则行人下一秒会走出整个空间
            if cell in self.__exci_cells:
                # 因此需要将ped矩阵该位置赋值为0
                tmp_peds[cell] = 0
                dff_diff[cell] = 1
                continue
            # 如果当前位置不是出口 则需要计算移动概率
            prob = 0
            probs = {}
            neighbors = self.get_neighbors(cell)
            for neighbor in neighbors:
                # 计算朝某一个邻居的行进概率
                probability = np.exp(kappaS * (self.__sff[cell] - self.__sff[neighbor])) * \
                          np.exp(kappaD * (self.dff[neighbor] - self.dff[cell])) * \
                          (1 - tmp_peds[neighbor])
                prob += probability
                probs[neighbor] = probability
            # 如果概率总和为0  则无法移动 跳出循环
            if prob == 0:continue
            
            r = np.random.rand * prob
            for neighbor in neighbors:
                r -= probs[neighbor]
                # 如果概率值小于0 则移动到元胞
                if r <= 0 :
                    tmp_peds[neighbor] = 1  # 将ped中该邻居元胞的值设置为1
                    tmp_peds[cell] = 0  # 将ped中该元胞的值设置为0
                    dff_diff[cell] += 1
                    break
        return tmp_peds, dff_diff
    
    def sim(self, steps:int):
        for t in steps:
            print(f"Floor {self.__id} Sim, t:{t}")
            
        

    def __eq__(self, target):
        assert hasattr(target, 'get_z'), cstr("the target should have a Z attribute", 0)
        assert hasattr(target, 'get_id'), cstr("the target should have a id attribute", 0)
        if self.get_z == target.get_z or self.get_id == target.get_id: return True
        else: return False

    def __str__(self):
        return f"Floor {self.get_id} at z = {self.get_z}"
        


