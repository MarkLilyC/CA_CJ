import utilies
from ped import Ped
class Cell(object):
    def __init__(self, x:int, y:int, sff:float = 0, dff:float = 0, ped:Ped = None) -> None:
        self.x = x
        self.y = y
        self.sff= sff
        self.dff = dff
        self.cname = "Cell"
        self.ped = ped
        self.dff_diff = 0
    
    @property
    def free(self): return True if self.ped is None else False
    
    @property
    def loc(self): return (self.x, self.y)
        
    def neighbor_loc(self) -> list[list]:
        '''计算当前元胞的所有邻居元胞坐标
            * 内部没有详细的逻辑判断 
            * 只是单纯返回了坐标意义上的邻居
        Returns:
            list: [上 下 左 右]元胞的坐标
        '''
        res = []
        x, y = self.x, self.y
        # 上侧元胞
        if x - 1 >= 0: res.append([x - 1, y])
        # 下侧元胞
        res.append([x + 1, y])
        # 左侧元胞
        if y - 1 >= 0: res.append([x, y - 1])
        # 右侧元胞
        res.append([x, y + 1])
        return res

    def init_ped(self, ped_id:int = 0):
        assert self.cname != "Wall", "Wall object can call this func"
        assert self.free, f"{self.__str__()} is occupied."
        self.ped = Ped(id=ped_id, location=self.loc)
    
    def insert_ped(self, ped:Ped):
        assert self.cname != "Wall", "Wall object can call this func"
        assert self.free, f"{self.__str__()} is occupied."
        if ped is None: self.init_ped()
        else:
            assert isinstance(ped, Ped), f"Can insert {ped.__class__} into cell"
            assert ped.get_loc == self.loc
            self.ped = ped
            
    def ped_movein(self, ped:Ped):
        assert self.cname != "Wall", "Wall object can call this func"
        assert self.free, f"{self.__str__()} is occupied"
        ped.set_loc(self.loc)
        self.ped = ped  
    
    def ped_moveout(self, target_cell):
        assert self.cname != "Wall", "Wall object can call this func"
        assert target_cell.free, f"{target_cell.__str__()} is occupied"
        target_cell.ped_movein(self.ped)
        self.ped = None

    def __hash__(self) -> int:
        res = str(self.x) + str(self.y) + str(self.y) + str(self.x)
        return int(res)

    def __eq__(self, target: object) -> bool:
        return self.loc() == target.loc()
    
    def __str__(self) -> str:
        return f"{self.cname}:{self.loc}, sff={self.sff}, dff:{self.dff}, ped:{self.ped}"
    
class Wall(Cell):
    def __init__(self, x: int, y: int, sff: float = 0, dff: float = 0, ped: Ped = None) -> None:
        super().__init__(x, y, sff, dff, ped)
        self.cname = 'Wall'
    
    @property
    def free(self): return False

class Exit(Cell):
    def __init__(self, x: int, y: int, sff: float = 0, dff: float = 0, ped: Ped = None) -> None:
        super().__init__(x, y, sff, dff, ped)
        self.sff = 0
        self.cname = 'Exit'
    
    def ped_exit(self, target_field = None):
        assert self.ped is not None, f"{self.__str__()} has no ped to exit"
        # 如果没有指定当前cell的ped前往哪一个场则直接让当前cell的ped消失
        if target_field is None:
            self.ped = None
        # 如果指定当前cell的ped前往某一个特定的场 则调用该场的receive_ped方法
        else:
            target_field.receive_ped(cell = self)

class Tran(Exit):
    def __init__(self, x: int, y: int, sff: float = 0, dff: float = 0, ped: Ped = None) -> None:
        super().__init__(x, y, sff, dff, ped)
        self.sff = 0
        self.cname = 'Tran'
    
class Stair(Cell):
    '''
    Stair元胞即普通楼层场与步梯的连接处的场，其特点在于：
        * 本类元胞属于可移动元胞
        * 本类元胞的静态场值在楼层场内和普通free元胞一致
        * 本类元胞的静态场值在楼道场内为exit
    '''
    def __init__(self, x: int, y: int, sff: float = 0, dff: float = 0, ped: Ped = None) -> None:
        super().__init__(x, y, sff, dff, ped)
        

class Free(Cell):
    def __init__(self, x: int, y: int, sff: float = 0, dff: float = 0, ped: Ped = None) -> None:
        super().__init__(x, y, sff, dff, ped)
        self.cname = 'Free'