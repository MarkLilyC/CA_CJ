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
    def free(self): return self.ped == None
    
    @property
    def loc(self): return (self.x, self.y)
        
    def neighbor_loc(self):
        '''
        reture [上 下 左 右]
        '''
        res = []
        x, y = self.x, self.y
        # 上侧元胞
        if x - 1 > 0: res.append([x - 1, y])
        # 下侧元胞
        res.append([x + 1, y])
        # 左侧元胞
        if y - 1 > 0: res.append([x, y - 1])
        # 右侧元胞
        res.append([x, y + 1])
        return res

    def init_ped(self, ped_id:int = 0):
        assert self.cname is not "Wall", "Wall object can call this func"
        assert self.free, f"{self.__str__()} is occupied."
        self.ped = Ped(id=ped_id, location=self.loc)
    
    def insert_ped(self, ped:Ped):
        assert self.cname is not "Wall", "Wall object can call this func"
        assert self.free, f"{self.__str__()} is occupied."
        if ped is None: self.init_ped()
        else:
            assert isinstance(ped, Ped), f"Can insert {ped.__class__} into cell"
            assert ped.get_loc == self.loc
            self.ped = ped
            
    def ped_movein(self, ped:Ped):
        assert self.cname is not "Wall", "Wall object can call this func"
        assert self.free, f"{self.__str__()} is occupied"
        ped.set_loc(self.loc)
        self.ped = ped  
    
    def ped_moveout(self, target_cell):
        assert self.cname is not "Wall", "Wall object can call this func"
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
    
    def free(self): return False

class Exit(Cell):
    def __init__(self, x: int, y: int, sff: float = 0, dff: float = 0, ped: Ped = None) -> None:
        super().__init__(x, y, sff, dff, ped)
        self.sff = 0
        self.cname = 'Exit'
    
    def ped_exit(self):
        assert self.ped is not None, f"{self.__str__()} has no ped to exit"
        self.ped = None

class Free(Cell):
    def __init__(self, x: int, y: int, sff: float = 0, dff: float = 0, ped: Ped = None) -> None:
        super().__init__(x, y, sff, dff, ped)
        self.cname = 'Free'