import utilies
from ped import Ped
from utilies import cstr

# 基本元胞类
class Cell(object):

    def __init__(self, x:int, y:int, sff:float = None, dff:float = None, ped:Ped = None) -> None:
        self.__x = x
        self.__y = y
        self.__sff = sff    # 静态场不提供额外的修改方法
        self.__dff = dff    # 动态场提供单独的set 方法
        self.__classname = "Cell"
        self.__ped = ped
    
    # 暂时不提供x y 的set方法 即元胞一旦被创建则无法修改其位置
    def get_x(self): return self.__x

    def get_y(self): return self.__y

    def get_loc(self): return (self.__x, self.__y)

    def get_classname(self):return self.__classname

    def __eq__(self, target: object) -> bool:
        return self.loc() == target.loc()
    
    def __str__(self) -> str:
        return f"Cell:{(self.__x, self.__y)}, sff={self.__sff}, dff:{self.__dff}"
    
# 墙体类
class Wall(Cell):
    def __init__(self, x:int, y:int, sff:float=None, dff:float=None) -> None:
        super().__init__(x, y, sff, dff)
        self.__x = x
        self.__y = y
        self.__sff = sff
        self.__dff = dff
        self.__classname = "Wall"
    
    def get_classname(self):return self.__classname
    
    def __str__(self) -> str:
        return f"Wall:{(self.__x, self.__y)}, sff={self.__sff}, dff:{self.__dff}"

class Exit(Cell):
    def __init__(self, x:int, y:int, sff:float = None, dff:float = None, ped:Ped = None) -> None:
        super().__init__(x, y, sff, dff, ped)
        self.__x = x
        self.__y = y
        self.__sff = sff
        self.__dff = dff
        self.__classname = "Exit"
        self.__ped = ped

    def get_classname(self):return self.__classname
    
    def get_ped(self): return self.__ped
    
    # 修改动态场的值
    def set_dff(self, value:float): self.__dff = value
    
    def init_ped(self, ped:Ped):
        # 初始化传入的ped必须是非none
        assert ped is not None, cstr(f"{self.__str__()}, you cannot initizied a None ped into it")
        # 如果传入的ped非none 则直接将该值赋给本元胞
        self.__ped = ped
        
    def insert_ped(self, ped = None):
        # 仅供测试 直接插入一个ped
        # 插入ped之前需要先查看当前exit是否存在一个ped
        assert self.__ped is None, cstr(f"{self.__str__()} already occupied by a Ped, cannot call insert_ped", 0)
        # 如果当前exit没有ped 并且当前并没有传入一个ped
        if ped is None: self.__ped = Ped(id=0, location=(self.__x, self.__y))
        else:
            # 如果当前有一个传入对象
            assert(isinstance(ped, Ped)), cstr(f"You can only initized a Ped object in a Cell, while given {ped.__class__}")
            # 当前传入对象的x y必须和当前cell的x y一致
            assert ped.get_loc == self.get_loc()
            self.__ped = ped
        
    def ped_movein(self, ped:Ped):
        # ped进入当前cell的前提是当前cell没有ped
        assert self.__ped is None, f"{self.__str__()} alreay occupied by {self.__ped}, cannot let other ped move in"
        # 当一个ped进入某元胞之后需要将该ped的loc设置为当前元胞的loc
        ped.set_loc(self.get_loc())
        self.__ped = ped
    
    def ped_exit(self):
        # 只有当前cell的存在一个非none的ped对象时 才能调用本方法
        assert self.__ped is not None, cstr(f"{self.__str__()} has no ped to exit", 0)
        # 存在一个非none的ped对象时 直接将本元胞的ped设置为none
        self.__ped = None
        
    def __str__(self) -> str:
        return f"Exit:{(self.__x, self.__y)}, sff={self.__sff}, dff:{self.__dff}, ped:{self.__ped.__str__()}"
        
        

class Free(Cell):
    def __init__(self, x, y, sff=None, dff=None, ped = None) -> None:
        super().__init__(x, y, sff, dff)
        self.__x = x
        self.__y = y
        self.__sff = sff
        self.__dff = dff
        self.__classname = "Free"
        self.__ped = ped
    
    def get_classname(self):return self.__classname
    
    # ped方法
    def get_ped(self):return self.__ped
    
    # 初始化一个ped
    def init_ped(self, ped_id:int = None):
        # 为元胞初始化一个ped的前提是本元胞不能含有非none的ped
        assert self.__ped is None, f"{self.__str__()} has already been initized with {self.__ped}"
        self.__ped = Ped(id=ped_id, location=self.get_loc())
    
    # 插入一个ped
    def insert_ped(self, ped = None):
        # 仅供测试 直接插入一个ped
        # 插入ped之前需要先查看当前exit是否存在一个ped
        assert self.__ped is None, cstr(f"{self.__str__()} already occupied by a Ped, cannot call insert_ped", 0)
        # 如果当前exit没有ped 并且当前并没有传入一个ped
        if ped is None: self.__ped = Ped(id=0, location=(self.__x, self.__y))
        else:
            # 如果当前有一个传入对象
            assert(isinstance(ped, Ped)), cstr(f"You can only initized a Ped object in a Cell, while given {ped.__class__}")
            # 当前传入对象的x y必须和当前cell的x y一致
            assert ped.get_loc == self.get_loc()
            self.__ped = ped
    
    # ped进入元胞
    def ped_movein(self, ped:Ped):
        # 传入ped必须非none
        assert ped is not None, f"Try to move a None Ped into {self.__str__()}"
        # 当前cell不能有非none的ped对象
        assert self.get_ped() is None, f"{self.__str__()} already occupied by {self.get_ped()}"
        # ped进入某元胞之后需要将该ped的loc设置为当前元胞的loc
        ped.set_loc(self.get_loc())
        self.__ped = ped
        
    
    # ped走出元胞
    def ped_moveout(self, ped_target_cell):
        # 只有当前cell存在一个非none的ped对象时才能调用本方法
        assert self.get_ped() is not None, f"{self.__str__()} has No Ped object to Move"
        # 直接调用目标元胞的movein方法，如果目标元胞无法接受movein则直接报错 如果目标元胞可以接受movein则程序正常运行
        ped_target_cell.ped_movein()
        # 如果没有报错则将本元胞的ped设置为none
        self.__ped = None
        
    def __str__(self) -> str:
        return f"Free:{(self.__x, self.__y)}, sff={self.__sff}, dff:{self.__dff}, ped:{self.__ped.__str__()}"
    
