import utilies
from cell2 import *
from field import Field


class Building(object):
    def __init__(self, id):
        self.__id = id
        self.__floors = {}
        self.__floors_index = []
    
    @property
    def get_id(self): return self.__id
    def set_id(self, new_id): self.__id = new_id
    
    def get_floors_index(self): return self.__floors.keys()
    
    def add_floor(self, field:Field):
        id = field.get_id()
        # 不能有重复id的楼层
        if utilies.contain(self.get_floors_index(), id):
            print(f"Id:{id} already exists ")
        else:
            self.__floors[id] = field
    
    def order_floor(self):
        indics = self.get_floors_index()
        tmp = {}
        for i in range(1, len(indics) + 1):
            tmp[i] = self.__floors[i]
        self.__floors = tmp

x_range = [0, 12]
y_range = [0, 12]

b = Building(id=0)
f1 = Field(x_range=x_range, y_range=y_range, id=1)
f2 = Field(x_range=x_range, y_range=y_range, id=2)
f3 = Field(x_range=x_range, y_range=y_range, id=3)
b.add_floor(f1)
b.add_floor(f2)
b.add_floor(f3)
b.order_floor()