import utilies
from floor import Floor
from utilies import cstr

class Building(object):
    def __init__(self, id):
        self.__id = id
        self.__floors = {}
    
    @property
    def get_id(self): return self.__id
    def set_id(self, new_id): self.__id = new_id
    
    def get_floors_ids(self):return list(self.__floors.keys())

    def get_floors(self):return self.__floors

    def get_floor(self, floor_id): return self.__floors[floor_id]

    def add_floor(self, new_floor:Floor):
        floors = list(self.get_floors().values())
        if utilies.contain(floors, new_floor):
            print(cstr(f"{new_floor} already exists in the building", 2))
            return False
        else:
            self.__floors[new_floor.get_id] = new_floor
            return True
    
    def sim(self):
        for i in self.__floors.values():
            pass
        
    def __str__(self):
        res = f'Building {self.get_id}, contains Floor: \n'
        for id, floor in self.get_floors().items():
            res += "    " + f"id: {id}, Floor: {floor.__str__()}"
        return res