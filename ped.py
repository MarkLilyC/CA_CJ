from utilies import cstr

class Ped(object):
    def __init__(self, id:int, location:tuple|list):
        assert len(location) == 3 or len(location) == 2, cstr(f'The input location must have 3 or 2 coordinates, while given {len(location)}', 0)
        self.__id = id
        self.__location = location
        self.__x = location[0]
        self.__y = location[1]
        self.__z = location[2] if len(location) == 3 else None

    @property
    def get_id(self): return self.__id
    def set_id(self, new_id): self.__id == new_id
    @property
    def get_loc(self): return self.__location
    def set_loc(self, new_loc): 
        assert hasattr(new_loc, '__iter__')
        self.__location = new_loc
    


