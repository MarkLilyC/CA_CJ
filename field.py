from ped import Ped
from cell import *
import utilies

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
    
    def __init__(self, x_range:tuple, y_range:tuple) -> None:
        assert x_range[0] < x_range[1] and y_range[0] < y_range[1], f"Wrong coordinate range"
        self.__x_range = x_range
        self.__y_range = y_range
        self.cells = Field.init_cells(x_range, y_range)
    
    def get(self, loc:tuple|list):
        return self.cells[loc[0]][loc[1]]
    
    def set(self, loc, cell):
        self.cells[loc[0]][loc[1]] = cell
    
    def get_x_range(self):return self.__x_range
    
    def get_y_range(self):return self.__y_range
    
    def init_walls(self) -> None:
        # 将整个field的外层cell直接变为Wall Cell
        for i in range(self.__y_range[0], self.__y_range[1]-1): 
            self.cells[self.__x_range[0]][i] = Wall(x=0,y=i)
            
        for i in range(self.__y_range[0], self.__y_range[1]-1): 
            self.cells[0][i] = Wall(x=0,y=i)
        
        for i in range(1, self.__y_range[1] - 1): 
            self.cells[i][self.__x_range[0]] =  Wall(x=self.__x_range[0],y=i)       
        for i in range(1, self.__y_range[1] - 1): 
            self.cells[i][self.__x_range[1] - 1] =  Wall(x=self.__x_range[1] - 1,y=i)       
        
    def show_obst(self):
        res = ''
        for row in self.cells.values():
            for cell in row:
                if cell.get_classname() == "Wall": res += '-1, '
                elif cell.get_classname() == "Free": res += ' 0, '
                elif cell.get_classname() == 'Exit': res += ' 0, '
                else:raise ValueError("Wrong cell type in the filed")
            res += '\n'
        print(res)
    
    
    def __str__(self) -> str:
        return f"Field x_range:{self.__x_range}, y_range:{self.__y_range}"
    
    

f = Field((0,12), (0,12))
print(f)

f.init_walls()

f.show_obst()
