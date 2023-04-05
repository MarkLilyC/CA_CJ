import numpy as np

dim_x = 12
dim_y = 12

class a:
    def __init__(self, name):
      self.__name = name
    def get(self):
        return self.__name
    
a1 = a(1)
b = a1.get()

b = 12
print(a1.get())