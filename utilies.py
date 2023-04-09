import os


LOGCOLOR = {
    0:'\033[0;30;41m',
    1:'\033[0;30;42m',
    2:'\033[0;30;43m'
}

class cstr():
    def __init__(self, input_str, mode = 1):
        self.LOGCOLOR = {
                0:'\033[0;30;41m',
                1:'\033[0;30;42m',
                2:'\033[0;30;43m'
            }
        self.__str = self.LOGCOLOR[mode] + str(input_str) + '\033[0m'
    def __str__(self) -> str:
        return self.__str
    
        
def contain(original:list|tuple, target):
    # 如果传入的是可迭代对象
    if hasattr(target, '__iter__') and isinstance(target, str) is not True:
        # 如果传入对象中还有元素
        if len(target) > 0:
            # 则取出最后一个对象传入本函数进行判断
            flag = contain(original=original, target=target[-1])    
            if flag:    # 如果上一步返回true 则将传入对象去掉最后一个元素后继续传入本函数
                target.pop()
                
                tmp = contain(original=original, target=target)
                return tmp
            else: return False  # 如果上一步返回false 则直接本函数返回false
        else:   # 如果已经完成迭代 则返回true
            return True
    # 如果传入的是单一对象
    else:
        for i in original:
            if target == i:return True
            else: pass
        return False

def moke_movie(img_folder_path:str):
    pass