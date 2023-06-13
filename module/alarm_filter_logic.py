# 位置
class Pos:
    x = 0
    y = 0

    def __init__(self, x, y):
        self.x = x
        self.y = y


# 帧结果
class OneResult:
    type = -1
    posDown = Pos(0, 0)
    posUp = Pos(0, 0)
    believe = 0

    def __init__(self, type, posDown, posUp, believe):
        self.type = type
        self.posDown = posDown
        self.posUp = posUp
        self.believe = believe

    def isDanger(self):
        return (
            abs((self.posUp.x - self.posDown.x) * (self.posUp.y - self.posDown.y)) > 0
            and self.type != -1
            and self.believe >= 0.6
        )


class FrameResult:
    resultList = list()

    def __init__(self, resultList):
        self.resultList = resultList
        pass

    def isDanger(self):
        for i in self.resultList:
            if i.isDanger():
                return True
        return False


# 处理逻辑
# 处理队列
# 从检测到有火灾可能开始计算时间
# 如果有火灾风险则开始将3秒为一个单位将结果进行判断
def alarm(lis):
    count = 0
    frameResult = list()
    for i in lis:
        if i.isDanger():
            count += 1
            frameResult = i
    if count / len(lis) >= 0.5:
        return frameResult
    else:
        return list()


# 仅看下面即可
# 下面是对每个队列单位元素的逻辑判断
def isDanger(data):
    danger_count = 0
    for i in data:
        if abs((i[1][0] - i[1][2]) * (i[1][1] - i[1][3])) >= 1 and i[2] >= 0.5:
            danger_count += 1
    return danger_count


def isStranger(data):
    stranger_count = 0
    for i in data:
        if i == "unknown":
            stranger_count += 1
    return stranger_count
