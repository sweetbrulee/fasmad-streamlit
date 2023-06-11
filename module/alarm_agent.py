import math
import queue
import time
from service.message_queue import MetadataQueueService

metadata_queue = MetadataQueueService.use_queue()

#位置
class Pos:
    x=0
    y=0
    def __init__(self,x,y):
        self.x=x
        self.y=y

#帧结果
class FrameResult:
    posDown=Pos(0,0)
    posUp=Pos(0,0)
    believe=0
    isFire=False
    def __init__(self,posDown,posUp,believe,isFire):
        self.posDown=posDown
        self.posUp=posUp
        self.believe=believe
        self.isFire=isFire

    def isDanger(self):
        return abs((self.posUp.x-self.posDown.x)*(self.posUp.y-self.posDown.y))>0 and self.isFire==True and believe>=0.6

#每秒结果
class TreeSecResult:
    frameResultList=list()
    def __init__(self,frameResultList):
        self.frameResultList=frameResultList

    def isDanger(self):
        return __countDangerFrame(self)/len(self.frameResultList)>0.5

    def __countDangerFrame(self):
        count=0
        for i in self.frameResultList:
            if i.isDanger():
                count+=1
        return count

#处理逻辑
#处理队列
#从检测到有火灾可能开始计算时间
#如果有火灾风险则开始将3秒为一个单位将结果进行判断
rQueue=queue.Queue()
def alarm():
    treeSecResult=rQueue.get()
    if(treeSecResult.isDanger()):
        print("有火警!")
        del treeSecResult
        #time.sleep(3)响铃3秒


