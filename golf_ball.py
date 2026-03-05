import numpy as np
from typing import Optional
from ball_state import BallState


class GolfBall:
    def __init__(self, position=None):
        self.__data: list[Optional[BallState]] = []
        self.__nextData: Optional[BallState] = None
        self.framesWithoutBall = 0
        self.framesWithBall = 0
        self.__lastKnownPosition: Optional[BallState] = None

        self.addData(position)

    def addData(self, data=None):
        if data is not None:
   
            if len(self.__data) == 0 or self.__data[-1] is None:
                dx, dy = 0, 0
            else:
                last = self.__data[-1]
                dx = data['x'] - last.x
                dy = data['y'] - last.y

            temp = BallState(
                x = data['x'], 
                y = data['y'], 
                dx = dx, 
                dy = dy, 
                speed = np.hypot(dx, dy)
            )

            self.__data.append(temp)
            self.predictNextData()
            self.framesWithoutBall = 0
            self.framesWithBall += 1

            self.__lastKnownPosition = temp
        else:
            self.__data.append(None)
            self.__nextData = None
            self.framesWithoutBall += 1
            self.framesWithBall = 0
      
    def getData(self, index):
        if 0 <= index < len(self.__data):
            return self.__data[index]
        return None
    
    def getCurrentData(self):
        return self.__data[-1] if self.__data else None 
    
    def getNextData(self):
        return self.__nextData
    
    def predictNextData(self):
        if len(self.__data) < 2:
            self.__nextData = None
            return
        else:
            lastLastData, lastData = self.__data[-2], self.__data[-1]
            
            if lastLastData is None or lastData is None:
                self.__nextData = None
                return
            
            # Simple linear extrapolation for next position
            dx = lastData.x - lastLastData.x
            dy = lastData.y - lastLastData.y
            # Assuming constant speed and direction for next prediction          
            self.__nextData = BallState(
                x = lastData.x + dx, 
                y = lastData.y + dy, 
                dx = dx, 
                dy = dy, 
                speed = np.hypot(dx, dy)
            )
    
    def getLastKnownPosition(self):
        
        return self.__lastKnownPosition
