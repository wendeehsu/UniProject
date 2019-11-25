import math
import matplotlib.pyplot as plt
import pickle
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def distance(p1,p2):
	return (p1[0]-p2[0])*(p1[0]-p2[0]) + (p1[1]-p2[1])*(p1[1]-p2[1])

def Show(pointList):
	for i in range(len(pointList)):
		plt.plot(pointList[i][0],pointList[i][1], color='blue', marker='o')
	plt.show()

def ShowLine(pointList):
	for i in range(len(pointList)-1):
		plt.plot([pointList[i][0],pointList[i+1][0]],[pointList[i][1],pointList[i+1][1]])
	plt.show()

def ShowLineWithoutNoise(pointList, noise = 50):
	for i in range(len(pointList)-1):
		if distance(pointList[i],pointList[i+1]) <= noise:
			plt.plot([pointList[i][0],pointList[i+1][0]],[pointList[i][1],pointList[i+1][1]])
	plt.show()

def ShowPoint(p):
	plt.plot(0,0)
	plt.plot(p[0],p[1], color='red', marker='o')
	plt.show()

def GetStart(points): 	# 找起點
	point = [0,0]
	for p in points:
		dis = p[0] + p[1]
		if dis > (point[0] + point[1]):
			point = p
	return point

def SetMatrix(points):
	zeors_array = np.zeros((MaxSize,MaxSize))
	for p in points:
		zeors_array[p[0]][p[1]] = 1
	return zeors_array

def CleanPoints(pos, nestPos, matrix):
	if nestPos[0] > pos[0]:     # 往右走  
		if nestPos[1] > pos[1]: # 往上走
			for i in range(pos[0], nestPos[0]+1):
				for j in range(pos[1], nestPos[1]+1):
					matrix[i,j] = 0
		elif nestPos[1] == pos[1]:
			for i in range(pos[0], nestPos[0]+1):
				matrix[i,pos[1]] = 0
		else:                    # 往下走
			for i in range(pos[0], nestPos[0]+1):
				for j in range(nestPos[1], pos[1]+1):
					matrix[i,j] = 0
	elif nestPos[0] == pos[0]:
		if nestPos[1] > pos[1]:
			for j in range(pos[1], nestPos[1]+1):
				matrix[nestPos[0],j] = 0
		else:
			for j in range(nestPos[1], pos[1]+1):
				matrix[nestPos[0],j] = 0
	else:                        # 往左走  
		if nestPos[1] > pos[1]:  # 往上走
			for i in range(nestPos[0], pos[0]+1):
				for j in range(pos[1], nestPos[1]+1):
					matrix[i,j] = 0
		elif nestPos[1] == pos[1]:
			for i in range(nestPos[0], pos[0]+1):
				matrix[i,pos[1]] = 0
		else:                    # 往下走
			for i in range(nestPos[0], pos[0]+1):
				for j in range(nestPos[1], pos[1]+1):
					matrix[i,j] = 0
	return matrix

def ClearPointsInRange(pos, r, matrix):
	matrix[pos[0]][pos[1]] = 0
	for i in range(pos[0]-r, pos[0]+r+1):
		if i >= 0 and i < MaxSize:
			for j in range(pos[1]-r, pos[1]+r+1):
				if j >= 0 and j < MaxSize:
					if matrix[i][j] == 1 and distance([i,j],pos) <= r*r:  # get points in radius = r
						matrix[i][j] = 0
	return matrix

def GetNextP(pos, matrix, prevPos, r = 2): # 找下一個點
# 	print("pos = ", pos)
	matrix[pos[0]][pos[1]] = 0
	candidates = []
	for i in range(pos[0]-r, pos[0]+r+1):
		if i >= 0 and i < MaxSize:
			for j in range(pos[1]-r, pos[1]+r+1):
				if j >= 0 and j < MaxSize:
					if matrix[i][j] == 1 and distance([i,j],pos) <= r*r:  # get points in radius = r
						candidates.append([i,j])
# 	print("candidates = ", candidates)
	numOfCandidates = len(candidates)
	if numOfCandidates > 0:
		if numOfCandidates == 1:
			nextPos = candidates[0]
		else:
			prevVector = [pos[0] - prevPos[0], pos[1] - prevPos[1]]
			nextPos = ChoosePoint(candidates, pos, prevVector)
		
		matrix = ClearPointsInRange(pos,r,matrix)
		return nextPos, matrix
	else:
		if r >= 40:
			return pos, np.zeros((MaxSize,MaxSize))
		return GetNextP(pos, matrix, prevPos, r + r)

"""
1. find max cosines
2. if cosine value is the same: 找遠的（之後會把近的點刪掉）

TBD:
	方位同：找遠的（之後會把近的點刪掉）
	方位不同：找近的
"""
def ChoosePoint(candidates, pos, prevVector):
	vectors = list(map(lambda x : [x[0]-pos[0],x[1]-pos[1]], candidates))
	cosines = cosine_similarity([prevVector],vectors)[0]
	print("cosine_similarity = ", cosines)
	maxCosine = max(cosines)
	tempPoints, = np.where(cosines == maxCosine)
	tempPoint = candidates[tempPoints[0]]
	maxDistance = distance(tempPoint,pos)
	for i in tempPoints:
		if distance(candidates[i],pos) > maxDistance:
			tempPoint = candidates[i]
	return tempPoint

def DFS(points, r = 2, tolerance = 5):
	currentP = GetStart(points)
	rawMatrix = SetMatrix(points)
	lastP = [MaxSize-1,MaxSize-1]
	path = [currentP]
	while sum(sum(rawMatrix)) > 0:
		nextPos, rawMatrix = GetNextP(currentP, rawMatrix, lastP, r)
		path.append(nextPos)
		lastP = currentP
		currentP = nextPos
		if sum(sum(rawMatrix)) <= tolerance:
			break
	return path


points = []
MaxSize = 500

with open('points.pickle', 'rb') as file:
	points = pickle.load(file)

# show full graph
allpoints = []
for i in points:
	allpoints += i

# show parts
Show(allpoints)
path = DFS(allpoints)
ShowLine(path)
ShowLineWithoutNoise(path)
"""
with open('catpoints', 'rb') as file:
	points = pickle.load(file)
Show(points)
catPath = DFS(points)
ShowLine(catPath)
ShowLineWithoutNoise(catPath)
"""
