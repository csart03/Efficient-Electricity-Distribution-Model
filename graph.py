#Problem Statement
#Given a city structure 
from collections import defaultdict
import numpy as np
from matplotlib import pyplot as plt
# from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
import pandas as pd
import pricing
class citytoken:
	def __init__(self,x,y,label):
		self.x=x
		self.y=y
		self.label=label

def addEdge(graph,u,v,weight):
    graph[u].append([v,weight,0])
    graph[v].append([u,weight,0])
# def generate_edges()

def generatecostofwire():
	pass
	#this will be a function of sdistance between two nodes

def constructgraph():

	#u and v will be of the type city token
	addEdge(graph,u,v,weight)
	#this function will construct the graph and assign weights to the edges

def euleriandistance(x,y):
	return np.sqrt(np.sum((x-y)**2))

def connecthousestotransformers(cityhouses,citytransformers,pred,x1,x2):
	graph = defaultdict(list)
	# x1=len(cityhouses)
	# x2=len(citytransformers)
	for i in range(len(cityhouses)):
		# print(cityhouses[i])
		# print(citytransformers[pred[i]])
		dis1=euleriandistance(cityhouses[i], citytransformers[pred[i]])
		addEdge(graph, x1+i, x2+pred[i], dis1)
	return graph

def connecttransformerstosubstations(citytransformers,citysubstations,pred,x1,x2):
	graph = defaultdict(list)
	for i in range(len(citytransformers)):
		dis1=euleriandistance(citytransformers[i], citysubstations[pred[i]])
		addEdge(graph, x1+i, x2+pred[i], dis1)
	return graph

def connectsubstationstosource(citysubstations,citysource,pred,x1,x2):
	graph = defaultdict(list)
	for i in range(len(citysubstations)):
		dis1=euleriandistance(citysubstations[i], citysource[0])
		addEdge(graph, x1+i, x2, dis1)
	return graph

def knapsackhouse(cityhouses,W):
	n=len(cityhouses)
	values=[]
	weights=[]
	for i in range(len(cityhouses)):
		weights.append(int(cityhouses[i][2]))
		if cityhouses[i][3]==2:
			values.append(cityhouses[i][3]*4.0)
		if cityhouses[i][3]==5:
			values.append(cityhouses[i][3]*4.5)
		if cityhouses[i][3]==10:
			values.append(cityhouses[i][3]*6.0)

	K = [[0 for x in range(W + 1)] for x in range(n + 1)] 

	for i in range(n):
		for w in range(W + 1): 
			if i == 0 or w == 0: 
				K[i][w] = 0
			elif weights[i] <= w: 
				K[i][w] = max(values[i] + K[i][w-weights[i]],  K[i][w]) 
			else: 
				K[i][w] = K[i][w] 
	res= K[n][W]
	w=W
	house_to_show=[]
	for i in range(n, 0, -1): 
		if res <= 0: 
			break
		if res == K[i - 1][w]: 
			continue
		else: 
			print(wt[i - 1]) 
			house_to_show.append(cityhouses[i-1])
			res = res - values[i - 1] 
			w = w - weights[i - 1]

	houses_to_show=np.array(houses_to_show)
	plt.scatter(houses_to_show[:,0],houses_to_show[:,1])
	plt.show()

def myFunc(e):
	return e[1]

def greedyalgo(cityhouses):
	amt=int(input('Enter the amount of electricity u need to distribute:-'))
	chs=[]
	for i in range(len(cityhouses)):
		chs.append((i+1,cityhouses[i][2]))
	sorted(chs,key=myFunc)
	ans1=0
	idx=-1
	for i in range(len(chs)):
		if amt <= 0:
			idx=i-1
			break
		ans1+=1
		amt-=chs[i][1]
	print('Total house that will receive electricity:-{}'.format(str(ans1)))
	houses_to_show=[]
	for i in range(0,idx+1):
		houses_to_show.append(cityhouses[chs[i][0]])
	houses_to_show=np.array(houses_to_show)
	plt.scatter(houses_to_show[:,0],houses_to_show[:,1])
	plt.show()
	print(graph1)
	# for i in range(len(graph1)):

def dynamicpricing():
	# Cc=0.0,Cd=0.0,Cg=0.0
	Cc=float(input('Enter price of coal per unit'))
	Cd=float(input('Enter price of diesel per unit'))
	Cg=float(input('Enter price of gas per unit'))	
	Pc=float(input('Enter percentage of coal usedused'))
	Pd=float(input('Enter percentage of diesel used'))
	Pg=float(input('Enter percentage of gas used'))
	price_one_unit_electricity=19.4
	# Cc*Pc+Cd*Pd+Cg*Pg
	df=pd.read_csv('sample.csv')
	df=np.array(df)
	x=df.shape[0]-170
	for i in range(0,x):
		consumption=df[i:168+i,7]
		consumption = consumption.astype(np.float)
		avgcon=np.sum(consumption)
		# avgcon=float(avgcon)
		avgcon/=float(168)
		D=float(df[169+i][7])
		D/=avgcon
		D*=price_one_unit_electricity
		print('Price of current hour :- {} on day and hour {}'.format(D,df[i+169][0]))

cityhouses=[]
no_of_houses=int(input("Enter the number of houses:-"))
print("Enter the coordinates of the houses one by one:-")
for i in range(no_of_houses):
	lh=input().split(',')
	cityhouses.append([float(lh[0]),float(lh[1]),int(lh[2]),int(lh[3])])
	# cityhouses.append([float(lh[0]),float(lh[1])])
# print(cityhouses)
cityhouses=np.array(cityhouses)
cityhouses1=cityhouses[:,0:2]
no_of_transformers=int(input("Enter the number of transformers you wish to install:-"))
kmeans=KMeans(n_clusters=no_of_transformers)
kmeans.fit(cityhouses1)
citytransformers=kmeans.cluster_centers_
pred1=kmeans.labels_
# print(pred1)
# print(citytranformers)
no_of_substations=int(input("Enter the number of substations you wish to install:-"))
kmeans=KMeans(n_clusters=no_of_substations)
kmeans.fit(citytransformers)
citysubstations=kmeans.cluster_centers_
pred2=kmeans.labels_
kmeans=KMeans(n_clusters=1)
kmeans.fit(citysubstations)
citysource=kmeans.labels_
plt.scatter(cityhouses[:,0],cityhouses[:,1])
# plt.scatter(citytranformers[:,0],citytranformers[:,1],marker='*',color='orange')
# plt.scatter(citysubstations[:,0], citysubstations[:,1],marker='v',color='red')
# plt.scatter(citysource[:,0], citysource[:,1],marker='s',color='blue')
plt.show()
graph1=connecthousestotransformers(cityhouses1, citytransformers, pred1,0,len(cityhouses1))
graph2=connecttransformerstosubstations(citytransformers, citysubstations, pred2,len(cityhouses),len(cityhouses)+len(citytransformers))
graph3=connectsubstationstosource(citysubstations, citysource, 0,len(cityhouses)+len(citytransformers),len(cityhouses)+len(citytransformers)+1)
# greedyalgo(cityhouses)
knapsackhouse(cityhouses, 500)
# print(graph1)
# dynamicpricing()
