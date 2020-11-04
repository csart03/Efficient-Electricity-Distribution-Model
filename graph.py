#Problem Statement
#Given a city structure 
from collections import defaultdict
import numpy as np
from matplotlib import pyplot as plt
# from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
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

def knapsack01(W,city):
	K = [[0 for x in range(W + 1)] for x in range(n + 1)] 
    for i in range(n + 1): 
        for w in range(W + 1): 
            if i == 0 or w == 0: 
                K[i][w] = 0
            elif wt[i-1] <= w: 
                K[i][w] = max(val[i-1] + K[i-1][w-wt[i-1]],  K[i-1][w]) 
            else: 
                K[i][w] = K[i-1][w] 
  
    return K[n][W]

cityhouses=[]
no_of_houses=int(input("Enter the number of houses:-"))
print("Enter the coordinates of the houses one by one:-")
for i in range(no_of_houses):
	lh=input().split(',')
	cityhouses.append([float(lh[0]),float(lh[1])])

# print(cityhouses)
cityhouses=np.array(cityhouses)
no_of_transformers=int(input("Enter the number of transformers you wish to install:-"))
kmeans=KMeans(n_clusters=no_of_transformers)
kmeans.fit(cityhouses)
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
# plt.scatter(cityhouses[:,0],cityhouses[:,1])
# plt.scatter(citytranformers[:,0],citytranformers[:,1],marker='*',color='orange')
# plt.scatter(citysubstations[:,0], citysubstations[:,1],marker='v',color='red')
# plt.scatter(citysource[:,0], citysource[:,1],marker='s',color='blue')
# plt.show()
graph1=connecthousestotransformers(cityhouses, citytransformers, pred1,0,len(cityhouses))
graph2=connecttransformerstosubstations(citytransformers, citysubstations, pred2,len(cityhouses),len(cityhouses)+len(citytransformers))
graph3=connectsubstationstosource(citysubstations, citysource, 0,len(cityhouses)+len(citytransformers),len(cityhouses)+len(citytransformers)+1)
print(graph1)