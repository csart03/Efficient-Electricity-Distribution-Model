#Problem Statement
#Given a city structure 
from collections import defaultdict
class citytoken:
	def __init__(self,x,y,label):
		self.x=x
		self.y=y
		self.label=label
graph = defaultdict(list)
def addEdge(graph,u,v,weight):
    graph[u].append([v,weight])
    graph[v].append([u,weight])
def generate_edges()

def generatecostofwire():
	#this will be a function of distance between two nodes

citytokens=[]
def constructgraph():
	#u and v will be of the type city token
	addEdge(graph,u,v,weight)
	#this function will construct the graph and assign weights to the edges

