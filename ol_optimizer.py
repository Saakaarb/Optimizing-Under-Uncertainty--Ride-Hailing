import numpy as np
import math
from math import floor


# Python program for Dijkstra's single 
# source shortest path algorithm. The program is 
# for adjacency matrix representation of the graph 

# Library for INT_MAX 
import sys 

class Graph(): 

	def __init__(self, vertices): 
		self.V = vertices 
		self.graph = [[0 for column in range(vertices)] 
					for row in range(vertices)] 

	def printSolution(self, dist): 
		print ("Vertex tDistance from Source") 
		for node in range(self.V): 
			print (node, "t", dist[node]) 

	# A utility function to find the vertex with 
	# minimum distance value, from the set of vertices 
	# not yet included in shortest path tree 
	def minDistance(self, dist, sptSet): 

		# Initilaize minimum distance for next node 
		min = sys.maxsize 

		# Search not nearest vertex not in the 
		# shortest path tree 
		for v in range(self.V): 
			if dist[v] < min and sptSet[v] == False: 
				min = dist[v] 
				min_index = v 

		return min_index 

	# Funtion that implements Dijkstra's single source 
	# shortest path algorithm for a graph represented 
	# using adjacency matrix representation 
	def dijkstra(self, src): 

            dist = [sys.maxsize] * self.V 
            dist[src] = 0
            sptSet = [False] * self.V 
            prev_index=[None]*self.V
            
            for cout in range(self.V): 

                    # Pick the minimum distance vertex from 
                    # the set of vertices not yet processed. 
                    # u is always equal to src in first iteration 
                    u = self.minDistance(dist, sptSet) 

                    # Put the minimum distance vertex in the 
                    # shotest path tree 
                    sptSet[u] = True

                    # Update dist value of the adjacent vertices 
                    # of the picked vertex only if the current 
                    # distance is greater than new distance and 
                    # the vertex in not in the shotest path tree 
                    for v in range(self.V): 
                            if self.graph[u][v] > 0 and sptSet[v] == False and dist[v] > dist[u] + self.graph[u][v]: 
                                    dist[v] = dist[u] + self.graph[u][v] 
                                    prev_index[v]=u
            
            
            return dist,prev_index

def get_adj_matrix(city_grid,alpha):# Obtain adjacency matrix for given city grid
    # Required to create graph

    var_mat=city_grid.variance_matrix
    sd_mat=np.sqrt(np.abs(var_mat))
    
    mean_mat=city_grid.mean_time_matrix
    
    UCB_mat=mean_mat+alpha*sd_mat
    num_nodes=int(np.sqrt(UCB_mat.shape[0]))
    adj_mat=np.zeros([UCB_mat.shape[0],UCB_mat.shape[0]])
    # Mapper from i to (x,y): (floor(i/num_nodes)=x,i%num_nodes=y)
    for i in range(adj_mat.shape[0]): # loop over all nodes on graph

        if i==0:
            adj_mat[i,i+1]=UCB_mat[i,1]
            adj_mat[i,i+num_nodes]=UCB_mat[i,2]
            continue
        if i==num_nodes-1:
            adj_mat[i,i-1]=UCB_mat[i,3]
            adj_mat[i,i+num_nodes]=UCB_mat[i,2]
            continue
        if i==(num_nodes-1)*num_nodes:
            adj_mat[i,i+1]=UCB_mat[i,1]
            adj_mat[i,i-num_nodes]=UCB_mat[i,0]
            continue
        if i==num_nodes**2-1:
            adj_mat[i,i-1]=UCB_mat[i,3]
            adj_mat[i,i-num_nodes]=UCB_mat[i,0]
            continue
        if floor(i/num_nodes)==0:

            adj_mat[i,i+1]=UCB_mat[i,1]
            adj_mat[i,i-1]=UCB_mat[i,3]
            adj_mat[i,i+num_nodes]=UCB_mat[i,2]
            continue
        if floor(i/num_nodes)==num_nodes-1:
            adj_mat[i,i+1]=UCB_mat[i,1]
            adj_mat[i,i-1]=UCB_mat[i,3]
            adj_mat[i,i-num_nodes]=UCB_mat[i,0]
            continue
        if i%num_nodes==0:
            adj_mat[i,i+1]=UCB_mat[i,1]
            adj_mat[i,i-num_nodes]=UCB_mat[i,0]
            adj_mat[i,i+num_nodes]=UCB_mat[i,2]
            continue
        if i%num_nodes==num_nodes-1:
            adj_mat[i,i-1]=UCB_mat[i,3]
            adj_mat[i,i-num_nodes]=UCB_mat[i,0]
            adj_mat[i,i+num_nodes]=UCB_mat[i,2]
            continue
        else:
            adj_mat[i,i+1]=UCB_mat[i,1]
            adj_mat[i,i-1]=UCB_mat[i,3]
            adj_mat[i,i-num_nodes]=UCB_mat[i,0]
            adj_mat[i,i+num_nodes]=UCB_mat[i,2]


    return adj_mat

def get_distance_dijkstra(start_pt,end_pt,g):

    dist, _ =g.dijkstra(start_pt)
    return dist[end_pt]


def get_path_dijkstra(start_pt,end_pt,g):

    _,path_index=g.dijkstra(start_pt)

    path=[end_pt]

    next_pt=end_pt
    
    while path_index[next_pt] != start_pt:
        
        path.append(path_index[next_pt])
        next_pt=path_index[next_pt]
    path.append(start_pt)
    return path

def solve_optim_ol(city_grid,var_noise,passenger_coords,cab_coords,walk_constraint,time_constraint,dest_pt):

    # parent function to solve optimization problem

    alpha=1 # hyper parameter to weight mean and variance
    adj_mat=get_adj_matrix(city_grid,alpha) # get adjacency matrix of the LCB values of travel time, for the grid
    num_nodes_dir=int(np.sqrt(city_grid.X_vec.shape[0]))
    total_nodes=city_grid.X_vec.shape[0]
    g=Graph(city_grid.X_vec.shape[0])
    g.graph=adj_mat
    
    cab_index=cab_coords[0]*num_nodes_dir+cab_coords[1]
    passenger_index=passenger_coords[0]*num_nodes_dir+passenger_coords[1]
    dest_index=dest_pt[0]*num_nodes_dir+dest_pt[1]
    #g.dijkstra(cab_index) # Gives distance from each node(or pickup-point) of cab
    
    # Find all points that are walkable to customer
    #----------------------
    walk_pts=[]
    dist,_=g.dijkstra(passenger_index) # change this later to eb more efficient
    for node in range(total_nodes):

        if dist[node]<=walk_constraint:

            walk_pts.append(node)

    #----------------------
    #Find the pt from walkable pts s.t total travel time is minimum
    best_pt=np.float('inf')
    best_time=np.float('inf')
    
    for pt in walk_pts:
        
        curr_time=get_distance_dijkstra(cab_index,pt,g)+get_distance_dijkstra(pt,dest_index,g)
        if curr_time < best_time:
            path_to_passenger=get_path_dijkstra(cab_index,pt,g)
            path_to_dest=get_path_dijkstra(pt,dest_index,g)
            best_time=curr_time
            best_pt=pt
    '''
    print(path_to_passenger)
    print(path_to_dest)
    print(passenger_index)
    print(best_pt)
    '''
    city_grid.visualize_path(path_to_passenger,path_to_dest,cab_index,passenger_index,dest_index,best_pt)

    if best_time < time_constraint:
        print("Trip Possible, Worst case trip Time:{:.2f}".format(best_time))
    #city_grid.visualize_path(path_to_passenger,path_to_dest)


    else:
        print("Trip impossible within set time parameters,best possible trip time is {:.2f}".format(best_time))
        
    return pt,path_to_passenger,path_to_dest,adj_matrix


