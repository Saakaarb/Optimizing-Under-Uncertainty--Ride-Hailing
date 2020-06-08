import numpy as np
import math
from matplotlib import pyplot as plt
from math import floor
class grid:
    

    def initialize_grid(self,num_pts,disturb_centre,time_centre,variance_centre,rel_coords,variance_random=1):
        """
        Creates grid at the time of requesting cab.
        
        Inputs:
        num_pts: number of sampled points on grid: each sample correponds to a "road"
        disturb_centre: Corresponds to the grid point where the disturbance is concentrated
        mean_centre: Mean time taken at centre of disturbance. This mean decays as we move away from centre of disturbance
        variance_centre: variance of time taken at centre of disturbance: also decays

        Defines:
        X: x coordinates of grid points
        Y: y coordinates of grid points
        d_matrix: distance from centre of disturbance
        mean_time_matrix: mean values of road traversal time for a node
        variance_matrix: variance of the above, will be sampled from a normal distribution
        These are all vectors sampled

        """
        self.num_pts=num_pts
        self.rel_coords=rel_coords
        l=2
        self.x_line=np.linspace(0,num_pts,num_pts) # 0,4 can be changed
        self.y_line=np.linspace(0,num_pts,num_pts)
        [self.X,self.Y]=np.meshgrid(self.x_line,self.y_line)
        #Plot initial grid
        '''
        x_plot=np.reshape(self.X,[self.x_line.shape[0]*self.y_line.shape[0],1])
        y_plot=np.reshape(self.Y,[self.x_line.shape[0]*self.y_line.shape[0],1])
        plt.plot(x_plot,y_plot,'x')
        plt.show()
        '''
        self.X_vec=np.reshape(self.X,[self.X.shape[0]*self.X.shape[1],1])
        self.Y_vec=np.reshape(self.Y,[self.Y.shape[0]*self.Y.shape[1],1])
        # Calculate distance of every sample point from centre of disturbance

        self.d_matrix=np.zeros(self.X_vec.shape) # distance from centre of disturbance

        disturb_coords=np.array(disturb_centre) 
        self.mean_time_matrix=np.zeros([self.X_vec.shape[0],4]) # Store mean times for every road
        self.variance_matrix=np.zeros([self.X_vec.shape[0],4]) # Store fluctuations in time for every road
        
        for i_x in range(self.X_vec.shape[0]):

               
                d_from_centre=np.linalg.norm(disturb_coords-np.array([self.X_vec[i_x],self.Y_vec[i_x]]),ord=2) # distance from centre of disturbance
                self.d_matrix[i_x]=d_from_centre
                
                for i_y in range(self.mean_time_matrix.shape[1]):

                    if self.X_vec[i_x]==disturb_coords[0] and self.Y_vec[i_x]==disturb_coords[1]:
                        self.mean_time_matrix[i_x,i_y]=time_centre
                        self.variance_matrix[i_x,i_y]=variance_centre
                        continue
                    self.mean_time_matrix[i_x,i_y]=abs(time_centre*np.exp(-d_from_centre/(2*l**2))+np.random.normal(0,variance_random))
                    self.variance_matrix[i_x,i_y]=abs(variance_centre*np.exp(-d_from_centre/(2*l**2))+np.random.normal(0,variance_random/2))

                    #self.mean_time_matrix[i_x,i_y]=abs(time_centre/d_from_centre+np.random.normal(0,variance_random)) # Assumption of inverse scaling with distance
                    #self.variance_matrix[i_x,i_y]=abs(variance_centre/d_from_centre+np.random.normal(0,variance_random/2)) # Assumption of inverse scaling with distance
#-------------------------------------------
    def visualize_traveltime(self):

        current_grid=self.sample_grid()
        x_plot=np.reshape(self.X,[self.x_line.shape[0]*self.y_line.shape[0],1])
        y_plot=np.reshape(self.Y,[self.x_line.shape[0]*self.y_line.shape[0],1])
        plt.plot(x_plot,y_plot,'x',markersize=1)
        plt.contourf(self.X,self.Y,current_grid,cmap='RdGy')
        plt.colorbar()
        x_rel=[]
        y_rel=[]
        for coord in self.rel_coords:

            x_rel.append(coord[0])
            y_rel.append(coord[1])

        plt.scatter(x_rel,y_rel)
        labels=["Disturbance","Passenger Start","Cab Start","Destination"]
        for i, txt in enumerate(labels):
            plt.annotate(txt, (x_rel[i], y_rel[i]))
        plt.xlim([-1,26])
        plt.ylim([-1,26])
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.show()

    
    def visualize_path(self,path_to_passenger,path_to_dest,cab_index,passenger_index,dest_index,pickup_index):

        current_grid=self.sample_grid()
        x_plot=np.reshape(self.X,[self.x_line.shape[0]*self.y_line.shape[0],1])
        y_plot=np.reshape(self.Y,[self.x_line.shape[0]*self.y_line.shape[0],1])
        plt.plot(x_plot,y_plot,'x',markersize=1)
        plt.contourf(self.X,self.Y,current_grid,cmap='RdGy')
        plt.colorbar()
        x_rel=[]
        y_rel=[]
        x_pickup=self.X[floor(pickup_index/self.num_pts),pickup_index%self.num_pts]
        y_pickup=self.Y[floor(pickup_index/self.num_pts),pickup_index%self.num_pts]
        for coord in self.rel_coords:
            x_rel.append(self.X[coord[0],coord[1]])
            y_rel.append(self.Y[coord[0],coord[1]])
        
        x_rel.append(x_pickup)
        y_rel.append(y_pickup)
        plt.scatter(x_rel,y_rel)
        labels=["Disturbance","Start","Cab Start","Destination","Pickup Point"]
        for i, txt in enumerate(labels):
            if i==len(labels)-1:
                plt.annotate(txt, (x_rel[i], y_rel[i]+1))
            else:
                plt.annotate(txt, (x_rel[i], y_rel[i]))
        # Mapper from i to (x,y): (floor(i/num_nodes)=x,i%num_nodes=y)
        #Plot path from cab to passenger
        prev_pt_x=floor(path_to_passenger[0]/self.num_pts)
        prev_pt_y=path_to_passenger[0]%self.num_pts
        for i in range(1,len(path_to_passenger)):
            pt=path_to_passenger[i]
            x_ind=floor(pt/self.num_pts)
            y_ind=pt%self.num_pts
            x_prev=self.X[prev_pt_x,prev_pt_y]
            y_prev=self.Y[prev_pt_x,prev_pt_y]
            x_next=self.X[x_ind,y_ind]
            y_next=self.Y[x_ind,y_ind]
            plt.plot([x_prev,x_next],[y_prev,y_next],color="black")
            prev_pt_x=x_ind
            prev_pt_y=y_ind
            #x_c_p.append(self.X[x_ind,y_ind])
            #y_c_p.append(self.Y[x_ind,y_ind])
            
        #Plot path from passenger to destination
        prev_pt_x=floor(dest_index/self.num_pts)
        prev_pt_y=dest_index%self.num_pts
        for i in range(len(path_to_dest)):
            pt=path_to_dest[i]
            x_ind=floor(pt/self.num_pts)
            y_ind=pt%self.num_pts

            x_prev=self.X[prev_pt_x,prev_pt_y]
            y_prev=self.Y[prev_pt_x,prev_pt_y]
            x_next=self.X[x_ind,y_ind]
            y_next=self.Y[x_ind,y_ind]
            plt.plot([x_prev,x_next],[y_prev,y_next],color="blue")

            prev_pt_x=x_ind
            prev_pt_y=y_ind
            #x_c_p.append(self.X[x_ind,y_ind])
            #y_c_p.append(self.Y[x_ind,y_ind])

        plt.xlim([-1,26])
        plt.ylim([-1,26])
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.show()

    def sample_grid(self):
        # sample a grid according to the stochastic parameters
        np.random.seed(0)
        current_grid=np.zeros(self.X.shape)
        for i_x in range(self.X.shape[0]):

            for i_y in range(self.X.shape[0]):
                current_grid[i_x,i_y]=abs(np.random.normal(self.mean_time_matrix[i_x*self.num_pts+i_y,0],self.variance_matrix[i_x*self.num_pts+i_y,0]))
                
        return current_grid

#--------------------------------------------
    def update_grid(self,variance):

        # Update the mean times of a grid according to a noise metric
        for i_x in range(self.X.shape[0]):

            for i_y in range(self.X.shape[0]):

                self.mean_time_matrix[i_x,i_y]= self.mean_time_matrix[i_x,i_y]+np.random.normal(0,variance)

        # Update the variance of the grid according to some noise metric

        for i_x in range(self.X.shape[0]):

            for i_y in range(self.X.shape[0]):

                self.variance_matrix[i_x,i_y]= self.variance_matrix[i_x,i_y]+np.random.normal(0,variance/2)
        return 



