import numpy as np
import math
from grid_functions import grid
from ol_optimizer import solve_optim_ol
if __name__=="__main__":
    np.random.seed(0)
    # Initial info req: 
    #1) Grid Information: described below
    #2) Location of passenger:(x,y)
    #3) location of cab:(x,y)
    #4) Destination point: (x,y)
    #5) Time constraint: a float
    #6) Walk constraint: an integer (numer of "blocks" the passenger can walk)
    
    #Grid details
    #--------------------
    num_pts=25
    disturb_centre=[11,10]
    time_centre=10
    variance_centre=4
    #----

    #Noise in grid values(specified by a variance) at every time step:
    var_noise=2
    #-------------------
    #Location of passenger
    passenger_coords=[9,9]

    #-------------------
    # Location of the cab
    cab_coords=[23,23]
    
    #-------------------
    #Constraints
    walk_constraint=20
    time_constraint=100

    #------------------
    #Destination point

    dest_pt=[1,1]

    rel_coords=[disturb_centre,passenger_coords,cab_coords,dest_pt]

    # Initialize grid with coordinates, distance from disturbance, mean jam time for each coordinate, variance in jam time for each coordinate
    city_grid=grid()
    city_grid.initialize_grid(num_pts,disturb_centre,time_centre,variance_centre,rel_coords)
    city_grid.visualize_traveltime() 
    # Get Solution:
    # Solve the optimization problem
    pt,path_to_passenger,path_to_dest,adj_matrix=solve_optim_ol(city_grid,var_noise,passenger_coords,cab_coords,walk_constraint,time_constraint,dest_pt)

    
    #----------------------------
    



