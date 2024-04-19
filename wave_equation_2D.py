import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
import imageio
import pygame
import random

animate_3d = 0 #Very very slow, see the GIF for the animation
use_initial_condition = 0
raindrops = 1

# Setting parameters
h = 1.5                   # Spatial step size
k = 1                   # Time step size
N_x_grid = 250         # Length of the solution area
N_y_grid = 250          # Width of the  solution area
N_t_gird = 500
cellsize = 2                # Cellsize for animation
raindrop_rate = 0.0110
print((k/h)**2)

def init_solution():
    sol = np.zeros((N_t_gird, N_x_grid, N_y_grid))  # solution for all the time steps
    return sol

def init_solver(initial_condition):
    u = np.zeros((3,N_x_grid,N_y_grid))             # u[0]: next time. u[1]: now time. u[2]: prev time

    if use_initial_condition:
        # Setting initial condition--------------------------
        # Do not use if using raindrops
        x, y = np.arange(1,N_x_grid,k), np.arange(1,N_y_grid,k)
        for i in range(N_x_grid-1):
            for j in range(N_y_grid-1):
                u[1,i,j] = initial_condition(x[i],y[j])
                u[2,i,j] = initial_condition(x[i],y[j])
    #--------------------------------------------------------

    c = 0.5                                         # A speed constant of some kind
    c_2 = 0
    alpha = np.zeros((N_x_grid,N_y_grid))           # Merging the speed constant (that does not have to me constant)
                                                    # and step sizes together in a matrix representing the grid
    # Setting nonconstant c ---------------------------------------------------
    for i in range(N_x_grid):
        for j in range(N_y_grid):
    # Squere at center
            # if (i > N_x_grid/2 -10 and i < N_x_grid/2 + 10 and j > N_y_grid/2 -10 and j < N_y_grid/2 + 10):
            #     alpha[i,j] = (k*c_2/h)**2
    # Gitter at middle line
            if (i == N_x_grid/4 and (j != N_y_grid/2 and j != (N_y_grid/2)+1 and j != (N_y_grid/2)-1) )\
                or (i == N_x_grid*3/4 and (j != N_y_grid/2 and j != (N_y_grid/2)+1 and j != (N_y_grid/2)-1)):
                alpha[i,j] = (k*c_2/h)**2
    # Two walls with collums
            # if (i == N_x_grid/4 and (j < N_y_grid*7/16 or j > N_y_grid*9/16) )\
            #     or (i == N_x_grid*3/4 and ((j < N_y_grid*7/16 or j > N_y_grid*9/16))):
            #     alpha[i,j] = (k*c_2/h)**2
            else:
                alpha[i,j] = (k*c/h)**2 
    #------------------------------------------------------------  
    # Constant c
    # alpha[0:N_x_grid,0:N_y_grid] = (k*c/h)**2       # Setting values for alpha
    return u, alpha

def initial_condition(x,y):
    # cone rising from zero
    a = 50            # magnitude of cone
    b = 10           # slope of cone
    if (np.sqrt((x-N_x_grid/2)**2+(y-N_y_grid/2)**2) < a/np.sqrt(b)):
        return -(a - np.sqrt(b*(x-N_x_grid/2)**2 + b*(y-N_y_grid/2)**2))
    else:
        return 0

def update(u, alpha):
    u[2] = u[1]
    u[1] = u[0]

    
    # This is my code, but it is very slow :(
    for i in range(1, N_x_grid-1):
        for j in range(1, N_y_grid-1):
            u[0,i,j] = alpha[i,j] * (u[1,i+1,j] + u[1,i-1,j] + u[1,i,j+1] + u[1,i,j-1] - 4*u[1,i,j])\
            + 2*u[1,i,j] - u[2,i,j]
    


    #Much faster code. But it is compleatly stolen :(
    
    # u[0, 1:N_x_grid-1, 1:N_y_grid-1]  = alpha[1:N_x_grid-1, 1:N_y_grid-1] * (u[1, 0:N_x_grid-2, 1:N_y_grid-1] + \
    #                                     u[1, 2:N_x_grid,   1:N_y_grid-1] + \
    #                                     u[1, 1:N_x_grid-1, 0:N_y_grid-2] + \
    #                                     u[1, 1:N_x_grid-1, 2:N_y_grid] - 4*u[1, 1:N_x_grid-1, 1:N_y_grid-1]) \
    #                                 + 2 * u[1, 1:N_x_grid-1, 1:N_y_grid-1] - u[2, 1:N_x_grid-1, 1:N_y_grid-1]
    
    u[0,1:N_x_grid-1,1:N_y_grid-1] *= 0.9995
    return u, alpha

def animate(sol):
    # Generate sample 3D matrix
    x, y = np.meshgrid(np.linspace(0, h*N_x_grid, N_x_grid), np.linspace(0, h*N_y_grid, N_y_grid))

    # Create a figure and 3D axis
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.axes.set_zlim3d(bottom=np.min(sol),top=np.max(sol))

    # Create initial surface plot
    surface = [ax.plot_surface(x, y, sol[0], cmap='viridis', rstride=1, cstride=1)]

    # Update function for animation
    def updateFrame(frame):
        ax.cla()  # Clear the current axis
        surface[0] = ax.plot_surface(x, y, sol[frame], cmap='viridis', rstride=1, cstride=1)
        ax.set_title(f'Time Step {frame}/{N_t_gird}')
        ax.axes.set_zlim3d(bottom=np.min(sol),top=np.max(sol))
        return surface

    # Create animation
    animation = FuncAnimation(fig, updateFrame, frames=N_t_gird, interval=100, blit=False) 

    plt.show()

def create_frame(t, sol):
    # Generate sample 3D matrix
    x, y = np.meshgrid(np.linspace(0, h*N_x_grid, N_x_grid), np.linspace(0, h*N_y_grid, N_y_grid))

    # Create a figure and 3D axis
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.axes.set_zlim3d(bottom=np.min(sol),top=np.max(sol))

    # Create surface plot
    surface = [ax.plot_surface(x, y, sol[t], cmap='viridis', rstride=1, cstride=1)]

    # Saving fig
    plt.savefig(f'./oblig_waveEquation/img/img_{t}.png', 
                transparent = False,  
                facecolor = 'white'
               )
    plt.close()

def makeGIF(sol):
    for t in range(N_t_gird):  # Only use this when making new GIF plot
        create_frame(t,sol)
    
    frames = []

    for t in range(N_t_gird):
        image = imageio.v2.imread(f'./oblig_waveEquation/img/img_{t}.png')
        frames.append(image)

    imageio.mimsave('./heat_eq_col.gif', 
                frames, 
                fps = 30, 
                loop = 1)

def place_raindrops(u, rate):
    if (random.random()<rate):                 # Making raindrops to appere random, not to often
        x = random.randrange(5, N_x_grid-5)
        y = random.randrange(5, N_y_grid-5)
        u[0, x-2:x+2, y-2:y+2] = 120            # Setting a pulse at a random point (a little area)

def main():

    if(animate_3d):
        u, alpha = init_solver(initial_condition)
        sol = init_solution()
        for i in range(N_t_gird):
            update(u,alpha)
            sol[i] = u[0]

        animate(sol)
        # makeGIF(sol)

    else:       # Stole this animaton, so do not understand it quite yet
        pygame.init()
        display = pygame.display.set_mode((300*cellsize, 300*cellsize))
        pygame.display.set_caption("Solving the 2d Wave Equation")

        u, alpha = init_solver(initial_condition)
        pixeldata = np.zeros((N_x_grid, N_y_grid, 3), dtype=np.uint8 )

        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return
            if (raindrops):
                place_raindrops(u, raindrop_rate)
            update(u, alpha)

            pixeldata[1:N_x_grid, 1:N_y_grid, 0] = np.clip(u[0, 1:N_x_grid, 1:N_y_grid] + 128, 0, 255)
            pixeldata[1:N_x_grid, 1:N_y_grid, 1] = np.clip(u[1, 1:N_x_grid, 1:N_y_grid] + 128, 0, 255)
            pixeldata[1:N_x_grid, 1:N_y_grid, 2] = np.clip(u[2, 1:N_x_grid, 1:N_y_grid] + 128, 0, 255)

            surf = pygame.surfarray.make_surface(pixeldata)
            display.blit(pygame.transform.scale(surf, (300 * cellsize, 300 * cellsize)), (0, 0))
            pygame.display.update()



main()


