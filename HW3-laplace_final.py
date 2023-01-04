from mpi4py import MPI
import numpy as np
from time import process_time
import matplotlib
import matplotlib.pyplot as plt

ROWS_Global , COLUMNS = 1000 , 1000
DOWN=100
UP=101

my_pe_num = MPI.COMM_WORLD.Get_rank()
num_of_PE = MPI.COMM_WORLD.Get_size()

ROWS=int(ROWS_Global/num_of_PE)

MAX_TEMP_ERROR = 0.01
temperature      = np.empty(( ROWS+2 , COLUMNS+2 ))
temperature_last = np.empty(( ROWS+2 ,COLUMNS+2  ))


def initialize_temperature(temp):

    temp[:,:] = 0
    tMin = (my_pe_num)*100.0/num_of_PE
    tMax = (my_pe_num+1)*100.0/num_of_PE

    #Set right side boundery condition
    for i in range(ROWS+1):
        temp[ i , 0 ] = 0
        temp[ i , COLUMNS+1 ] = tMin + ((tMax-tMin)/ROWS)*i

    #Set bottom boundery condition
    if my_pe_num == 0:
        for i in range(COLUMNS+1):
            temp[ 0 , i ] = 0

    if my_pe_num == num_of_PE -1:
        for i in range(COLUMNS+1):
            temp[ ROWS+1 , i ] = ( 100/COLUMNS ) * i

    return temp


def output(data):
    data.tofile("plate.out")
        
initialize_temperature(temperature_last)


max_iterations=10

if my_pe_num == 0:
    max_iterations = int(input("Maximum iterations: "))
    t1_start = process_time()
    
max_iterations=MPI.COMM_WORLD.bcast(max_iterations, root=0)


dt_global = 100
iteration = 1

while ( dt_global > MAX_TEMP_ERROR ) and ( iteration < max_iterations ):

    for i in range( 1 , ROWS+1 ):
        for j in range( 1 , COLUMNS+1 ):
            temperature[ i , j ] = 0.25 * ( temperature_last[i+1,j] + temperature_last[i-1,j] +
                                            temperature_last[i,j+1] + temperature_last[i,j-1]   )
    

    if (my_pe_num!=num_of_PE-1):
        MPI.COMM_WORLD.send(temperature[ROWS,:], dest=my_pe_num+1, tag=DOWN)
            
        
    if (my_pe_num!=0):
        temperature_last[0,:]=MPI.COMM_WORLD.recv(source=my_pe_num-1, tag=DOWN)
            

    if (my_pe_num!=0):
        MPI.COMM_WORLD.send(temperature[1,:], dest=my_pe_num-1, tag=UP)
        
    if (my_pe_num!=num_of_PE-1):
        temperature_last[ROWS+1,:]=MPI.COMM_WORLD.recv(source=my_pe_num+1, tag=UP)
        

    dt = 0.0

    for i in range( 1 , ROWS+1 ):
        for j in range( 1 , COLUMNS+1 ):
            dt = max( dt, temperature[i,j] - temperature_last[i,j])
            temperature_last[ i , j ] = temperature [ i , j ]

    dt_global=MPI.COMM_WORLD.reduce(dt, op=MPI.MAX, root=0)
    dt_global=MPI.COMM_WORLD.bcast(dt_global, root=0)

    iteration += 1
    print(iteration,flush=True)
    MPI.COMM_WORLD.barrier()


if (my_pe_num!= 0):
    MPI.COMM_WORLD.send(temperature_last, dest=0, tag=0)


if my_pe_num==0:
    t1_stop = process_time()
    t1_elapse = t1_stop-t1_start
    print("Total running time (in sec): \n", t1_elapse);
    print("Total iteration \n", iteration);
    
    temperature_last=np.array(temperature_last[0:ROWS+1,:])
    
    for i in range(1,num_of_PE -1):
        result=MPI.COMM_WORLD.recv(source=i, tag=0)
        temperature_last=np.concatenate((temperature_last,np.array(result)[1:ROWS+1,:]))

    result=MPI.COMM_WORLD.recv(source=num_of_PE-1, tag=0)
    
    temperature_last=np.concatenate((temperature_last,np.array(result)[1:ROWS+2,:]))

    output(temperature_last)

    plate = np.fromfile("plate.out", dtype=float).reshape((ROWS_Global+2,COLUMNS+2))
    plt.imshow(plate, norm=matplotlib.colors.LogNorm(0.1,50,clip=True))
    plt.show()