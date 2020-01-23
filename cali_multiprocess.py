import numpy as np
from scipy.interpolate import interp1d
from scipy.integrate import tplquad, dblquad, quad
import matplotlib.pyplot as plt
import time
import argparse
from mpi4py import MPI
import copy

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
nprocs = comm.Get_size()

parser = argparse.ArgumentParser(description='test running')
parser.add_argument('-lmax',default=200,type=int,
                    help='The maximum number of multipole, default=200')
parser.add_argument('-lmin',default=2,type=int,
                    help='The minimum number of multipole, default=2')
parser.add_argument('-alphas','--coherence scale',default=2,type=int,
                    help='coherence scale, from 2 to 8, means 2^2 to 2^8')
parser.add_argument('-As',default=1e-2,type=int,
                    help='rms of amplitude, default=1e-2')
args = parser.parse_args()

config = vars(args)

llist, Cl_EE = np.loadtxt(r'/home/wudl/one_freq/data/CMB_r/r_0.01/test_totCls.dat').T[[0, 2]]
# llist, Cl_EE = np.loadtxt(r'C:/Users/wudl/Desktop/recent work/test/r_0.1/test_totCls.dat').T[[0,2]]
Cl_EE = Cl_EE/(llist*(llist+1)/(2*np.pi))
func = interp1d(llist, Cl_EE, kind='linear',
                bounds_error=False, fill_value=(0, 0))

# plt.loglog(llist,Cl_EE)
# plt.show()

def fac_B_fun(alpha_s):
    return lambda l: l/(2*np.pi)*np.exp(-l*(l+1)*alpha_s**2)

def binary_split(nprocs):
    '''
    The range of l1 is (0,4*lmax)
    This function split the data (i.e., ell) to two parts at each time;
    which means , for example, if the range of ell is [2,100] namely l1
    is [0,4*100], therefore given nprocs = 5, the output will be:
    1st process: [0,200]
    2nd process: [200,300]
    3rd process: [300,350]
    4th process: [350, 375]
    5th process: [375,400] 

    output: return the range of l1 for each processes
    '''
    frac = 0
    end = []
    l1min = 0 
    l1max = 4*lmax
    for i in range(1, nprocs):
        frac += 1/np.power(2, i)
        end.append(frac)
    end = list(np.round(np.array(end)*l1max))
    end.append(l1max)
    end = [int(x) for x in end]
    start = copy.copy(end)
    start.insert(0, l1min)
    start.pop(-1)
    data = [(start[i], end[i]) for i in range(nprocs)]
    return data

def delta_Cl():
    sigma = 1/60*np.pi/180/np.sqrt(8*np.log(2))  # rad
    alpha_s = 2**fact_alpha/60*np.pi/180/np.sqrt(8*np.log(2))  # rad
    fac_B = A_s**2/quad(fac_B_fun(alpha_s), 0, np.inf)[0]
    if rank == 0:
        print('alpha_s:>>> ', alpha_s)
        print("fac_B:>>> ", fac_B)
    sum_B = 0
    for l1 in np.arange(data[0],data[1]):
        l = np.tile(np.arange(lmin, lmax), (4*l1+10, 1)).T
        phi1 = np.tile(2*np.pi/(4*l1+10)*np.arange(4*l1+10), (lmax-lmin, 1))
        l2 = np.sqrt(l**2+l1**2-2*l*l1*np.cos(phi1))
        l_zero = np.zeros_like(l2)
        temp = np.divide(l1, l2, out=l_zero, where=l2 != 0)
        w_a_square = 4*temp**2*np.sin(phi1)**2*(1-temp**2*np.sin(phi1)**2)
        Cl2_EE_sigma = func(l2)*np.exp(-l2*(l2+1)*sigma**2)
        k = l1/(2*np.pi)**2*fac_B*np.exp(-l1*(l1+1)*alpha_s**2) *\
            Cl2_EE_sigma*w_a_square*2*np.pi/(4*l1+10)
        delta_Cl_BB = np.sum(k, axis=1)
        sum_B = sum_B + delta_Cl_BB
        if l1 in np.arange(0, 4*lmax, round((lmax-lmin)/20)):
            print(rank,l1)
    return sum_B

lmin = config['lmin']
lmax = config['lmax']
A_s = config['As']
fact_alpha = config['coherence scale']  # from 2^2 to 2^8
if rank == 0:
    print('\n','*'*10)
    print('running...')
    print("Time start at: %s" % time.ctime())
    start_t = time.time()    
    data = binary_split(nprocs)
    print('number of processes: ',nprocs)
    print('process interval: ',data)
else:
    data = None
data = comm.scatter(data, root=0)
sum_B = delta_Cl()
sum_B = comm.gather(sum_B,root=0)
if rank == 0:
    ell_B = np.arange(lmin, lmax)
    sum_B = np.array(np.sum(sum_B,axis=0))
    delta_B = (ell_B*(ell_B+1)/(2*np.pi)*sum_B)**(1/2)
    np.savetxt('../data/calib_'+str(lmax)+'_'+str(2**fact_alpha)+'.txt', delta_B)
    end_t = time.time()
    print('Time costs:>>> %.4f mins' % ((end_t-start_t)/60))
    print("Time end at: %s" % time.ctime())
    plt.loglog(np.arange(len(delta_B)), delta_B, label='Calibration')
    plt.legend()
    plt.show()
