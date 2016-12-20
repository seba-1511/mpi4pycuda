import numpy as np
import pycuda.gpuarray as gpu
import pycuda.driver as cuda
import pycuda.autoinit

from mpi4py import MPI
from mpi4pycuda import GPUComm
from time import sleep

if __name__ == '__main__':
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Set different GPU devices
    dev = cuda.Device(rank % 4)
    ctx = dev.make_context()
    ctx.push()
    print 'Num devices: ', dev.count()

    # CPU Allreduce
    send_buff = np.array([[1, 2], [5, 6]])
    recv_buff = np.zeros_like(send_buff)
    comm.Bcast(send_buff, root=0)

    # GPU Allreduce
    send_gpu = gpu.to_gpu(send_buff.astype(np.float32))
    recv_gpu = gpu.zeros_like(send_gpu)
    gpu_comm = GPUComm(comm)
    gpu_comm.Bcast(send_gpu, root=0)

    sleep(1000)

    if rank == 0:
        assert np.allclose(recv_gpu.get(), recv_buff)
        print 'GPU Allreduce working'
