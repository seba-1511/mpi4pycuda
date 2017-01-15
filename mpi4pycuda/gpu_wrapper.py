#!/usr/bin/env python

import numpy as np
import pycuda.gpuarray as gpu
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.autoinit import context
from math import log, ceil, pow, floor

from mpi4py import MPI

""" 
Wrapper for GPU-GPU Communication.

TODO:
    * Complete the list of Ops. (Custom def and in .get_op())
"""

def logg(*msg):
    print(MPI.COMM_WORLD.Get_rank(), ': ', msg)

# CUSTOM Op Definitions
def SUM(inp, inout):
    inout += inp

def MAX(inp, inout):
    gpu.maximum(inp, inout, out=inout)

def MIN(inp, inout):
    gpu.minimum(inp, inout, out=inout)

def PROD(inp, inout):
    inout *= inp


class GPUComm(object):

    def __init__(self, comm):
        self.comm = comm
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()

        # Temporary buffers
        self.temporaries = {}

    # Helpers and Buffer Management
    @staticmethod
    def buff(ary):
        return ary.gpudata.as_buffer(ary.nbytes)

    @staticmethod
    def get_op(op):
        if op is None:
            return SUM
        if op.py2f() == 3:
            return SUM
        if op.py2f() == 1:
            return MAX
        if op.py2f() == 2:
            return MIN
        if op.py2f() == 4:
            return PROD

    def temp(self, shape):
        if shape in self.temporaries:
            temp = self.temporaries[shape]
        else:
            temp = gpu.zeros(shape, np.float32)
            self.temporaries[shape] = temp
        return temp

    # Point-to-Point Communication
    def Send(self, buff, dest):
        self.comm.Send(self.buff(buff), dest=dest)

    def Recv(self, buff, source=None):
        self.comm.Recv(self.buff(buff), source=source)

    # Allreduce Implementations
    def Allreduce(self, send, recv, op=None):
        self.ring_allreduce(send, recv, op)

    def ring_allreduce(self, send, recv, op=None):
        op = self.get_op(op)
        send_buff = self.buff(send)
        recv_buff = self.buff(recv)
        accum = self.temp(send.shape)
        accum[:] = send[:]
        context.synchronize()

        left = ((self.rank - 1) + self.size) % self.size
        right = (self.rank + 1) % self.size

        for i in range(self.size - 1):
            if i % 2 == 0:
                # Send send_buff
                send_req = self.comm.Isend(send_buff, dest=right)
                self.comm.Recv(recv_buff, source=left)
                # accum[:] += recv[:]
                op(recv, accum)
            else:
                # Send recv_buff
                send_req = self.comm.Isend(recv_buff, dest=right)
                self.comm.Recv(send_buff, source=left)
                # accum[:] += send[:]
                op(send, accum)
            send_req.Wait()
        context.synchronize()
        recv[:] = accum[:]

    # Broadcast Implementations
    def Bcast(self, buf, root=0):
        self.ring_bcast(buf, root)

    def ring_bcast(self, buf, root=0):
        root_rank = (self.size + self.rank - root) % self.size
        gpu_buff = self.buff(buf)
        num_levels = int(log(self.size, 2))
        my_level = 0
        for i in range(0, num_levels+1):
            if root_rank % (self.size/pow(2, i)) == 0:
                my_level = i
                break
        if root_rank != 0:
            self.comm.Recv(gpu_buff)

        for i in range(my_level+1, num_levels+1):
            dest = (root_rank + (self.size // pow(2, i))) % self.size
            logg(root_rank, 'sending to', dest)
            self.comm.Send(gpu_buff, dest=dest)
        logg('finished')


    def mesh_bcast(self, buf, root=0):
        pass

    def hypercube_bcast(self, buf, root=0):
        pass

    # Reduce Implementations
    def Reduce(self, sendbuf, recvbuf, op=None, root=0):
        self.ring_reduce(sendbuf, recvbuf, op, root)

    def ring_reduce(self, sendbuf, recvbuf, op=None, root=0):
        op = self.get_op(op)

    def mesh_reduce(self, sendbuf, recvbuf, op=None, root=0):
        pass

    def hypercube_reduce(self, sendbuf, recvbuf, op=None, root=0):
        pass

