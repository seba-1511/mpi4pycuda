#!/usr/bin/env python

import unittest
import numpy as np
import pycuda.gpuarray as gpu
import pycuda.driver as cuda
import pycuda.autoinit

from mpi4py import MPI
from mpi4pycuda import GPUComm

np.random.seed(1234)

TEST_DIMS = (4, 4)

"""
Tests Point-to-Point GPU-GPU Communications.

Expects to be run with exactly 2 MPI Processes.
"""


class TestPoint2Point(unittest.TestCase):

    def setUp(self):
        self.comm = MPI.COMM_WORLD
        self.gpu_comm = GPUComm(MPI.COMM_WORLD)
        self.cpu_send = np.random.rand(*TEST_DIMS).astype(np.float32)
        self.gpu_send = gpu.to_gpu(self.cpu_send)
        self.cpu_recv = np.zeros_like(self.cpu_send)
        self.gpu_recv = gpu.zeros_like(self.gpu_send)

    def tearDown(self):
        pass

    def test_send_recv(self):
        rank = self.comm.Get_rank()
        dest = 1 - rank
        if rank == 0:
            self.comm.Recv(self.cpu_recv, source=dest)
            self.gpu_comm.Recv(self.gpu_recv, source=dest)
            self.comm.Send(self.cpu_send, dest=dest) 
            self.gpu_comm.Send(self.gpu_send, dest=dest)
        else: 
            self.comm.Send(self.cpu_send, dest=dest) 
            self.gpu_comm.Send(self.gpu_send, dest=dest)
            self.comm.Recv(self.cpu_recv, source=dest)
            self.gpu_comm.Recv(self.gpu_recv, source=dest)

        self.assertTrue(np.allclose(self.cpu_send, self.cpu_recv),
                        'CPU-CPU Send-Recv mismatch')
        self.assertTrue(np.allclose(self.gpu_recv.get(), self.cpu_recv),
                        'GPU-CPU Recv mismatch')


if __name__ == '__main__':
    unittest.main()
