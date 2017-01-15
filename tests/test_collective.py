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
Tests Collectives GPU-GPU Communications.
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

    def test_allreduce_ops(self):
        size = self.comm.Get_size()
        self.comm.Allreduce(self.cpu_send, self.cpu_recv, MPI.PROD)
        self.gpu_comm.Allreduce(self.gpu_send, self.gpu_recv, MPI.PROD)
        self.assertTrue(np.allclose(self.gpu_recv.get(), self.cpu_recv),
                        'GPU-CPU Recv mismatch')
        self.comm.Allreduce(self.cpu_send, self.cpu_recv, MPI.MAX)
        self.gpu_comm.Allreduce(self.gpu_send, self.gpu_recv, MPI.MAX)
        self.assertTrue(np.allclose(self.gpu_recv.get(), self.cpu_recv),
                        'GPU-CPU Recv mismatch')
        self.comm.Allreduce(self.cpu_send, self.cpu_recv, MPI.MIN)
        self.gpu_comm.Allreduce(self.gpu_send, self.gpu_recv, MPI.MIN)
        self.assertTrue(np.allclose(self.gpu_recv.get(), self.cpu_recv),
                        'GPU-CPU Recv mismatch')
        self.comm.Allreduce(self.cpu_send, self.cpu_recv, MPI.SUM)
        self.gpu_comm.Allreduce(self.gpu_send, self.gpu_recv, MPI.SUM)
        self.assertTrue(np.allclose(self.gpu_recv.get(), self.cpu_recv),
                        'GPU-CPU Recv mismatch')

    def test_allreduce(self):
        size = self.comm.Get_size()
        self.comm.Allreduce(self.cpu_send, self.cpu_recv, MPI.SUM)
        self.gpu_comm.Allreduce(self.gpu_send, self.gpu_recv, MPI.SUM)
        self.assertTrue(np.allclose(self.cpu_send*size, self.cpu_recv),
                        'CPU-CPU Send-Recv mismatch')
        self.assertTrue(np.allclose(self.gpu_recv.get(), self.cpu_recv),
                        'GPU-CPU Recv mismatch')

    def test_ring_allreduce(self):
        size = self.comm.Get_size()
        self.comm.Allreduce(self.cpu_send, self.cpu_recv, MPI.SUM)
        self.gpu_comm.ring_allreduce(self.gpu_send, self.gpu_recv, MPI.SUM)
        self.assertTrue(np.allclose(self.cpu_send*size, self.cpu_recv),
                        'CPU-CPU Send-Recv mismatch')
        self.assertTrue(np.allclose(self.gpu_recv.get(), self.cpu_recv),
                        'GPU-CPU Recv mismatch')

    def test_ring_bcast(self):
        self.comm.Bcast(self.gpu_send, root=1)
        self.comm.Bcast(self.cpu_send, root=1)
        self.assertTrue(np.allclose(self.gpu_recv.get(), self.cpu_recv),
                        'GPU-CPU Recv mismatch')

if __name__ == '__main__':
    unittest.main()
