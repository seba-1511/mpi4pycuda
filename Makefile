
.PHONY: all allreduce

all: bcast

allreduce:
	mpirun -n 2 python examples/allreduce.py

bcast:
	mpirun -n 4 python examples/bcast.py

test: p2p collectives

p2p:
	mpirun -n 2 python tests/test_point_to_point.py

collectives:
	mpirun -n 4 python tests/test_collective.py
