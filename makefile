all: make

make: main.cpp
	sudo mpic++ /home/mpiuser/mpi/integral/mpi.cpp -o /home/mpiuser/mpi/mpifinal/finale -std=c++14
.PHONY: clean

clean: 
	rm counter
