CC = g++
NVCC = nvcc


all: pr 

pr: main.o file_manager.o push_relabel_serial_basic.o
	$(NVCC) main.o file_manager.o push_relabel_serial_basic.o -o pr 

main.o: main.cpp file_manager.o push_relabel_serial_basic.o
	$(NVCC) -c main.cpp -o main.o

file_manager.o: src/file_manager.cpp
	$(NVCC) -c src/file_manager.cpp -o file_manager.o

push_relabel_serial_basic.o: src/push_relabel_serial_basic.cu
	$(NVCC) -c src/push_relabel_serial_basic.cu -o push_relabel_serial_basic.o

clean:
	rm -f *.o pr