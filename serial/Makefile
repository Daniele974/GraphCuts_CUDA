CC = g++
CCOPTS = 

NVCC = nvcc
NVCCOPTS = -std=c++14

all: pr 

pr: main.o file_manager.o push_relabel_serial_basic.o
	$(CC) $(CCOPTS) main.o file_manager.o push_relabel_serial_basic.o -o pr 

main.o: main.cpp
	$(CC) $(CCOPTS) -c main.cpp -o main.o

file_manager.o: src/file_manager.cpp
	$(CC) $(CCOPTS) -c src/file_manager.cpp -o file_manager.o

push_relabel_serial_basic.o: src/push_relabel_serial_basic.cpp
	$(CC) $(CCOPTS) -c src/push_relabel_serial_basic.cpp -o push_relabel_serial_basic.o

clean:
	rm -f *.o pr

cleanwin:
	del *.o *.exe *.exp *.lib 