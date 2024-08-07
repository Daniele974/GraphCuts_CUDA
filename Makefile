CC = g++
CCOPTS = 

NVCC = nvcc
NVCCOPTS = -std=c++14

all: prserial

prserial: serial/main.o serial/file_manager.o serial/push_relabel_serial_basic.o
	$(CC) $(CCOPTS) serial/main.o serial/file_manager.o serial/push_relabel_serial_basic.o -o serial/prserial

serial/main.o: serial/main.cpp
	$(CC) $(CCOPTS) -c serial/main.cpp -o serial/main.o

serial/file_manager.o: serial/src/file_manager.cpp
	$(CC) $(CCOPTS) -c serial/src/file_manager.cpp -o serial/file_manager.o

serial/push_relabel_serial_basic.o: serial/src/push_relabel_serial_basic.cpp
	$(CC) $(CCOPTS) -c serial/src/push_relabel_serial_basic.cpp -o serial/push_relabel_serial_basic.o

clean:
	find . -name "*.o" -type f -delete 
	find . -name "prserial" -type f -delete 

cleanwin:
	del /s *.o *.exe *.exp *.lib 

testserial: prserial serial/testserial.sh
	./serial/testserial.sh