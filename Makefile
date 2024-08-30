CC = g++
CCOPTS = 

NVCC = nvcc
NVCCOPTS = -std=c++14

all: prserial prparallel

# SERIAL
prserial: serial/main.o serial/file_manager.o serial/push_relabel_serial_basic.o
	$(CC) $(CCOPTS) serial/main.o serial/file_manager.o serial/push_relabel_serial_basic.o -o serial/prserial

serial/main.o: serial/main.cpp
	$(CC) $(CCOPTS) -c serial/main.cpp -o serial/main.o

serial/file_manager.o: serial/src/file_manager.cpp
	$(CC) $(CCOPTS) -c serial/src/file_manager.cpp -o serial/file_manager.o

serial/push_relabel_serial_basic.o: serial/src/push_relabel_serial_basic.cpp
	$(CC) $(CCOPTS) -c serial/src/push_relabel_serial_basic.cpp -o serial/push_relabel_serial_basic.o

# PARALLEL
prparallel: parallel/main.o parallel/push_relabel_parallel.o parallel/utils.o parallel/file_manager.o
	$(NVCC) $(NVCCOPTS) parallel/main.o parallel/push_relabel_parallel.o parallel/utils.o parallel/file_manager.o -o parallel/prparallel

prparallel2: parallel/main2.o parallel/push_relabel_parallel2.o parallel/utils.o parallel/file_manager2.o
	$(NVCC) $(NVCCOPTS) parallel/main2.o parallel/push_relabel_parallel2.o parallel/utils.o parallel/file_manager2.o -o parallel/prparallel2

parallel/main.o: parallel/main.cu
	$(NVCC) $(NVCCOPTS) -c parallel/main.cu -o parallel/main.o

parallel/main2.o: parallel/main2.cu
	$(NVCC) $(NVCCOPTS) -c parallel/main2.cu -o parallel/main2.o

parallel/push_relabel_parallel.o: parallel/src/push_relabel_parallel.cu
	$(NVCC) $(NVCCOPTS) -c parallel/src/push_relabel_parallel.cu -o parallel/push_relabel_parallel.o

parallel/push_relabel_parallel2.o: parallel/src/push_relabel_parallel2.cu
	$(NVCC) $(NVCCOPTS) -c parallel/src/push_relabel_parallel2.cu -o parallel/push_relabel_parallel2.o

parallel/utils.o: parallel/src/utils.cpp
	$(CC) $(CCOPTS) -c parallel/src/utils.cpp -o parallel/utils.o

parallel/file_manager.o: parallel/src/file_manager.cpp
	$(CC) $(CCOPTS) -c parallel/src/file_manager.cpp -o parallel/file_manager.o

parallel/file_manager2.o: parallel/src/file_manager2.cu
	$(NVCC) $(NVCCOPTS) -c parallel/src/file_manager2.cu -o parallel/file_manager2.o

# TEST
n ?= 1

test: test.sh all
	chmod +x test.sh
	./test.sh

testserial: prserial serial/testserial.sh
	chmod +x ./serial/testserial.sh
	n=$(n); \
	while [ $${n} -gt 0 ] ; do \
		./serial/testserial.sh; \
		n=`expr $$n - 1`; \
	done; \
	true

testparallel: prparallel parallel/testparallel.sh
	chmod +x ./parallel/testparallel.sh
	n=$(n); \
	while [ $${n} -gt 0 ] ; do \
		./parallel/testparallel.sh; \
		n=`expr $$n - 1`; \
	done; \
	true

testparallel2: prparallel2 parallel/testparallel2.sh
	chmod +x ./parallel/testparallel2.sh
	n=$(n); \
	while [ $${n} -gt 0 ] ; do \
		./parallel/testparallel2.sh; \
		n=`expr $$n - 1`; \
	done; \
	true

# CLEAN
clean:
	find . -name "*.o" -type f -delete 
	find . -name "prserial" -type f -delete 
	find . -name "prparallel" -type f -delete
	find . -name "prparallel2" -type f -delete

cleanwin:
	del /s *.o *.exe *.exp *.lib 

cleanallresults:
	find ./results -name "*.json" -type f -delete

cleanserialresults:
	find ./results -name "*serial_*.json" -type f -delete

cleanparallelresults:
	find ./results -name "*parallel_*.json" -type f -delete

cleanparallel2results:
	find ./results -name "*parallel2_*.json" -type f -delete