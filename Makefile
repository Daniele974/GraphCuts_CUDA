CC = g++
CCOPTS = 

NVCC = nvcc
NVCCOPTS = -std=c++14 -Xcompiler -fopenmp -lgomp

all: prserial prparallel prparallelbcsr

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

parallel/main.o: parallel/main.cu
	$(NVCC) $(NVCCOPTS) -c parallel/main.cu -o parallel/main.o

parallel/push_relabel_parallel.o: parallel/src/push_relabel_parallel.cu
	$(NVCC) $(NVCCOPTS) -c parallel/src/push_relabel_parallel.cu -o parallel/push_relabel_parallel.o

parallel/utils.o: parallel/src/utils.cpp
	$(CC) $(CCOPTS) -c parallel/src/utils.cpp -o parallel/utils.o

parallel/file_manager.o: parallel/src/file_manager.cpp
	$(CC) $(CCOPTS) -c parallel/src/file_manager.cpp -o parallel/file_manager.o

# PARALLEL BCSR TC
prparallelbcsrtc: parallel_bcsr_tc/main.o parallel_bcsr_tc/push_relabel_parallel.o parallel_bcsr_tc/utils.o parallel_bcsr_tc/file_manager.o
	$(NVCC) $(NVCCOPTS) parallel_bcsr_tc/main.o parallel_bcsr_tc/push_relabel_parallel.o parallel_bcsr_tc/utils.o parallel_bcsr_tc/file_manager.o -o parallel_bcsr_tc/prparallelbcsrtc

parallel_bcsr_tc/main.o: parallel_bcsr_tc/main.cu
	$(NVCC) $(NVCCOPTS) -c parallel_bcsr_tc/main.cu -o parallel_bcsr_tc/main.o

parallel_bcsr_tc/push_relabel_parallel.o: parallel_bcsr_tc/src/push_relabel_parallel.cu
	$(NVCC) $(NVCCOPTS) -c parallel_bcsr_tc/src/push_relabel_parallel.cu -o parallel_bcsr_tc/push_relabel_parallel.o

parallel_bcsr_tc/utils.o: parallel_bcsr_tc/src/utils.cpp
	$(CC) $(CCOPTS) -c parallel_bcsr_tc/src/utils.cpp -o parallel_bcsr_tc/utils.o

parallel_bcsr_tc/file_manager.o: parallel_bcsr_tc/src/file_manager.cpp
	$(CC) $(CCOPTS) -c parallel_bcsr_tc/src/file_manager.cpp -o parallel_bcsr_tc/file_manager.o

# PARALLEL BCSR VC
prparallelbcsrvc: parallel_bcsr_vc/main.o parallel_bcsr_vc/push_relabel_parallel.o parallel_bcsr_vc/utils.o parallel_bcsr_vc/file_manager.o
	$(NVCC) $(NVCCOPTS) parallel_bcsr_vc/main.o parallel_bcsr_vc/push_relabel_parallel.o parallel_bcsr_vc/utils.o parallel_bcsr_vc/file_manager.o -o parallel_bcsr_vc/prparallelbcsrvc

parallel_bcsr_vc/main.o: parallel_bcsr_vc/main.cu
	$(NVCC) $(NVCCOPTS) -c parallel_bcsr_vc/main.cu -o parallel_bcsr_vc/main.o

parallel_bcsr_vc/push_relabel_parallel.o: parallel_bcsr_vc/src/push_relabel_parallel.cu
	$(NVCC) $(NVCCOPTS) -c parallel_bcsr_vc/src/push_relabel_parallel.cu -o parallel_bcsr_vc/push_relabel_parallel.o

parallel_bcsr_vc/utils.o: parallel_bcsr_vc/src/utils.cpp
	$(CC) $(CCOPTS) -c parallel_bcsr_vc/src/utils.cpp -o parallel_bcsr_vc/utils.o

parallel_bcsr_vc/file_manager.o: parallel_bcsr_vc/src/file_manager.cpp
	$(CC) $(CCOPTS) -c parallel_bcsr_vc/src/file_manager.cpp -o parallel_bcsr_vc/file_manager.o

# PARALLEL OLD
prparallelOld: parallel_old/main.o parallel_old/push_relabel_parallel.o parallel_old/utils.o parallel_old/file_manager.o
	$(NVCC) $(NVCCOPTS) parallel_old/main.o parallel_old/push_relabel_parallel.o parallel_old/utils.o parallel_old/file_manager.o -o parallel_old/prparallel_old

prparallel2: parallel_old/main2.o parallel_old/push_relabel_parallel2.o parallel_old/utils.o parallel_old/file_manager2.o
	$(NVCC) $(NVCCOPTS) parallel_old/main2.o parallel_old/push_relabel_parallel2.o parallel_old/utils.o parallel_old/file_manager2.o -o parallel_old/prparallel2

parallel_old/main.o: parallel_old/main.cu
	$(NVCC) $(NVCCOPTS) -c parallel_old/main.cu -o parallel_old/main.o

parallel_old/main2.o: parallel_old/main2.cu
	$(NVCC) $(NVCCOPTS) -c parallel_old/main2.cu -o parallel_old/main2.o

parallel_old/push_relabel_parallel.o: parallel_old/src/push_relabel_parallel.cu
	$(NVCC) $(NVCCOPTS) -c parallel_old/src/push_relabel_parallel.cu -o parallel_old/push_relabel_parallel.o

parallel_old/push_relabel_parallel2.o: parallel_old/src/push_relabel_parallel2.cu
	$(NVCC) $(NVCCOPTS) -c parallel_old/src/push_relabel_parallel2.cu -o parallel_old/push_relabel_parallel2.o

parallel_old/file_manager.o: parallel_old/src/file_manager.cpp
	$(CC) $(CCOPTS) -c parallel_old/src/file_manager.cpp -o parallel_old/file_manager.o

parallel_old/file_manager2.o: parallel_old/src/file_manager2.cu
	$(NVCC) $(NVCCOPTS) -c parallel_old/src/file_manager2.cu -o parallel_old/file_manager2.o

parallel_old/utils.o: parallel_old/src/utils.cpp
	$(CC) $(CCOPTS) -c parallel_old/src/utils.cpp -o parallel_old/utils.o


# TEST
n ?= 1

test: 
	make testserial
	make testparallel
	make testparallelbcsrvc

testserial: prserial serial/testserial.sh
	if [ ! -d "./results" ]; then mkdir results; fi
	chmod +x ./serial/testserial.sh
	n=$(n); \
	while [ $${n} -gt 0 ] ; do \
		echo "Cicli rimanenti: $$n"; \
		./serial/testserial.sh; \
		n=`expr $$n - 1`; \
	done; \
	true

testparallel: prparallel parallel/testparallel.sh
	if [ ! -d "./results" ]; then mkdir results; fi
	chmod +x ./parallel/testparallel.sh
	n=$(n); \
	while [ $${n} -gt 0 ] ; do \
		echo "Cicli rimanenti: $$n"; \
		./parallel/testparallel.sh; \
		n=`expr $$n - 1`; \
	done; \
	true

testparallelbcsrtc: prparallelbcsrtc parallel_bcsr_tc/testparallelbcsrtc.sh
	if [ ! -d "./results" ]; then mkdir results; fi
	chmod +x ./parallel_bcsr_tc/testparallelbcsrtc.sh
	n=$(n); \
	while [ $${n} -gt 0 ] ; do \
		echo "Cicli rimanenti: $$n"; \
		./parallel_bcsr_tc/testparallelbcsrtc.sh; \
		n=`expr $$n - 1`; \
	done; \
	true

testparallelbcsrvc: prparallelbcsrvc parallel_bcsr_vc/testparallelbcsrvc.sh
	if [ ! -d "./results" ]; then mkdir results; fi
	chmod +x ./parallel_bcsr_vc/testparallelbcsrvc.sh
	n=$(n); \
	while [ $${n} -gt 0 ] ; do \
		echo "Cicli rimanenti: $$n"; \
		./parallel_bcsr_vc/testparallelbcsrvc.sh; \
		n=`expr $$n - 1`; \
	done; \
	true

testdensityparallel: prparallel parallel/testparalleldensity.sh
	if [ ! -d "./results" ]; then mkdir results; fi
	chmod +x ./parallel/testparalleldensity.sh
	n=$(n); \
	while [ $${n} -gt 0 ] ; do \
		echo "Cicli rimanenti: $$n"; \
		./parallel/testparalleldensity.sh; \
		n=`expr $$n - 1`; \
	done; \
	true

testdensityparallelbcsrtc: prparallelbcsrtc parallel_bcsr_tc/testparallelbcsrtcdensity.sh
	if [ ! -d "./results" ]; then mkdir results; fi
	chmod +x ./parallel_bcsr_tc/testparallelbcsrtcdensity.sh
	n=$(n); \
	while [ $${n} -gt 0 ] ; do \
		echo "Cicli rimanenti: $$n"; \
		./parallel_bcsr_tc/testparallelbcsrtcdensity.sh; \
		n=`expr $$n - 1`; \
	done; \
	true

testdensityparallelbcsrvc: prparallelbcsrvc parallel_bcsr_vc/testparallelbcsrvcdensity.sh
	if [ ! -d "./results" ]; then mkdir results; fi
	chmod +x ./parallel_bcsr_vc/testparallelbcsrvcdensity.sh
	n=$(n); \
	while [ $${n} -gt 0 ] ; do \
		echo "Cicli rimanenti: $$n"; \
		./parallel_bcsr_vc/testparallelbcsrvcdensity.sh; \
		n=`expr $$n - 1`; \
	done; \
	true

# CLEAN
clean:
	find . -name "*.o" -type f -delete 
	find . -name "prserial" -type f -delete 
	find . -name "prparallel" -type f -delete
	find . -name "prparallelbcsrtc" -type f -delete
	find . -name "prparallelbcsrvc" -type f -delete
	find . -name "prparallelOld" -type f -delete
	find . -name "prparallel2" -type f -delete

cleanwin:
	del /s *.o *.exe *.exp *.lib 

cleanallresults:
	find ./results -name "*.json" -type f -delete

cleanserialresults:
	find ./results -name "*serial_*.json" -type f -delete

cleanparallelresults:
	find ./results -name "*parallel_*.json" -type f -delete

cleanparallelbcsrtcresults:
	find ./results -name "*parallelbcsrtc_*.json" -type f -delete

cleanparallelbcsrvcresults:
	find ./results -name "*parallelbcsrvc_*.json" -type f -delete