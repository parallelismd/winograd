CC1 = gcc
CC2 = clang
CFLAGS1 = -std=c11 -fopenmp -O3 -g 
CFLAGS2 = -std=c11 -fopenmp -O3 -g -ljemalloc -flto -mcpu=tsv110
INCLUDES = -I/shareofs/apps/mpi/hmpi/2.3.0-bisheng3.2.0/include \
           -I/shareofs/apps/libs/kml/2.2.0-bisheng3.2.0/include
LIBS = -L/shareofs/apps/libs/kml/2.2.0-bisheng3.2.0/lib/kblas/locking -lkblas \
       -L/shareofs/apps/mpi/hmpi/2.3.0-bisheng3.2.0/lib -lmpi -flto
SRC1 = driver.c winograd.c
SRC2 = driver.c optimized.c
TARGET1 = winograd
TARGET2 = optimized

all: $(TARGET1) $(TARGET2)

MODULES = module use /scratch/apps/modules && \
          module load bisheng/3.2.0-aarch64 && \
          module load mpi/hmpi/2.3.0-bisheng3.2.0-aarch64 && \
          module load libs/kml/2.2.0-bisheng3.2.0-hmpi2.3.0-aarch64 &&  

$(TARGET1): $(SRC1)
	$(MODULES) $(CC1) $(CFLAGS1) $(INCLUDES) $^ $(LIBS) -o $@

$(TARGET2): $(SRC2)
	$(MODULES) $(CC2) $(CFLAGS2) $(INCLUDES) $^ $(LIBS) -o $@

.PHONY: clean

clean:
	rm -f $(TARGET1) $(TARGET2)

#all:
	
#	gcc -std=c11 -fopenmp -O3 -g driver.c winograd.c -o winograd
#	clang -std=c11 -fopenmp -O2 -g driver.c optimized.c -I/shareofs/apps/mpi/hmpi/2.3.0-bisheng3.2.0/include  -I/shareofs/apps/libs/kml/2.2.0-bisheng3.2.0/include  -L/shareofs/apps/libs/kml/2.2.0-bisheng3.2.0/lib/kblas/omp -lkblas -L/shareofs/apps/mpi/hmpi/2.3.0-bisheng3.2.0/lib -lmpi  -o optimized

# gcc -std=c11 -D__DEBUG -O0 -g driver.c winograd.c -o winograd
