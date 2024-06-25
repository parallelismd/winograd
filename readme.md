
# Winograd

## Environments 
```bash
module use /scratch/apps/modules 

module load bisheng/3.2.0-aarch64
module load mpi/hmpi/2.3.0-bisheng3.2.0-aarch64
module load libs/kml/2.2.0-bisheng3.2.0-hmpi2.3.0-aarch64 

```
# module use /scratch/apps/modules && module load bisheng/3.2.0-aarch64 && module load mpi/hmpi/2.3.0-bisheng3.2.0-aarch64 && module load libs/kml/2.2.0-bisheng3.2.0-hmpi2.3.0-aarch64
## Run
```
Benchmark mode: ./winograd small.conf 0
Validation mode: ./winograd small.conf 1
```
