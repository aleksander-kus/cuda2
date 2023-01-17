main:
	nvcc main.cu kmeansgpu.cu kmeanscpu.cu -o kmeans.out