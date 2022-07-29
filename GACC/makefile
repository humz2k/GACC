f4:
	nvcc cuda_eval.cu kernels/*.cu -lineinfo -Xptxas -v -Xcompiler="-fopenmp -fPIC" -shared -o cuda_eval_f4.dll

f8:
	nvcc cuda_eval.cu kernels/*.cu -lineinfo -Xptxas -v -Xcompiler="-fopenmp -fPIC" -shared -o cuda_eval_f8.dll

ompf4:
	g++ omp_eval.cpp -fopenmp -fPIC -shared -o omp_eval_f4.dll

ompf8:
	g++ omp_eval.cpp -fopenmp -fPIC -shared -o omp_eval_f8.dll

half:
	nvcc cuda_half.cu half_kernels/*.cu -arch=sm_53 -lineinfo -Xptxas -v -Xcompiler="-fopenmp -fPIC" -shared -o cuda_eval_f2.dll

cheap:
	nvcc cuda_cheap.cu cheap_kernels/*.cu -arch=sm_53 -lineinfo -Xptxas -v -Xcompiler="-fopenmp -fPIC" -shared -o cuda_eval_cheap.dll

