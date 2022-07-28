eval:
	nvcc cuda_eval.cu kernels/*.cu -lineinfo -Xptxas -v -Xcompiler="-fopenmp -fPIC" -shared -o cuda_eval.dll

eval2:
	nvcc eval.cu kernels/*.cu -lineinfo -Xptxas -v -Xcompiler="-fopenmp -fPIC" -o eval

solve:
	nvcc force_solver.cu -lineinfo -Xptxas -v -Xcompiler="-fopenmp -fPIC" -shared -o force_solver.dll