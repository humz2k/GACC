thing:
	nvcc eval.cu -lineinfo -Xptxas -v -Xcompiler="-fopenmp -fPIC" -shared -o eval.dll