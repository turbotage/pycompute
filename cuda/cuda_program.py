import cupy as cp

class CudaTensor:
    def __init__(self, shape: list[int], dtype: cp.dtype):
        self.shape = shape
        self.dtype = dtype

class CudaFunction:
    def __init__(self):
        self.deps: dict[str, CudaFunction] = []

    def gen_code(self):
        raise NotImplementedError()

    def get_deps(self):
        return self.deps
