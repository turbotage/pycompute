import numpy as np
import cupy as cp

ldl_function = """
{{linkage_qualifier}}
void ldl({{fp_type}}* mat) {
    {{fp_type}} arr[{{ndim}}];
    for (int i = 0; i < ndim; ++i) {
        {{fp_type}} d = mat[i*{{ndim}} + i];

        for (int j = i + 1; j < {{ndim}}; ++j) {
            arr[j] = mat[j*{{ndim}} + i];
            mat[j*{{ndim}} + i] /= d;
        }

        for (int j = i + 1; j < {{ndim}}; ++j) {
            {{fp_type}} aj = arr[j];
            for (int k = j; k < ndim; ++k) {
                mat[k*{{ndim}} + j] -= aj * mat[k*{{ndim}} + i];
            }
        }
    }
}
"""



x1 = cp.arange(25, dtype=cp.float32).reshape(5, 5)
x2 = cp.arange(25, dtype=cp.float32).reshape(5, 5)
y = cp.zeros((5,5), dtype=cp.float32)

add_kernel((5,), (5,), (x1, x2, y))

print(y)
