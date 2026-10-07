#[pyo3::pymodule]
mod _core {
    use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
    use pyo3::exceptions::PyValueError;
    use pyo3::prelude::*;

    /// Numerically stable softmax over a 1-D array.
    #[pyfunction]
    fn softmax<'py>(
        py: Python<'py>,
        x: PyReadonlyArray1<'py, f64>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let x = x.as_array();
        if x.is_empty() {
            return Err(PyValueError::new_err("softmax of empty array"));
        }
        let max = x.fold(f64::NEG_INFINITY, |m, &v| m.max(v));
        let exps = x.mapv(|v| (v - max).exp());
        let sum = exps.sum();
        Ok((exps / sum).into_pyarray(py))
    }

    /// Transpose
    #[pyfunction]
    fn transpose<'py>(
        py: Python<'py>,
        x: PyReadonlyArray2<'py, f64>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let x = x.as_array();

        if x.is_empty() {
            return Err(PyValueError::new_err("transpose of empty array"));
        }

        let (rows, cols) = x.dim();

        let mut out = numpy::ndarray::Array2::<f64>::zeros((cols, rows));

        for col in 0..cols {
            for row in 0..rows {
                out[[col, row]] = x[[row, col]];
            }
        }

        Ok(out.into_pyarray(py))
    }

    /// Matrix multiplication: C = A @ B
    #[pyfunction]
    fn matmul<'py>(
        py: Python<'py>,
        a: PyReadonlyArray2<'py, f64>,
        b: PyReadonlyArray2<'py, f64>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let a = a.as_array();
        let b = b.as_array();

        if a.is_empty() || b.is_empty() {
            return Err(PyValueError::new_err(
                "matmul requires non-empty 2-D arrays",
            ));
        }

        let (a_rows, a_cols) = a.dim();
        let (b_rows, b_cols) = b.dim();

        if a_cols != b_rows {
            return Err(PyValueError::new_err(format!(
                "matmul shape mismatch: ({}, {}) cannot multiply ({}, {})",
                a_rows, a_cols, b_rows, b_cols
            )));
        }

        let mut out = numpy::ndarray::Array2::<f64>::zeros((a_rows, b_cols));

        for i in 0..a_rows {
            for j in 0..b_cols {
                let mut sum = 0.0;

                for k in 0..a_cols {
                    sum += a[[i, k]] * b[[k, j]];
                }

                out[[i, j]] = sum;
            }
        }

        Ok(out.into_pyarray(py))
    }
}
