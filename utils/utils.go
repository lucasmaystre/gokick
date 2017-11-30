package utils

import (
	"gonum.org/v1/gonum/mat"
	"math"
)

// Concatenate multiple vectors.
func ConcatVecs(size int, vecs ...*mat.VecDense) *mat.VecDense {
	out := mat.NewVecDense(size, nil)
	offset := 0
	var slice *mat.VecDense
	for _, vec := range vecs {
		slice = out.SliceVec(offset, size)
		slice.CopyVec(vec)
		offset += vec.Len()
	}
	return out
}

// Make a block diagonal matrix.
func BlockDiag(size int, mats ...mat.Matrix) *mat.Dense {
	out := mat.NewDense(size, size, nil)
	offset := 0
	var r int
	var slice mat.Matrix
	for _, matrix := range mats {
		slice = out.Slice(offset, size, offset, size)
		slice.(*mat.Dense).Copy(matrix)
		r, _ = matrix.Dims()
		offset += r
	}
	return out
}

// Identity Matrix.
func Eye(n int) *mat.Dense {
	out := mat.NewDense(n, n, nil)
	for i := 0; i < n; i++ {
		out.Set(i, i, 1)
	}
	return out
}

// Normal probability density function.
func NormalPdf(x float64) float64 {
	return math.Exp(-x*x/2.0) / (math.Sqrt2 * math.SqrtPi)
}

// Normal cumulative density function.
func NormalCdf(x float64) float64 {
	// If X ~ N(0,1), returns P(X < x).
	return math.Erfc(-x/math.Sqrt2) / 2.0
}
