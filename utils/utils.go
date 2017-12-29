package utils

import (
	"errors"
	"fmt"
	"gonum.org/v1/gonum/blas"
	"gonum.org/v1/gonum/blas/blas64"
	"gonum.org/v1/gonum/mat"
	"math"
)

// Concatenate multiple vectors.
func ConcatVecs(size int, vecs ...blas64.Vector) blas64.Vector {
	data := make([]float64, size)
	offset := 0
	for _, vec := range vecs {
		copy(data[offset:], vec.Data)
		offset += len(vec.Data)
	}
	return blas64.Vector{
		Inc:  1,
		Data: data,
	}
}

// Make a general block diagonal matrix.
func BlockDiagGen(size int, mats ...blas64.General) blas64.General {
	data := make([]float64, size*size)
	offset := 0
	for _, mat := range mats {
		for i := 0; i < mat.Rows; i++ {
			// Copy i-th row.
			copy(data[size*(offset+i)+offset:],
				mat.Data[i*mat.Stride:i*mat.Stride+mat.Cols])
		}
		offset += mat.Rows
	}
	return blas64.General{
		Rows:   size,
		Cols:   size,
		Stride: size,
		Data:   data,
	}
}

// Make a symmetric block diagonal matrix.
func BlockDiagSym(size int, mats ...blas64.Symmetric) blas64.Symmetric {
	data := make([]float64, size*size)
	offset := 0
	for _, mat := range mats {
		for i := 0; i < mat.N; i++ {
			if mat.Uplo != blas.Upper {
				panic(errors.New("matrix not upper triangular"))
			}
			// Copy i-th row.
			copy(data[size*(offset+i)+offset+i:],
				mat.Data[i*mat.Stride+i:i*mat.Stride+mat.N])
		}
		offset += mat.N
	}
	return blas64.Symmetric{
		N:      size,
		Stride: size,
		Data:   data,
		Uplo:   blas.Upper,
	}
}

// Make an identity Matrix.
func Eye(size int) blas64.General {
	data := make([]float64, size*size)
	for i := 0; i < size; i++ {
		data[i*size+i] = 1.0
	}
	return blas64.General{
		Rows:   size,
		Cols:   size,
		Stride: size,
		Data:   data,
	}
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

// Print a `blas64` matrix.
func PrintBlas64Mat(blasMat interface{}) {
	var m mat.Matrix
	switch blasMat := blasMat.(type) {
	case blas64.General:
		tmp := mat.NewDense(blasMat.Rows, blasMat.Cols, nil)
		tmp.SetRawMatrix(blasMat)
		m = tmp
	case blas64.Symmetric:
		tmp := mat.NewSymDense(blasMat.N, nil)
		tmp.SetRawSymmetric(blasMat)
		m = tmp
	case blas64.Triangular:
		switch blasMat.Uplo {
		case blas.Upper:
			m = mat.NewTriDense(blasMat.N, mat.Upper, blasMat.Data)
		case blas.Lower:
			m = mat.NewTriDense(blasMat.N, mat.Lower, blasMat.Data)
		}
	default:
		panic(errors.New("matrix type not supported"))
	}
	f := mat.Formatted(m, mat.Squeeze())
	fmt.Printf("%v\n", f)
}
