package kern

import (
	"gonum.org/v1/gonum/blas"
	"gonum.org/v1/gonum/blas/blas64"
)

var (
	constant *Constant
	_        Kernel = constant // Check that Constant respects the Kernel interface.
)

type Constant struct {
	variance float64
}

func NewConstant(variance float64) *Constant {
	return &Constant{
		variance: variance,
	}
}

func (k *Constant) Order() int {
	return 1
}

func (k *Constant) StateMean(t float64) blas64.Vector {
	return blas64.Vector{
		Inc:  1,
		Data: []float64{0.0},
	}
}

func (k *Constant) StateCov(t float64) blas64.Symmetric {
	return blas64.Symmetric{
		N:      1,
		Stride: 1,
		Data:   []float64{k.variance},
		Uplo:   blas.Upper,
	}
}

func (k *Constant) MeasurementVec() blas64.Vector {
	return blas64.Vector{
		Inc:  1,
		Data: []float64{1.0},
	}
}

func (k *Constant) Transition(delta float64) blas64.General {
	return blas64.General{
		Rows:   1,
		Cols:   1,
		Stride: 1,
		Data:   []float64{1.0},
	}
}

func (k *Constant) NoiseCov(delta float64) blas64.Symmetric {
	return blas64.Symmetric{
		N:      1,
		Stride: 1,
		Data:   []float64{0.0},
		Uplo:   blas.Upper,
	}
}
