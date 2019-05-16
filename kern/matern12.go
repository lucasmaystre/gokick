package kern

import (
	"gonum.org/v1/gonum/blas"
	"gonum.org/v1/gonum/blas/blas64"
	"math"
)

var (
	matern12 *Matern12
	_        Kernel = matern12 // Check that Matern32 respects the Kernel interface.
)

type Matern12 struct {
	variance float64
	lscale   float64
}

func NewMatern12(variance, lscale float64) *Matern12 {
	return &Matern12{
		variance: variance,
		lscale:   lscale,
	}
}

func (k *Matern12) Order() int {
	return 1
}

func (k *Matern12) StateMean(t float64) blas64.Vector {
	return blas64.Vector{
		Inc:  1,
		Data: []float64{0.0},
	}
}

func (k *Matern12) StateCov(t float64) blas64.Symmetric {
	return blas64.Symmetric{
		N:      1,
		Stride: 1,
		Data:   []float64{k.variance},
		Uplo:   blas.Upper,
	}
}

func (k *Matern12) MeasurementVec() blas64.Vector {
	return blas64.Vector{
		Inc:  1,
		Data: []float64{1.0},
	}
}

func (k *Matern12) Transition(delta float64) blas64.General {
	return blas64.General{
		Rows:   1,
		Cols:   1,
		Stride: 1,
		Data:   []float64{math.Exp(-delta / k.lscale)},
	}
}

func (k *Matern12) NoiseCov(delta float64) blas64.Symmetric {
	val := k.variance * (1 - math.Exp(-2 * delta / k.lscale))
	return blas64.Symmetric{
		N:      1,
		Stride: 1,
		Data:   []float64{val},
		Uplo:   blas.Upper,
	}
}
