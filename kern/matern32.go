package kern

import (
	"gonum.org/v1/gonum/blas"
	"gonum.org/v1/gonum/blas/blas64"
	"math"
)

var (
	matern32 *Matern32
	_        Kernel = matern32 // Check that Matern32 respects the Kernel interface.
)

type Matern32 struct {
	variance float64
	lambda   float64
}

func NewMatern32(variance, lscale float64) *Matern32 {
	return &Matern32{
		variance: variance,
		lambda:   math.Sqrt(3) / lscale,
	}
}

func (k *Matern32) Order() int {
	return 2
}

func (k *Matern32) StateMean(t float64) blas64.Vector {
	return blas64.Vector{
		Inc:  1,
		Data: []float64{0.0, 0.0},
	}
}

func (k *Matern32) StateCov(t float64) blas64.Symmetric {
	a := k.lambda
	data := []float64{1.0, 0.0, 0.0, a * a}
	for i := range data {
		data[i] *= k.variance
	}
	return blas64.Symmetric{
		N:      2,
		Stride: 2,
		Data:   data,
		Uplo:   blas.Upper,
	}
}

func (k *Matern32) MeasurementVec() blas64.Vector {
	return blas64.Vector{
		Inc:  1,
		Data: []float64{1.0, 0.0},
	}
}

func (k *Matern32) Transition(delta float64) blas64.General {
	d := delta
	a := k.lambda
	data := []float64{d*a + 1, d, -d * a * a, 1 - d*a}
	scale := math.Exp(-d * a)
	for i := range data {
		data[i] *= scale
	}
	return blas64.General{
		Rows:   2,
		Cols:   2,
		Stride: 2,
		Data:   data,
	}
}

func (k *Matern32) NoiseCov(delta float64) blas64.Symmetric {
	a := k.lambda
	da := delta * a
	c := math.Exp(-2 * da)
	data := make([]float64, 4)
	data[0] = 1 - c*(2*da*da+2*da+1)
	data[1] = c * (2 * da * da * a)
	data[2] = data[1]
	data[3] = a * a * (1 - c*(2*da*da-2*da+1))
	for i := range data {
		data[i] *= k.variance
	}
	return blas64.Symmetric{
		N:      2,
		Stride: 2,
		Data:   data,
		Uplo:   blas.Upper,
	}
}
