package kernels

import (
	"gonum.org/v1/gonum/mat"
	"math"
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

func (k *Matern32) StateMean(t float64) *mat.VecDense {
	return mat.NewVecDense(2, []float64{0.0, 0.0})
}

func (k *Matern32) StateCov(t float64) *mat.Dense {
	a := k.lambda
	cov := mat.NewDense(2, 2, []float64{1.0, 0.0, 0.0, a * a})
	cov.Scale(k.variance, cov)
	return cov
}

func (k *Matern32) MeasurementVec() *mat.VecDense {
	return mat.NewVecDense(2, []float64{1.0, 0.0})
}

func (k *Matern32) Feedback() *mat.Dense {
	a := k.lambda
	return mat.NewDense(2, 2, []float64{0.0, 1.0, -a * a, -2 * a})
}

func (k *Matern32) NoiseEffect() *mat.Dense {
	return mat.NewDense(1, 2, []float64{0.0, 1.0})
}

func (k *Matern32) NoiseDensity() *mat.Dense {
	val := 4.0 * k.variance * math.Pow(k.lambda, 3.0)
	return mat.NewDense(1, 1, []float64{val})
}

func (k *Matern32) Transition(delta float64) *mat.Dense {
	d := delta
	a := k.lambda
	mat := mat.NewDense(2, 2, []float64{d*a + 1, d, -d * a * a, 1 - d*a})
	mat.Scale(math.Exp(-d*a), mat)
	return mat
}

func (k *Matern32) NoiseCov(delta float64) *mat.Dense {
	d := delta
	a := k.lambda
	da := d * a
	c := math.Exp(-2 * da)
	data := make([]float64, 4)
	data[0] = 1 - c*(2*da*da+2*da+1)
	data[1] = c * (2 * da * da * a)
	data[2] = data[1]
	data[3] = a * a * (1 - c*(2*da*da-2*da+1))
	cov := mat.NewDense(2, 2, data)
	cov.Scale(k.variance, cov)
	return cov
}
