package kern

import (
	"gonum.org/v1/gonum/mat"
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

func (k *Constant) StateMean(t float64) *mat.VecDense {
	return mat.NewVecDense(1, []float64{0.0})
}

func (k *Constant) StateCov(t float64) *mat.Dense {
	return mat.NewDense(1, 1, []float64{k.variance})
}

func (k *Constant) MeasurementVec() *mat.VecDense {
	return mat.NewVecDense(1, []float64{1.0})
}

func (k *Constant) Feedback() *mat.Dense {
	return mat.NewDense(1, 1, []float64{0.0})
}

func (k *Constant) NoiseEffect() *mat.Dense {
	return mat.NewDense(1, 1, []float64{1.0})
}

func (k *Constant) NoiseDensity() *mat.Dense {
	return mat.NewDense(1, 1, []float64{0.0})
}

func (k *Constant) Transition(delta float64) *mat.Dense {
	return mat.NewDense(1, 1, []float64{1.0})
}

func (k *Constant) NoiseCov(delta float64) *mat.Dense {
	return mat.NewDense(1, 1, []float64{0.0})
}
