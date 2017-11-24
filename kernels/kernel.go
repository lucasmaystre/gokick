package kernels

import (
	"gonum.org/v1/gonum/mat"
)

type Kernel interface {
	// Order of the SDE :math:`m`.
	Order() int

	// Prior mean of the state vector, :math:`\mathbf{m}_0(t)`.
	StateMean(t float64) *mat.VecDense

	// Prior covariance of the state vector, :math:`\mathbf{P}_0(t)`.
	StateCov(t float64) *mat.Dense

	// Measurement vector :math:`\mathbf{h}`.
	MeasurementVec() *mat.VecDense

	// Feedback matrix :math:`\mathbf{F}`.
	Feedback() *mat.Dense

	// Noise effect matrix :math:`\mathbf{L}`.
	NoiseEffect() *mat.Dense

	// Power spectral density of the noise :math:`\mathbf{Q}`.
	NoiseDensity() *mat.Dense

	// Transition matrix :math:`\mathbf{A}` for a given time interval.
	Transition(delta float64) *mat.Dense

	// Noise covariance matrix :math:`\mathbf{Q}` for a given time interval.
	NoiseCov(delta float64) *mat.Dense
}

// TODO This is not allowed, need to understand composition.
//func (k1 Kernel) Add(k2 Kernel) Kernel {
//	return NewAdd(k1, k2)
//}
