package kern

import (
	"gonum.org/v1/gonum/blas/blas64"
)

type Kernel interface {
	// Order of the SDE :math:`m`.
	Order() int

	// Prior mean of the state vector, :math:`\mathbf{m}_0(t)`.
	StateMean(t float64) blas64.Vector

	// Prior covariance of the state vector, :math:`\mathbf{P}_0(t)`.
	StateCov(t float64) blas64.Symmetric

	// Measurement vector :math:`\mathbf{h}`.
	MeasurementVec() blas64.Vector

	// Transition matrix :math:`\mathbf{A}` for a given time interval.
	Transition(delta float64) blas64.General

	// Noise covariance matrix :math:`\mathbf{Q}` for a given time interval.
	NoiseCov(delta float64) blas64.Symmetric
}

// TODO This is not allowed, need to understand composition.
//func (k1 Kernel) Add(k2 Kernel) Kernel {
//	return NewAdd(k1, k2)
//}
