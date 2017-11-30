package kern

import (
	"c4science.ch/source/gokick/utils"
	"gonum.org/v1/gonum/mat"
)

type Add struct {
	parts []Kernel
	order int
}

func NewAdd(first, second Kernel) *Add {
	parts := make([]Kernel, 0, 2)
	switch first := first.(type) {
	case *Add:
		parts = append(parts, first.parts...)
	default:
		parts = append(parts, first)
	}
	switch second := second.(type) {
	case *Add:
		parts = append(parts, second.parts...)
	default:
		parts = append(parts, second)
	}
	order := 0
	for _, part := range parts {
		order += part.Order()
	}
	return &Add{
		parts: parts,
		order: order,
	}
}

func (k *Add) Order() int {
	return k.order
}

func (k *Add) StateMean(t float64) *mat.VecDense {
	vecs := make([]*mat.VecDense, len(k.parts))
	for i, part := range k.parts {
		vecs[i] = part.StateMean(t)
	}
	return utils.ConcatVecs(k.order, vecs...)
}

func (k *Add) StateCov(t float64) *mat.Dense {
	mats := make([]mat.Matrix, len(k.parts))
	for i, part := range k.parts {
		mats[i] = part.StateCov(t)
	}
	return utils.BlockDiag(k.order, mats...)
}

func (k *Add) MeasurementVec() *mat.VecDense {
	vecs := make([]*mat.VecDense, len(k.parts))
	for i, part := range k.parts {
		vecs[i] = part.MeasurementVec()
	}
	return utils.ConcatVecs(k.order, vecs...)
}

func (k *Add) Feedback() *mat.Dense {
	mats := make([]mat.Matrix, len(k.parts))
	for i, part := range k.parts {
		mats[i] = part.Feedback()
	}
	return utils.BlockDiag(k.order, mats...)
}

func (k *Add) NoiseEffect() *mat.Dense {
	mats := make([]mat.Matrix, len(k.parts))
	for i, part := range k.parts {
		mats[i] = part.NoiseEffect()
	}
	return utils.BlockDiag(k.order, mats...)
}

func (k *Add) NoiseDensity() *mat.Dense {
	mats := make([]mat.Matrix, len(k.parts))
	for i, part := range k.parts {
		mats[i] = part.NoiseDensity()
	}
	return utils.BlockDiag(k.order, mats...)
}

func (k *Add) Transition(delta float64) *mat.Dense {
	mats := make([]mat.Matrix, len(k.parts))
	for i, part := range k.parts {
		mats[i] = part.Transition(delta)
	}
	return utils.BlockDiag(k.order, mats...)
}

func (k *Add) NoiseCov(delta float64) *mat.Dense {
	mats := make([]mat.Matrix, len(k.parts))
	for i, part := range k.parts {
		mats[i] = part.NoiseCov(delta)
	}
	return utils.BlockDiag(k.order, mats...)
}
