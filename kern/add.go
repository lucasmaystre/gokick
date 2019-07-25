package kern

import (
	"github.com/lucasmaystre/gokick/utils"
	"gonum.org/v1/gonum/blas/blas64"
)

var (
	add *Add
	_   Kernel = add // Check that Add respects the Kernel interface.
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

func (k *Add) StateMean(t float64) blas64.Vector {
	vecs := make([]blas64.Vector, len(k.parts))
	for i, part := range k.parts {
		vecs[i] = part.StateMean(t)
	}
	return utils.ConcatVecs(k.order, vecs...)
}

func (k *Add) StateCov(t float64) blas64.Symmetric {
	mats := make([]blas64.Symmetric, len(k.parts))
	for i, part := range k.parts {
		mats[i] = part.StateCov(t)
	}
	return utils.BlockDiagSym(k.order, mats...)
}

func (k *Add) MeasurementVec() blas64.Vector {
	vecs := make([]blas64.Vector, len(k.parts))
	for i, part := range k.parts {
		vecs[i] = part.MeasurementVec()
	}
	return utils.ConcatVecs(k.order, vecs...)
}

func (k *Add) Transition(delta float64) blas64.General {
	mats := make([]blas64.General, len(k.parts))
	for i, part := range k.parts {
		mats[i] = part.Transition(delta)
	}
	return utils.BlockDiagGen(k.order, mats...)
}

func (k *Add) NoiseCov(delta float64) blas64.Symmetric {
	mats := make([]blas64.Symmetric, len(k.parts))
	for i, part := range k.parts {
		mats[i] = part.NoiseCov(delta)
	}
	return utils.BlockDiagSym(k.order, mats...)
}
