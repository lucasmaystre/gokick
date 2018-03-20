package elo

import (
	"errors"
	"math"
)

var ErrNotChronological = errors.New("observation not in chronological order")

type baseModel struct {
	ratings map[string]float64
	time    float64
	k		float64
}

func (m *baseModel) Rating(name string) float64 {
	if val, ok := m.ratings[name]; ok {
		return val
	} else {
		return 0.0
	}
}

func (m *baseModel) UpdateRating(name string, delta, time float64) {
	if time < m.time {
		panic(ErrNotChronological)
	}
	if val, ok := m.ratings[name]; ok {
		m.ratings[name] = val + delta
	} else {
		m.ratings[name] = 0.0 + delta
	}
	m.time = time
}

type BinaryModel struct {
	baseModel
}

func NewBinaryModel(k float64) *BinaryModel {
	return &BinaryModel{
		baseModel: baseModel{
			ratings: make(map[string]float64),
			time:    math.Inf(-1),
			k:       k,
		},
	}
}

func (m *BinaryModel) Predict(a, b string) (float64, float64) {
	ra := m.Rating(a)
	rb := m.Rating(b)
	prob := 1.0 / (1.0 + math.Exp(-(ra - rb)))
	return prob, 1.0 - prob
}

func (m *BinaryModel) Observe(winner, loser string, time float64) {
	probw, _ := m.Predict(winner, loser)
	delta := m.k * (1.0 - probw)
	m.UpdateRating(winner, +delta, time)
	m.UpdateRating(loser, -delta, time)
}

type TernaryModel struct {
	baseModel
	margin float64
}

func NewTernaryModel(k, margin float64) *TernaryModel {
	return &TernaryModel{
		baseModel: baseModel{
			ratings: make(map[string]float64),
			time:    math.Inf(-1),
			k:       k,
		},
		margin:    margin,
	}
}

func (m *TernaryModel) Predict(a, b string) (float64, float64, float64) {
	ra := m.Rating(a)
	rb := m.Rating(b)
	proba := 1.0 / (1.0 + math.Exp(-(ra - rb - m.margin)))
	probb := 1.0 / (1.0 + math.Exp(-(rb - ra - m.margin)))
	probt := 1.0 - proba - probb
	return proba, probt, probb
}

func (m *TernaryModel) Observe(winner, loser string,
	time float64, isTie bool) {
	probw, _, probl := m.Predict(winner, loser)
	if isTie {
		delta := m.k * (probl - probw)
		m.UpdateRating(winner, +delta, time)
		m.UpdateRating(loser, -delta, time)
	} else {
		delta := m.k * (1.0 - probw)
		m.UpdateRating(winner, +delta, time)
		m.UpdateRating(loser, -delta, time)
	}
}
