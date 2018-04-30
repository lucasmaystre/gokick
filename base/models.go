package base

import (
	"c4science.ch/source/gokick/kern"
	"c4science.ch/source/gokick/obs"
	"c4science.ch/source/gokick/score"
	"fmt"
	"math"
	"sync"
)

type Model interface {
	// Fit the model.
	Fit(damping float64, nWorkers, maxIter int, verbose bool) bool
	LogLikelihood() float64
}

type baseModel struct {
	items        map[string]*Item
	observations []obs.Observation
}

func (m *baseModel) Fit(damping float64, nWorkers, maxIter int, verbose bool) {
	itemChan := make(chan *Item, 100)
	defer close(itemChan)
	var wg sync.WaitGroup

	for i := 0; i < nWorkers; i++ {
		go func() {
			for item := range itemChan {
				item.Fit()
				wg.Done()
			}
		}()
	}

	for i := 0; i < maxIter; i++ {
		max := 0.0
		for _, o := range m.observations {
			diff := o.UpdateApprox(damping)
			max = math.Max(max, math.Abs(diff))
		}
		for _, item := range m.items {
			wg.Add(1)
			itemChan <- item
		}
		wg.Wait()
		if verbose {
			fmt.Printf("Iteration %v, max diff: %.8f\n", i+1, max)
		}
	}
}

func (m *baseModel) processItems(winners, losers map[string]float64,
	time float64) (samples []*score.Sample, coeffs []float64) {
	nElems := len(winners) + len(losers)
	samples = make([]*score.Sample, nElems)
	coeffs = make([]float64, nElems)
	i := 0
	for name, coeff := range winners {
		samples[i] = m.items[name].AddSample(time)
		coeffs[i] = coeff
		i++
	}
	for name, coeff := range losers {
		samples[i] = m.items[name].AddSample(time)
		coeffs[i] = -coeff
		i++
	}
	return
}

func (m *baseModel) AddItem(name string, kernel kern.Kernel) {
	m.items[name] = NewItem(kernel)
}

func (m *baseModel) Item(name string) *Item {
	return m.items[name]
}

func (m *baseModel) LogLikelihood() float64 {
	ll := 0.0
	for _, o := range m.observations {
		ll += o.LogLikelihoodContrib()
	}
	for _, item := range m.items {
		ll += item.LogLikelihoodContrib()
	}
	return ll
}

// Model for comparisons with three possible outcomes: win, loss, tie.
type TernaryModel struct {
	baseModel
	margin float64
}

func NewTernaryModel(margin float64) *TernaryModel {
	return &TernaryModel{
		baseModel: baseModel{
			items:        make(map[string]*Item),
			observations: make([]obs.Observation, 0, 10),
		},
		margin: margin,
	}
}

func (m *TernaryModel) Observe(winners, losers map[string]float64,
	time float64, isTie bool) {
	samples, coeffs := m.processItems(winners, losers, time)
	if isTie {
		o := obs.NewProbitTie(samples, coeffs, time, m.margin)
		m.observations = append(m.observations, o)
	} else {
		o := obs.NewProbit(samples, coeffs, time, m.margin)
		m.observations = append(m.observations, o)
	}
}

// Model for comparisons with two possible outcomes: win and loss.
type BinaryModel struct {
	baseModel
}

func NewBinaryModel() *BinaryModel {
	return &BinaryModel{
		baseModel: baseModel{
			items:        make(map[string]*Item),
			observations: make([]obs.Observation, 0, 10),
		},
	}
}

func (m *BinaryModel) Observe(winners, losers map[string]float64,
	time float64) {
	samples, coeffs := m.processItems(winners, losers, time)
	o := obs.NewProbit(samples, coeffs, time, 0.0)
	m.observations = append(m.observations, o)
}
