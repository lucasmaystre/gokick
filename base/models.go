package base

import (
	"github.com/lucasmaystre/gokick/kern"
	"github.com/lucasmaystre/gokick/obs"
	"github.com/lucasmaystre/gokick/score"
	"fmt"
	"math"
	"sync"
)

type Model interface {
	// Fit the model.
	Fit(damping float64, nWorkers, maxIter int, verbose bool) bool
	// Compute the marginal log-likelihood.
	LogLikelihood() float64
}

type baseModel struct {
	items        map[string]*Item
	observations []obs.Observation
}

func (m *baseModel) Fit(damping float64, nWorkers, maxIter int,
	verbose bool) bool {
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
		if max < 1e-3 {
			return true
		}
	}

	return false
}

func (m *baseModel) processItems(winners, losers map[string]float64) (
	procs []*score.Process, coeffs []float64) {
	nElems := len(winners) + len(losers)
	procs = make([]*score.Process, nElems)
	coeffs = make([]float64, nElems)
	i := 0
	for name, coeff := range winners {
		procs[i] = &m.items[name].Process
		coeffs[i] = coeff
		i++
	}
	for name, coeff := range losers {
		procs[i] = &m.items[name].Process
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
	procs, coeffs := m.processItems(winners, losers)
	if isTie {
		o := obs.NewTie(procs, coeffs, time, m.margin)
		m.observations = append(m.observations, o)
	} else {
		o := obs.NewWin(procs, coeffs, time, m.margin)
		m.observations = append(m.observations, o)
	}
}

func (m *TernaryModel) Probabilities(team1, team2 map[string]float64,
	time float64) (win, draw, loss float64) {
	procs, coeffs := m.processItems(team1, team2)
	win = obs.WinProbability(procs, coeffs, time, m.margin)
	draw = obs.TieProbability(procs, coeffs, time, m.margin)
	loss = 1.0 - win - draw
	return
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
	procs, coeffs := m.processItems(winners, losers)
	o := obs.NewWin(procs, coeffs, time, 0.0)
	m.observations = append(m.observations, o)
}
