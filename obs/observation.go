package obs

import (
	"c4science.ch/source/gokick/score"
	"math"
)

type Observation interface {
	UpdateApprox(damping float64) float64
	LogLikelihoodContrib() float64
}

type baseObservation struct {
	matchMoments func(m, v float64) (x, y, z float64)
	samples      []*score.Sample
	coeffs       []float64
	logpart      float64
}

func (o *baseObservation) UpdateApprox(damping float64) float64 {
	// Compute cavity mean and variance in the 1D subspace.
	meanCav := 0.0
	varCav := 0.0
	for i, sample := range o.samples {
		xCav, nCav := sample.CavityParams()
		c := o.coeffs[i]
		meanCav += c * (nCav / xCav)
		varCav += c * c / xCav
	}
	logpart, dlogpart, d2logpart := o.matchMoments(meanCav, varCav)
	// Update the Gaussian pseudo-observation associated to each score.
	for i, sample := range o.samples {
		xCav, nCav := sample.CavityParams()
		c := o.coeffs[i]
		denom := 1 + c*c*d2logpart/xCav
		x := -c * c * d2logpart / denom
		n := (c*dlogpart - c*c*(nCav/xCav)*d2logpart) / denom
		sample.UpdatePseudoObs(x, n, damping)
	}
	diff := o.logpart - logpart
	// Save log partition function value for the log-likelihood
	o.logpart = logpart
	return diff
}

func (o *baseObservation) LogLikelihoodContrib() float64 {
	contrib := o.logpart
	for _, sample := range o.samples {
		xCav, nCav := sample.CavityParams()
		xObs, nObs := sample.PseudoObsParams()
		contrib += (0.5*math.Log(xObs/xCav+1) +
			(-nObs*nObs-2*nObs*nCav+xObs*nCav*nCav/xCav)/(2*(xObs+xCav)))
	}
	return contrib
}
