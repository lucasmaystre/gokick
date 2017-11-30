package obs

import (
	"c4science.ch/source/gokick/score"
	"c4science.ch/source/gokick/utils"
	"math"
)

var (
	cs = []float64{
		0.00048204, -0.00142906, 0.0013200243174, 0.0009461589032,
		-0.0045563339802, 0.00556964649138, 0.00125993961762116,
		-0.01621575378835404, 0.02629651521057465, -0.001829764677455021,
		-0.09439510239319526, 0.28613578213673563, 1.0, 1.0}
	rs = []float64{
		1.2753666447299659525, 5.019049726784267463450, 6.1602098531096305441,
		7.409740605964741794425, 2.9788656263939928886}
	qs = []float64{
		2.260528520767326969592, 9.3960340162350541504,
		12.048951927855129036034, 17.081440747466004316,
		9.608965327192787870698, 3.3690752069827527677}
)

func logPhi(z float64) (res, dres float64) {
	if z*z < 0.0492 {
		coef := -z / (math.Sqrt2 * math.SqrtPi)
		val := 0.0
		for _, c := range cs {
			val = coef * (c + val)
		}
		res = -2*val - math.Ln2
		dres = math.Exp(-(z*z)/2-res) / (math.Sqrt2 * math.SqrtPi)
	} else if z < -11.3137 {
		num := 0.5641895835477550741
		for _, r := range rs {
			num = -z*num/math.Sqrt2 + r
		}
		den := 1.0
		for _, q := range qs {
			den = -z*den/math.Sqrt2 + q
		}
		res = math.Log(num/(2*den)) - (z*z)/2
		dres = math.Abs(den/num) * math.Sqrt(2.0/math.Pi)
	} else {
		res = math.Log(utils.NormalCdf(z))
		dres = math.Exp(-(z*z)/2-res) / (math.Sqrt2 * math.SqrtPi)
	}
	return
}

func matchMomentsProbit(meanCav, varCav float64) (
	logpart, dlogpart, d2logpart float64) {
	denom := math.Sqrt(1 + varCav)
	z := meanCav / denom
	logpart, val := logPhi(z)
	dlogpart = val / denom
	d2logpart = -val * (z + val) / (1 + varCav)
	return
}

func matchMomentsProbitTie(meanCav, varCav, margin float64) (
	logpart, dlogpart, d2logpart float64) {
	denom := math.Sqrt(1 + varCav)
	z1 := (meanCav + margin) / denom
	z2 := (meanCav - margin) / denom
	phi1 := utils.NormalCdf(z1)
	phi2 := utils.NormalCdf(z2)
	v1 := utils.NormalPdf(z1)
	v2 := utils.NormalPdf(z2)
	logpart = math.Log(phi1 - phi2)
	dlogpart = (v1 - v2) / (denom * (phi1 - phi2))
	d2logpart = (-z1*v1+z2*v2)/((1+varCav)*(phi1-phi2)) -
		dlogpart*dlogpart
	return
}

func NewProbit(samples []*score.Sample, coeffs []float64,
	time, margin float64) Observation {
	return &baseObservation{
		samples: samples,
		coeffs:  coeffs,
		matchMoments: func(m, v float64) (x, y, z float64) {
			return matchMomentsProbit(m-margin, v)
		},
	}
}

func NewProbitTie(samples []*score.Sample, coeffs []float64,
	time, margin float64) Observation {
	return &baseObservation{
		samples: samples,
		coeffs:  coeffs,
		matchMoments: func(m, v float64) (x, y, z float64) {
			return matchMomentsProbitTie(m, v, margin)
		},
	}
}
