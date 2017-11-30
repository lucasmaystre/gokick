package score

import (
	"c4science.ch/source/gokick/kern"
	"c4science.ch/source/gokick/utils"
	"errors"
	"gonum.org/v1/gonum/mat"
)

var ErrNotChronological = errors.New("observation not in chronological order")

type Sample struct {
	process *Process
	idx     int
}

func (s *Sample) CavityParams() (xCav, nCav float64) {
	xTot := 1.0 / s.process.Vs[s.idx]
	nTot := xTot * s.process.Ms[s.idx]
	xCav = xTot - s.process.Xs[s.idx]
	nCav = nTot - s.process.Ns[s.idx]
	return
}

func (s *Sample) PseudoObsParams() (float64, float64) {
	return s.process.Xs[s.idx], s.process.Ns[s.idx]
}

func (s *Sample) UpdatePseudoObs(x, n, damping float64) {
	s.process.Xs[s.idx] = (1-damping)*s.process.Xs[s.idx] + damping*x
	s.process.Ns[s.idx] = (1-damping)*s.process.Ns[s.idx] + damping*n
}

type Process struct {
	kernel kern.Kernel
	Ts     []float64
	Ms     []float64
	Vs     []float64
	Ns     []float64
	Xs     []float64
	_h     *mat.VecDense
	_I     *mat.Dense
	_A     []*mat.Dense
	_Q     []*mat.Dense
	_m_p   []*mat.VecDense
	_P_p   []*mat.Dense
	_m_f   []*mat.VecDense
	_P_f   []*mat.Dense
	_m_s   []*mat.VecDense
	_P_s   []*mat.Dense
}

func NewProcess(kernel kern.Kernel) *Process {
	return &Process{
		kernel: kernel,
		Ts:     make([]float64, 0, 10),
		Ms:     make([]float64, 0, 10),
		Vs:     make([]float64, 0, 10),
		Ns:     make([]float64, 0, 10),
		Xs:     make([]float64, 0, 10),
		_h:     kernel.MeasurementVec(),
		_I:     utils.Eye(kernel.Order()),
		_A:     make([]*mat.Dense, 0, 10),
		_Q:     make([]*mat.Dense, 0, 10),
		_m_p:   make([]*mat.VecDense, 0, 10),
		_P_p:   make([]*mat.Dense, 0, 10),
		_m_f:   make([]*mat.VecDense, 0, 10),
		_P_f:   make([]*mat.Dense, 0, 10),
		_m_s:   make([]*mat.VecDense, 0, 10),
		_P_s:   make([]*mat.Dense, 0, 10),
	}
}

func (p *Process) AddSample(time float64) *Sample {
	idx := len(p.Ts)
	if idx > 0 && time < p.Ts[idx-1] {
		panic(ErrNotChronological)
	}
	p.Ts = append(p.Ts, time)
	p.Ns = append(p.Ns, 0.0)
	p.Xs = append(p.Xs, 0.0)
	p._m_p = append(p._m_p, p.kernel.StateMean(time))
	p._P_p = append(p._P_p, p.kernel.StateCov(time))
	p._m_f = append(p._m_f, p.kernel.StateMean(time))
	p._P_f = append(p._P_f, p.kernel.StateCov(time))
	p._m_s = append(p._m_s, p.kernel.StateMean(time))
	p._P_s = append(p._P_s, p.kernel.StateCov(time))
	// Initialize mean and variance.
	h := p.kernel.MeasurementVec()
	var tmp mat.VecDense
	tmp.MulVec(p.kernel.StateCov(time).T(), h)
	p.Ms = append(p.Ms, mat.Dot(h, p.kernel.StateMean(time)))
	p.Vs = append(p.Vs, mat.Dot(&tmp, h))
	if idx > 0 {
		delta := time - p.Ts[idx-1]
		p._A = append(p._A, p.kernel.Transition(delta))
		p._Q = append(p._Q, p.kernel.NoiseCov(delta))
	}
	return &Sample{
		process: p,
		idx:     idx,
	}
}

func (p *Process) Fit() {
	var (
		ts  = p.Ts
		ms  = p.Ms
		vs  = p.Vs
		ns  = p.Ns
		xs  = p.Xs
		h   = p._h
		I   = p._I
		A   = p._A
		Q   = p._Q
		m_p = p._m_p
		P_p = p._P_p
		m_f = p._m_f
		P_f = p._P_f
		m_s = p._m_s
		P_s = p._P_s
	)
	var k, tmp1 mat.VecDense
	var Z, G, tmp2 mat.Dense
	// Forward pass (Kalman filter).
	for i := 0; i < len(ts); i++ {
		if i > 0 {
			m_p[i].MulVec(A[i-1], m_f[i-1])
			P_p[i].Product(A[i-1], P_f[i-1], A[i-1].T())
			P_p[i].Add(P_p[i], Q[i-1])
		}
		// k = np.dot(P_p[i], h) / (1 + xs[i] * np.dot(np.dot(h, P_p[i]), h))
		tmp1.MulVec(P_p[i].T(), h)
		k.MulVec(P_p[i], h)
		k.ScaleVec(1.0/(1+xs[i]*mat.Dot(&tmp1, h)), &k)
		// m_f[i] = m_p[i] + k * (ns[i] - xs[i] * np.dot(h, m_p[i]))
		m_f[i].ScaleVec(ns[i]-xs[i]*mat.Dot(h, m_p[i]), &k)
		m_f[i].AddVec(m_f[i], m_p[i])
		// Z = I - np.outer(xs[i] * k, h)
		Z.Outer(xs[i], &k, h)
		Z.Sub(I, &Z)
		// P_f[i] = np.dot(np.dot(Z, P_p[i]), Z.T) + xs[i] * np.outer(k, k)
		tmp2.Outer(xs[i], &k, &k)
		P_f[i].Product(&Z, P_p[i], Z.T())
		P_f[i].Add(P_f[i], &tmp2)
	}
	// Backward pass (RTS smoother).
	for i := len(ts) - 1; i >= 0; i-- {
		if i == len(ts)-1 {
			m_s[i] = m_f[i]
			P_s[i] = P_f[i]
		} else {
			// G = np.linalg.solve(P_p[i+1], np.dot(A[i], P_f[i]))
			tmp2.Mul(A[i], P_f[i])
			G.Solve(P_p[i+1], &tmp2)
			// m_s[i] = m_f[i] + np.dot(G.T, m_s[i+1] - m_p[i+1])
			tmp1.SubVec(m_s[i+1], m_p[i+1])
			tmp1.MulVec(G.T(), &tmp1)
			m_s[i].AddVec(m_f[i], &tmp1)
			// P_s[i] = P_f[i] + np.dot(np.dot(G.T, P_s[i+1] - P_p[i+1]), G)
			tmp2.Sub(P_s[i+1], P_p[i+1])
			tmp2.Product(G.T(), &tmp2, &G)
			P_s[i].Add(P_f[i], &tmp2)
		}
		ms[i] = mat.Dot(h, m_s[i])
		tmp1.MulVec(P_s[i].T(), h)
		vs[i] = mat.Dot(&tmp1, h)
	}
}

func (p *Process) Predict(ts []float64) (ms, vs []float64) {
	return
}
