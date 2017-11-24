package fitters

import (
	"c4science.ch/source/gokick/kernels"
	"c4science.ch/source/gokick/utils"
	"errors"
	"gonum.org/v1/gonum/mat"
)

var ErrNotChronological = errors.New("observation not in chronological order")

type Recursive struct {
	kernel kernels.Kernel
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

func NewRecursive(kernel kernels.Kernel) *Recursive {
	return &Recursive{
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

func (f *Recursive) AddSample(t, n, x float64) {
	idx := len(f.Ts)
	if idx > 0 && t < f.Ts[idx-1] {
		panic(ErrNotChronological)
	}
	f.Ts = append(f.Ts, t)
	f.Ms = append(f.Ms, 0.0)
	f.Vs = append(f.Vs, 0.0)
	f.Ns = append(f.Ns, n)
	f.Xs = append(f.Xs, x)
	m := f.kernel.Order()
	f._m_p = append(f._m_p, mat.NewVecDense(m, nil))
	f._P_p = append(f._P_p, mat.NewDense(m, m, nil))
	f._m_f = append(f._m_f, mat.NewVecDense(m, nil))
	f._P_f = append(f._P_f, mat.NewDense(m, m, nil))
	f._m_s = append(f._m_s, mat.NewVecDense(m, nil))
	f._P_s = append(f._P_s, mat.NewDense(m, m, nil))
	if idx > 0 {
		delta := t - f.Ts[idx-1]
		f._A = append(f._A, f.kernel.Transition(delta))
		f._Q = append(f._Q, f.kernel.NoiseCov(delta))
	}
}

func (f *Recursive) Fit() {
	var (
		ts  = f.Ts
		ms  = f.Ms
		vs  = f.Vs
		ns  = f.Ns
		xs  = f.Xs
		h   = f._h
		I   = f._I
		A   = f._A
		Q   = f._Q
		m_p = f._m_p
		P_p = f._P_p
		m_f = f._m_f
		P_f = f._P_f
		m_s = f._m_s
		P_s = f._P_s
	)
	var k, tmp1 mat.VecDense
	var Z, G, tmp2 mat.Dense
	// Forward pass (Kalman filter).
	for i := 0; i < len(f.Ts); i++ {
		if i == 0 {
			m_p[i] = f.kernel.StateMean(ts[i])
			P_p[i] = f.kernel.StateCov(ts[i])
		} else {
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
	for i := len(f.Ts) - 1; i >= 0; i-- {
		if i == len(f.Ts)-1 {
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

func (f *Recursive) Predict(ts []float64) (ms, vs []float64) {
	return
}
