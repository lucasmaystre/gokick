package score

import (
	"c4science.ch/source/gokick/kern"
	"c4science.ch/source/gokick/utils"
	"errors"
	"gonum.org/v1/gonum/blas"
	"gonum.org/v1/gonum/blas/blas64"
	"gonum.org/v1/gonum/lapack/lapack64"
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
	_h     blas64.Vector
	_A     []blas64.General
	_Q     []blas64.Symmetric
	_m_p   []blas64.Vector
	_P_p   []blas64.Symmetric
	_U     []blas64.Triangular
	_m_f   []blas64.Vector
	_P_f   []blas64.Symmetric
	_m_s   []blas64.Vector
	_P_s   []blas64.Symmetric
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
		_A:     make([]blas64.General, 0, 10),
		_Q:     make([]blas64.Symmetric, 0, 10),
		_m_p:   make([]blas64.Vector, 0, 10),
		_P_p:   make([]blas64.Symmetric, 0, 10),
		_U:     make([]blas64.Triangular, 0, 10),
		_m_f:   make([]blas64.Vector, 0, 10),
		_P_f:   make([]blas64.Symmetric, 0, 10),
		_m_s:   make([]blas64.Vector, 0, 10),
		_P_s:   make([]blas64.Symmetric, 0, 10),
	}
}

func (p *Process) AddSample(time float64) *Sample {
	idx := len(p.Ts)
	if idx > 0 && time < p.Ts[idx-1] {
		panic(ErrNotChronological)
	}
	m := p.kernel.Order()
	p.Ts = append(p.Ts, time)
	p.Ns = append(p.Ns, 0.0)
	p.Xs = append(p.Xs, 0.0)
	p._m_p = append(p._m_p, p.kernel.StateMean(time))
	p._P_p = append(p._P_p, p.kernel.StateCov(time))
	p._U = append(p._U, blas64.Triangular{
		N:      m,
		Stride: m,
		Data:   make([]float64, m*m),
		Uplo:   blas.Upper,
		Diag:   blas.NonUnit,
	})
	p._m_f = append(p._m_f, p.kernel.StateMean(time))
	p._P_f = append(p._P_f, p.kernel.StateCov(time))
	p._m_s = append(p._m_s, p.kernel.StateMean(time))
	p._P_s = append(p._P_s, p.kernel.StateCov(time))

	// Initialize mean and variance.
	//     m = dot(h, state_mean(t))
	//     v = dot(np.dot(h, state_cov(t)), h)
	h := p.kernel.MeasurementVec()
	tmp := blas64.Vector{Inc: 1, Data: make([]float64, m)}
	blas64.Symv(1.0, p.kernel.StateCov(time), h, 0.0, tmp)
	p.Ms = append(p.Ms, blas64.Dot(m, h, p.kernel.StateMean(time)))
	p.Vs = append(p.Vs, blas64.Dot(m, h, tmp))

	// Compute transition and noise covariance matrices.
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
		U   = p._U
		A   = p._A
		Q   = p._Q
		m_p = p._m_p
		P_p = p._P_p
		m_f = p._m_f
		P_f = p._P_f
		m_s = p._m_s
		P_s = p._P_s
	)
	m := p.kernel.Order()
	var eye = utils.Eye(m)
	// Temporary variables.
	var alpha float64
	vec1 := blas64.Vector{
		Inc:  1,
		Data: make([]float64, m),
	}
	gen1 := blas64.General{
		Rows:   m,
		Cols:   m,
		Stride: m,
		Data:   make([]float64, m*m),
	}
	sym1 := blas64.Symmetric{
		N:      m,
		Stride: m,
		Data:   make([]float64, m*m),
		Uplo:   blas.Upper,
	}
	G := blas64.General{
		Rows:   m,
		Cols:   m,
		Stride: m,
		Data:   make([]float64, m*m),
	}
	SymAsGen := blas64.General{
		Rows:   m,
		Cols:   m,
		Stride: m,
		Data:   make([]float64, m*m),
	}
	k := blas64.Vector{Inc: 1, Data: make([]float64, m)}

	// Forward pass (Kalman filter).
	for i := 0; i < len(ts); i++ {
		if i > 0 {
			// m_p[i] = dot(A[i-1], m_f[i-1])
			blas64.Gemv(blas.NoTrans, 1.0, A[i-1], m_f[i-1], 0.0, m_p[i])

			// P_p[i] = dot(dot(A[i-1], P_f[i-1]), A[i-1].T) + Q[i-1]
			blas64.Symm(blas.Right, 1.0, P_f[i-1], A[i-1], 0.0, gen1)
			SymAsGen.Data = P_p[i].Data
			copy(SymAsGen.Data, Q[i-1].Data) // TODO Dangerous, assumes, P_f.Uplo == All.
			blas64.Gemm(blas.NoTrans, blas.Trans, 1.0, gen1, A[i-1], 1.0, SymAsGen)
		}

		// U[i] = cholesky(P_p[i]) (upper triangular)
		copy(sym1.Data, P_p[i].Data)
		lapack64.Potrf(sym1)
		copy(U[i].Data, sym1.Data)

		// k = dot(P_p[i], h) / (1 + xs[i] * dot(dot(h, P_p[i]), h))
		blas64.Symv(1.0, P_p[i], h, 0.0, k)
		alpha = 1.0 / (1.0 + xs[i]*blas64.Dot(m, h, k))
		blas64.Scal(m, alpha, k)

		// m_f[i] = m_p[i] + k * (ns[i] - xs[i] * dot(h, m_p[i]))
		blas64.Copy(m, m_p[i], m_f[i])
		alpha = ns[i] - xs[i]*blas64.Dot(m, h, m_p[i])
		blas64.Axpy(m, alpha, k, m_f[i])

		// Z = I - xs[i] * np.outer(k, h)
		copy(gen1.Data, eye.Data)
		blas64.Ger(-xs[i], k, h, gen1)

		// Z = dot(Z, U[i].T)
		blas64.Trmm(blas.Right, blas.Trans, 1.0, U[i], gen1)

		// P_f[i] = dot(dot(Z, Z.T) + xs[i] * outer(k, k)
		blas64.Syrk(blas.NoTrans, 1.0, gen1, 0.0, P_f[i])
		blas64.Syr(xs[i], k, P_f[i])
	}

	// Backward pass (RTS smoother).
	for i := len(ts) - 1; i >= 0; i-- {
		if i == len(ts)-1 {
			copy(m_s[i].Data, m_f[i].Data)
			copy(P_s[i].Data, P_f[i].Data)
		} else {
			// G = (dot(A[i], P_f[i]) \ U[i+1].T) \ U[i+1]
			blas64.Symm(blas.Right, 1.0, P_f[i], A[i], 0.0, G)
			lapack64.Trtrs(blas.Trans, U[i+1], G)
			lapack64.Trtrs(blas.NoTrans, U[i+1], G)

			// m_s[i] = m_f[i] + dot(G.T, m_s[i+1] - m_p[i+1])
			blas64.Copy(m, m_s[i+1], vec1)
			blas64.Axpy(m, -1.0, m_p[i+1], vec1)
			blas64.Copy(m, m_f[i], m_s[i])
			blas64.Gemv(blas.Trans, 1.0, G, vec1, 1.0, m_s[i])

			// sym1 = P_s[i+1] - P_p[i+1]
			for k := 0; k < sym1.N; k++ {
				for j := k; j < sym1.N; j++ {
					sym1.Data[k*sym1.Stride+j] = P_s[i+1].Data[k*P_s[i+1].Stride+j] -
						P_p[i+1].Data[k*P_p[i+1].Stride+j]
				}
			}

			// P_s[i] = P_f[i] + dot(G.T, dot(sym1, G))
			blas64.Symm(blas.Left, 1.0, sym1, G, 0.0, gen1)
			SymAsGen.Data = P_s[i].Data
			copy(SymAsGen.Data, P_f[i].Data) // TODO Dangerous, assumes, P_f.Uplo == All.
			blas64.Gemm(blas.Trans, blas.NoTrans, 1.0, G, gen1, 1.0, SymAsGen)
		}

		// ms[i] = dot(h, m_s[i]),  vs[i] = dot(np.dot(h, P_s[i]), h)
		ms[i] = blas64.Dot(m, h, m_s[i])
		blas64.Symv(1.0, P_s[i], h, 0.0, vec1)
		vs[i] = blas64.Dot(m, h, vec1)
	}
}

func (p *Process) Predict(ts []float64) (ms, vs []float64) {
	return
}
