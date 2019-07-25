package score

import (
	"github.com/lucasmaystre/gokick/kern"
	"github.com/lucasmaystre/gokick/utils"
	"errors"
	"gonum.org/v1/gonum/blas"
	"gonum.org/v1/gonum/blas/blas64"
	"gonum.org/v1/gonum/lapack/lapack64"
	"math"
	"sort"
)

var ErrNotChronological = errors.New("observation not in chronological order")
var ErrNotImplemented = errors.New("not yet implemented")

type Sample struct {
	process *Process
	idx     int
}

func (s *Sample) CavityParams() (xCav, nCav float64) {
	xTot := 1.0 / s.process.vs[s.idx]
	nTot := xTot * s.process.ms[s.idx]
	xCav = xTot - s.process.xs[s.idx]
	nCav = nTot - s.process.ns[s.idx]
	return
}

func (s *Sample) PseudoObsParams() (float64, float64) {
	return s.process.xs[s.idx], s.process.ns[s.idx]
}

func (s *Sample) UpdatePseudoObs(x, n, damping float64) {
	s.process.xs[s.idx] = (1-damping)*s.process.xs[s.idx] + damping*x
	s.process.ns[s.idx] = (1-damping)*s.process.ns[s.idx] + damping*n
}

type Process struct {
	kernel kern.Kernel
	ts     []float64 // Samples' times.
	ms     []float64 // Samples' means.
	vs     []float64 // Samples' variances.
	ns     []float64 // Pseudo-observations' precision-adjusted means.
	xs     []float64 // Pseudo-observations' precisions.

	// State-space model variables.
	vecH   blas64.Vector       // Measurement vector.
	matsA  []blas64.General    // Transition matrices.
	matsQ  []blas64.Symmetric  // Noise covariance matrices.
	vecsMp []blas64.Vector     // Predictive means.
	matsPp []blas64.Symmetric  // Predictive covariances.
	matsU  []blas64.Triangular // Cholesky factors of predictive variances
	vecsMf []blas64.Vector     // Filtering means.
	matsPf []blas64.Symmetric  // Filtering covariances.
	vecsMs []blas64.Vector     // Smoothing means.
	matsPs []blas64.Symmetric  // Smoothing covariances.
}

func NewProcess(kernel kern.Kernel) *Process {
	return &Process{
		kernel: kernel,
		ts:     make([]float64, 0, 10),
		ms:     make([]float64, 0, 10),
		vs:     make([]float64, 0, 10),
		ns:     make([]float64, 0, 10),
		xs:     make([]float64, 0, 10),
		vecH:   kernel.MeasurementVec(),
		matsA:  make([]blas64.General, 0, 10),
		matsQ:  make([]blas64.Symmetric, 0, 10),
		vecsMp: make([]blas64.Vector, 0, 10),
		matsPp: make([]blas64.Symmetric, 0, 10),
		matsU:  make([]blas64.Triangular, 0, 10),
		vecsMf: make([]blas64.Vector, 0, 10),
		matsPf: make([]blas64.Symmetric, 0, 10),
		vecsMs: make([]blas64.Vector, 0, 10),
		matsPs: make([]blas64.Symmetric, 0, 10),
	}
}

func (p *Process) AddSample(time float64) *Sample {
	idx := len(p.ts)
	if idx > 0 && time < p.ts[idx-1] {
		panic(ErrNotChronological)
	}
	m := p.kernel.Order()
	p.ts = append(p.ts, time)
	p.ns = append(p.ns, 0.0)
	p.xs = append(p.xs, 0.0)
	p.vecsMp = append(p.vecsMp, p.kernel.StateMean(time))
	p.matsPp = append(p.matsPp, p.kernel.StateCov(time))
	p.matsU = append(p.matsU, blas64.Triangular{
		N:      m,
		Stride: m,
		Data:   make([]float64, m*m),
		Uplo:   blas.Upper,
		Diag:   blas.NonUnit,
	})
	p.vecsMf = append(p.vecsMf, p.kernel.StateMean(time))
	p.matsPf = append(p.matsPf, p.kernel.StateCov(time))
	p.vecsMs = append(p.vecsMs, p.kernel.StateMean(time))
	p.matsPs = append(p.matsPs, p.kernel.StateCov(time))

	// Initialize mean and variance.
	//     m = dot(h, state_mean(t))
	//     v = dot(np.dot(h, state_cov(t)), h)
	tmp := blas64.Vector{Inc: 1, Data: make([]float64, m)}
	blas64.Symv(1.0, p.kernel.StateCov(time), p.vecH, 0.0, tmp)
	p.ms = append(p.ms, blas64.Dot(m, p.vecH, p.kernel.StateMean(time)))
	p.vs = append(p.vs, blas64.Dot(m, p.vecH, tmp))

	// Compute transition and noise covariance matrices.
	if idx > 0 {
		delta := time - p.ts[idx-1]
		p.matsA = append(p.matsA, p.kernel.Transition(delta))
		p.matsQ = append(p.matsQ, p.kernel.NoiseCov(delta))
	}
	return &Sample{
		process: p,
		idx:     idx,
	}
}

// Kalman filter (forward pass).
func (p *Process) filter() {
	m := p.kernel.Order()
	var eye = utils.Eye(m)

	// Temporary variables.
	var alpha float64
	k := blas64.Vector{Inc: 1, Data: make([]float64, m)}
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
	symAsGen := blas64.General{
		Rows:   m,
		Cols:   m,
		Stride: m,
		Data:   make([]float64, m*m),
	}

	for i := 0; i < len(p.ts); i++ {
		if i > 0 {
			// m_p[i] = dot(A[i-1], m_f[i-1])
			blas64.Gemv(blas.NoTrans,
				1.0, p.matsA[i-1], p.vecsMf[i-1], 0.0, p.vecsMp[i])

			// P_p[i] = dot(dot(A[i-1], P_f[i-1]), A[i-1].T) + Q[i-1]
			blas64.Symm(blas.Right, 1.0, p.matsPf[i-1], p.matsA[i-1], 0.0, gen1)
			symAsGen.Data = p.matsPp[i].Data
			copy(symAsGen.Data, p.matsQ[i-1].Data) // TODO Dangerous, assumes, P_f.Uplo == All.
			blas64.Gemm(blas.NoTrans, blas.Trans,
				1.0, gen1, p.matsA[i-1], 1.0, symAsGen)
		}

		// U[i] = cholesky(P_p[i]) (upper triangular)
		copy(sym1.Data, p.matsPp[i].Data)
		lapack64.Potrf(sym1)
		copy(p.matsU[i].Data, sym1.Data)

		// k = dot(P_p[i], h) / (1 + xs[i] * dot(dot(h, P_p[i]), h))
		blas64.Symv(1.0, p.matsPp[i], p.vecH, 0.0, k)
		alpha = 1.0 / (1.0 + p.xs[i]*blas64.Dot(m, p.vecH, k))
		blas64.Scal(m, alpha, k)

		// m_f[i] = m_p[i] + k * (ns[i] - xs[i] * dot(h, m_p[i]))
		blas64.Copy(m, p.vecsMp[i], p.vecsMf[i])
		alpha = p.ns[i] - p.xs[i]*blas64.Dot(m, p.vecH, p.vecsMp[i])
		blas64.Axpy(m, alpha, k, p.vecsMf[i])

		// Z = I - xs[i] * np.outer(k, h)
		copy(gen1.Data, eye.Data)
		blas64.Ger(-p.xs[i], k, p.vecH, gen1)

		// Z = dot(Z, U[i].T)
		blas64.Trmm(blas.Right, blas.Trans, 1.0, p.matsU[i], gen1)

		// P_f[i] = dot(dot(Z, Z.T) + xs[i] * outer(k, k)
		blas64.Syrk(blas.NoTrans, 1.0, gen1, 0.0, p.matsPf[i])
		blas64.Syr(p.xs[i], k, p.matsPf[i])
	}
}

// Rauch-Tung-Striebel smoother (backward pass).
func (p *Process) smooth() {
	m := p.kernel.Order()
	// Temporary variables.
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
	symAsGen := blas64.General{
		Rows:   m,
		Cols:   m,
		Stride: m,
		Data:   make([]float64, m*m),
	}

	for i := len(p.ts) - 1; i >= 0; i-- {
		if i == len(p.ts)-1 {
			copy(p.vecsMs[i].Data, p.vecsMf[i].Data)
			copy(p.matsPs[i].Data, p.matsPf[i].Data)
		} else {
			// G = (dot(A[i], P_f[i]) \ U[i+1].T) \ U[i+1]
			blas64.Symm(blas.Right, 1.0, p.matsPf[i], p.matsA[i], 0.0, G)
			lapack64.Trtrs(blas.Trans, p.matsU[i+1], G)
			lapack64.Trtrs(blas.NoTrans, p.matsU[i+1], G)

			// m_s[i] = m_f[i] + dot(G.T, m_s[i+1] - m_p[i+1])
			blas64.Copy(m, p.vecsMs[i+1], vec1)
			blas64.Axpy(m, -1.0, p.vecsMp[i+1], vec1)
			blas64.Copy(m, p.vecsMf[i], p.vecsMs[i])
			blas64.Gemv(blas.Trans, 1.0, G, vec1, 1.0, p.vecsMs[i])

			// sym1 = P_s[i+1] - P_p[i+1]
			for k := 0; k < sym1.N; k++ {
				for j := k; j < sym1.N; j++ {
					sym1.Data[k*sym1.Stride+j] = p.matsPs[i+1].Data[k*p.matsPs[i+1].Stride+j] -
						p.matsPp[i+1].Data[k*p.matsPp[i+1].Stride+j]
				}
			}

			// P_s[i] = P_f[i] + dot(G.T, dot(sym1, G))
			blas64.Symm(blas.Left, 1.0, sym1, G, 0.0, gen1)
			symAsGen.Data = p.matsPs[i].Data
			copy(symAsGen.Data, p.matsPf[i].Data) // TODO Dangerous, assumes, P_f.Uplo == All.
			blas64.Gemm(blas.Trans, blas.NoTrans, 1.0, G, gen1, 1.0, symAsGen)
		}

		// ms[i] = dot(h, m_s[i]),  vs[i] = dot(np.dot(h, P_s[i]), h)
		p.ms[i] = blas64.Dot(m, p.vecH, p.vecsMs[i])
		blas64.Symv(1.0, p.matsPs[i], p.vecH, 0.0, vec1)
		p.vs[i] = blas64.Dot(m, p.vecH, vec1)
	}
}

func (p *Process) Fit() {
	p.filter()
	p.smooth()
}

func (p *Process) Predict(t float64) (mean, var_ float64) {
	m := p.kernel.Order()
	// Temporary vector.
	vec := blas64.Vector{
		Inc:  1,
		Data: make([]float64, m),
	}
	if len(p.ts) == 0 {
		mean = 0.0
		blas64.Symv(1.0, p.kernel.StateCov(t), p.vecH, 0.0, vec)
		var_ = blas64.Dot(m, p.vecH, vec)
		return
	} else {
		nxt := sort.SearchFloat64s(p.ts, t)
		// Temporary variables.
		matP := blas64.Symmetric{
			N:      m,
			Stride: m,
			Data:   make([]float64, m*m),
			Uplo:   blas.Upper,
		}
		gen := blas64.General{
			Rows:   m,
			Cols:   m,
			Stride: m,
			Data:   make([]float64, m*m),
		}
		symAsGen := blas64.General{
			Rows:   m,
			Cols:   m,
			Stride: m,
			Data:   make([]float64, m*m),
		}

		if nxt == len(p.ts) {
			// New point is *after* last observation.
			delta := t - p.ts[nxt-1]
			matA := p.kernel.Transition(delta)
			matQ := p.kernel.NoiseCov(delta)

			// m = dot(A, m_f[nxt-1])
			blas64.Gemv(blas.NoTrans,
				1.0, matA, p.vecsMf[nxt-1], 0.0, vec)

			// P = dot(dot(A, P_f[nxt-1]), A.T) + Q
			blas64.Symm(blas.Right, 1.0, p.matsPf[nxt-1], matA, 0.0, gen)
			symAsGen.Data = matP.Data
			copy(symAsGen.Data, matQ.Data) // TODO Dangerous, assumes, P_f.Uplo == All.
			blas64.Gemm(blas.NoTrans, blas.Trans,
				1.0, gen, matA, 1.0, symAsGen)

			// m = dot(h, m),  v = dot(np.dot(h, P), h)
			mean = blas64.Dot(m, p.vecH, vec)
			blas64.Symv(1.0, matP, p.vecH, 0.0, vec)
			var_ = blas64.Dot(m, p.vecH, vec)
			return
		} else {
			panic(ErrNotImplemented)
		}
	}
}

func (p *Process) LogLikelihoodContrib() float64 {
	contrib := 0.0
	m := p.kernel.Order()
	// Temporary variables.
	tmp := blas64.Vector{Inc: 1, Data: make([]float64, m)}
	for i := 0; i < len(p.ts); i++ {
		o := blas64.Dot(m, p.vecH, p.vecsMp[i])
		// v = dot(dot(h, P_p[i]), h))
		blas64.Symv(1.0, p.matsPp[i], p.vecH, 0.0, tmp)
		v := blas64.Dot(m, p.vecH, tmp)
		n := p.ns[i]
		x := p.xs[i]
		contrib += -0.5 * (math.Log(x*v+1.0) + (-n*n*v-2*n*o+x*o*o)/(x*v+1))
	}
	return contrib
}
