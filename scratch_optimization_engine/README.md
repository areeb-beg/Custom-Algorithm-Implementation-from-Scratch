# Scratch Optimization Engine: Dependency-Free Numerical Optimization & Machine Learning

A research-grade numerical optimization, machine learning, and trajectory planning engine implemented entirely from mathematical first principles in Python using **only `numpy` and `matplotlib`**. All high-level frameworks (`scikit-learn`, `scipy.optimize`, `pytorch`, `tensorflow`) are strictly excluded.

---

## 1. System Architecture & Directory Tree

```
/scratch_optimization_engine
├── core/
│   ├── __init__.py            # Library exports and unified API
│   ├── optimizers.py          # Scratch implementations of SGD (Momentum/Nesterov), RMSProp, and Adam
│   ├── regression.py          # Multivariate Polynomial Regression with Closed-Form (Cholesky), GD (ISTA), and Coordinate Descent
│   └── path_planner.py        # A* State-Space Graph Search, Coordinate Gradient Trajectory Smoothing, and Catmull-Rom Splines
├── benchmarks/
│   ├── compare_convergence.py # Empirical convergence, wall-clock timing, and tracemalloc memory profiling
│   └── visualize_path.py      # 2D obstacle grid, A* search, smoothed spline trajectory, and curvature profiling
├── tests/
│   └── test_gradients.py      # Finite-difference gradient checking suite verified within 1e-7 tolerance
└── README.md                  # Comprehensive mathematical proofs, matrix calculus derivations, and execution guide
```

---

## 2. Complete Mathematical Proofs & Matrix Calculus Derivations

### 2.1 Matrix Calculus Conventions & Foundational Identities

We adopt the **numerator layout convention** (Jacobian formulation):
For a scalar field $f: \mathbb{R}^d \to \mathbb{R}$ and column vector $x \in \mathbb{R}^d$:
$$\nabla_x f(x) = \left[ \frac{\partial f}{\partial x_1}, \frac{\partial f}{\partial x_2}, \dots, \frac{\partial f}{\partial x_d} \right]^T \in \mathbb{R}^d$$

#### Key Identities:
1. Linear form:
   $$\nabla_x (a^T x) = \nabla_x (x^T a) = a, \quad a \in \mathbb{R}^d$$
2. Quadratic form:
   $$\nabla_x (x^T A x) = (A + A^T) x$$
   If $A$ is symmetric ($A = A^T$), then $\nabla_x (x^T A x) = 2 A x$.
3. Squared Euclidean residual:
   $$\nabla_x \|A x - b\|_2^2 = 2 A^T (A x - b)$$

---

### 2.2 Multivariate Polynomial Regression & Regularization

#### 2.2.1 Multivariate Polynomial Feature Map
Let $X \in \mathbb{R}^{n \times d}$ denote the raw feature matrix with $n$ observations and $d$ variables. A multivariate polynomial feature map of degree $p$ constructs the basis tensor:
$$\Phi(X) = \left[ \phi(x^{(1)}), \phi(x^{(2)}), \dots, \phi(x^{(n)}) \right]^T \in \mathbb{R}^{n \times D}$$
where each row $\phi(x) \in \mathbb{R}^D$ is the vector of all monomial terms:
$$\prod_{j=1}^d x_j^{k_j} \quad \text{such that} \quad \sum_{j=1}^d k_j \le p, \quad k_j \in \mathbb{N}_0$$
The expanded dimensionality is given combinatorially by:
$$D = \binom{d + p}{p}$$

#### 2.2.2 Regularized Loss Formulation
The objective function with Elastic Net regularization is:
$$J(\theta) = \frac{1}{2n} \|\Phi \theta - y\|_2^2 + \alpha \left[ \rho \|\theta_{1:}\|_1 + \frac{1 - \rho}{2} \|\theta_{1:}\|_2^2 \right]$$
where:
- $\theta = [\theta_0, \theta_1, \dots, \theta_{D-1}]^T \in \mathbb{R}^D$,
- $\theta_0$ is the unpenalized intercept parameter,
- $\alpha \ge 0$ is the global regularization scale,
- $\rho \in [0, 1]$ is the Elastic Net mixing ratio ($\rho = 0 \implies \text{Ridge } L_2$, $\rho = 1 \implies \text{Lasso } L_1$).

#### 2.2.3 Derivation of the Normal Equation (Closed-Form Ridge)
For $\rho = 0$ (pure Ridge), the loss is smooth and strictly convex:
$$J_{\text{Ridge}}(\theta) = \frac{1}{2n} (\Phi \theta - y)^T (\Phi \theta - y) + \frac{\alpha}{2} \theta^T I' \theta$$
where $I' = \text{diag}(0, 1, 1, \dots, 1) \in \mathbb{R}^{D \times D}$ leaves $\theta_0$ unpenalized.

Expanding the quadratic objective:
$$J_{\text{Ridge}}(\theta) = \frac{1}{2n} \left( \theta^T \Phi^T \Phi \theta - 2 y^T \Phi \theta + y^T y \right) + \frac{\alpha}{2} \theta^T I' \theta$$

Differentiating with respect to the parameter vector $\theta$:
$$\nabla_\theta J_{\text{Ridge}}(\theta) = \frac{1}{n} \Phi^T (\Phi \theta - y) + \alpha I' \theta = \left( \frac{1}{n} \Phi^T \Phi + \alpha I' \right) \theta - \frac{1}{n} \Phi^T y$$

Setting the gradient to zero for first-order optimality $\nabla_\theta J_{\text{Ridge}}(\theta^*) = 0$:
$$\left( \frac{1}{n} \Phi^T \Phi + \alpha I' \right) \theta^* = \frac{1}{n} \Phi^T y \iff (\Phi^T \Phi + n \alpha I') \theta^* = \Phi^T y$$

Solving for $\theta^*$:
$$\theta^* = (\Phi^T \Phi + n \alpha I')^{-1} \Phi^T y$$

#### Proof of Invertibility & Conditioning:
The Gram matrix $\Phi^T \Phi$ is symmetric positive semi-definite ($\lambda_{\min}(\Phi^T \Phi) \ge 0$). When features exhibit multicollinearity, $\Phi^T \Phi$ is ill-conditioned or singular ($\det(\Phi^T \Phi) \approx 0$). For $\alpha > 0$, the regularized matrix satisfies:
$$\lambda_{\min}(\Phi^T \Phi + n \alpha I') \ge \min\left( \lambda_{\min}(\Phi_0^T \Phi_0), n \alpha \right) > 0$$
ensuring that the operator is strictly positive definite and invertible with condition number bounded by $\kappa \le \frac{\sigma_{\max}^2 + n\alpha}{n\alpha}$.

#### 2.2.4 Cholesky Decomposition Linear Solver
To solve $A \theta = b$ where $A = \frac{1}{n}\Phi^T \Phi + \alpha I'$:
1. Compute Cholesky factorization: $A = L L^T$, where $L$ is a lower-triangular matrix with strictly positive diagonal entries:
   $$L_{j,j} = \sqrt{A_{j,j} - \sum_{k=1}^{j-1} L_{j,k}^2}, \quad L_{i,j} = \frac{1}{L_{j,j}} \left( A_{i,j} - \sum_{k=1}^{j-1} L_{i,k} L_{j,k} \right) \quad (i > j)$$
2. Forward substitution ($L z = b$):
   $$z_i = \frac{1}{L_{i,i}} \left( b_i - \sum_{k=1}^{i-1} L_{i,k} z_k \right), \quad i = 1, \dots, D$$
3. Backward substitution ($L^T \theta = z$):
   $$\theta_i = \frac{1}{L_{i,i}} \left( z_i - \sum_{k=i+1}^D L_{k,i} \theta_k \right), \quad i = D, D-1, \dots, 1$$
**Complexity:** $\frac{1}{3} D^3 + 2 D^2$ FLOPs, requiring half the operations of general Gaussian elimination ($LU$).

#### 2.2.5 Proximal Gradient Descent (ISTA) for Lasso ($L_1$)
For $\rho > 0$, the $L_1$ norm is non-smooth at the origin. We split the objective into smooth and non-smooth components:
$$J(\theta) = f(\theta) + g(\theta), \quad f(\theta) = \frac{1}{2n}\|\Phi \theta - y\|_2^2 + \frac{\alpha(1-\rho)}{2}\|\theta_{1:}\|_2^2, \quad g(\theta) = \alpha \rho \|\theta_{1:}\|_1$$
The proximal operator for $g$ scaled by step size $\eta$ is:
$$\text{prox}_{\eta g}(v) = \arg\min_u \left( \frac{1}{2\eta} \|u - v\|_2^2 + g(u) \right)$$
Separating across individual coordinates $j \ge 1$:
$$\arg\min_{u_j} \left( \frac{1}{2\eta} (u_j - v_j)^2 + \alpha \rho |u_j| \right)$$
Checking subgradient optimality $0 \in \frac{1}{\eta}(u_j - v_j) + \alpha \rho \partial |u_j|$:
- If $u_j > 0$: $\frac{1}{\eta}(u_j - v_j) + \alpha \rho = 0 \implies u_j = v_j - \eta \alpha \rho$ (valid when $v_j > \eta \alpha \rho$).
- If $u_j < 0$: $\frac{1}{\eta}(u_j - v_j) - \alpha \rho = 0 \implies u_j = v_j + \eta \alpha \rho$ (valid when $v_j < -\eta \alpha \rho$).
- If $u_j = 0$: $|v_j| \le \eta \alpha \rho$.

This yields the **Soft-Thresholding Operator**:
$$\mathcal{S}_{\lambda}(v) = \text{sign}(v) \max(|v| - \lambda, 0), \quad \lambda = \eta \alpha \rho$$

---

### 2.3 First-Order Optimization Mechanics

#### 2.3.1 Polyak Heavy-Ball Momentum
Classical momentum models the physical equation of a particle with mass in a potential field with viscous friction:
$$m \ddot{\theta}(t) + b \dot{\theta}(t) + \nabla J(\theta(t)) = 0$$
Discretizing via backward Euler on velocity and forward Euler on position:
$$v_t = \mu v_{t-1} + g_t, \quad \theta_t = \theta_{t-1} - \eta v_t$$
where $\mu = 1 - \frac{b \Delta t}{m} \in [0, 1)$ dampens high-frequency oscillations across ravines with high condition number $\kappa = \frac{\lambda_{\max}(H)}{\lambda_{\min}(H)}$, accelerating the asymptotic convergence rate from $\frac{\kappa - 1}{\kappa + 1}$ to $\frac{\sqrt{\kappa} - 1}{\sqrt{\kappa} + 1}$.

#### 2.3.2 RMSProp (Root Mean Square Propagation)
RMSProp resolves gradient scale disparity across heterogeneous coordinate directions by maintaining an exponentially decaying average of squared gradients:
$$v_t = \alpha v_{t-1} + (1 - \alpha) g_t^2, \quad \theta_t = \theta_{t-1} - \frac{\eta}{\sqrt{v_t} + \epsilon} \odot g_t$$
This ensures that coordinates with persistently large gradients take dampened steps, while coordinates with sparse or gentle slopes take accelerated steps.

#### 2.3.3 Adam: Unbiased Moment Estimator Derivations
Adam computes exponentially weighted moving averages of first moments $m_t$ and second uncentered raw moments $v_t$:
$$m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t = (1 - \beta_1) \sum_{i=1}^t \beta_1^{t-i} g_i$$
$$v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2 = (1 - \beta_2) \sum_{i=1}^t \beta_2^{t-i} g_i^2$$

#### Proof of Bias Correction:
Taking mathematical expectations of both sides under the assumption that the true gradients $g_i$ are drawn from a stationary distribution with mean $\mathbb{E}[g_i] = \mathbb{E}[g]$:
$$\mathbb{E}[m_t] = \mathbb{E}\left[ (1 - \beta_1) \sum_{i=1}^t \beta_1^{t-i} g_i \right] = \mathbb{E}[g] \cdot (1 - \beta_1) \sum_{i=1}^t \beta_1^{t-i}$$
Recognizing the finite geometric progression sum:
$$\sum_{i=1}^t \beta_1^{t-i} = \sum_{k=0}^{t-1} \beta_1^k = \frac{1 - \beta_1^t}{1 - \beta_1}$$
Substituting this identity:
$$\mathbb{E}[m_t] = \mathbb{E}[g] \cdot (1 - \beta_1) \cdot \frac{1 - \beta_1^t}{1 - \beta_1} = \mathbb{E}[g] \cdot (1 - \beta_1^t)$$
Therefore, the unbiased estimator $\hat{m}_t$ satisfies $\mathbb{E}[\hat{m}_t] = \mathbb{E}[g]$:
$$\hat{m}_t = \frac{m_t}{1 - \beta_1^t}$$

Similarly, for the second moment $v_t$:
$$\mathbb{E}[v_t] = \mathbb{E}[g^2] \cdot (1 - \beta_2^t) \implies \hat{v}_t = \frac{v_t}{1 - \beta_2^t}$$

Parameter update:
$$\theta_t = \theta_{t-1} - \frac{\eta}{\sqrt{\hat{v}_t} + \epsilon} \odot \hat{m}_t$$

---

### 2.4 Discrete Search & Continuous Trajectory Optimization

#### 2.4.1 A* Admissibility & Consistency
Let $h(n)$ be the heuristic estimate from node $n$ to goal $g$.
**Octile Heuristic**:
$$h(n) = \Delta x + \Delta y + (\sqrt{2} - 2) \min(\Delta x, \Delta y)$$
For any adjacent nodes $n, n'$ connected by edge cost $c(n, n')$:
$$h(n) \le c(n, n') + h(n')$$
satisfying monotonic consistency (triangle inequality), which guarantees that the first time any node is popped from the priority queue, its path $g(n)$ is globally optimal.

#### 2.4.2 Variational Trajectory Smoothing Functional
Let a continuous 2D planar curve be parameterized by arc length or index: $p(t) = [x(t), y(t)]^T \in \mathbb{R}^2$ for $t \in [0, T]$. We minimize the variational action functional:
$$\mathcal{J}[p] = \int_0^T \left( \frac{w_{\text{smooth}}}{2} \|\ddot{p}(t)\|_2^2 + \frac{w_{\text{curv}}}{2} \|\dot{p}(t)\|_2^2 + w_{\text{obs}} U_{\text{obs}}(p(t)) \right) dt$$
subject to fixed boundary conditions $p(0) = p_{\text{start}}$ and $p(T) = p_{\text{goal}}$.

#### Discrete Biharmonic Acceleration Operator:
Discretizing with $N+1$ waypoints $P = [p_0, p_1, \dots, p_N]^T$:
$$J_{\text{smooth}}(P) = \frac{1}{2} \sum_{i=1}^{N-1} \|p_{i+1} - 2p_i + p_{i-1}\|_2^2$$
Let $a_i = p_{i+1} - 2p_i + p_{i-1}$. For interior waypoint $p_i$ ($1 \le i \le N-1$):
$$\nabla_{p_i} J_{\text{smooth}} = a_{i-1} - 2a_i + a_{i+1} = p_{i-2} - 4 p_{i-1} + 6 p_i - 4 p_{i+1} + p_{i+2}$$
This is the discrete biharmonic fourth-order spatial differential operator $\Delta^2 p_i$.

#### Discrete Curvature / Tension Operator:
$$J_{\text{curv}}(P) = \frac{1}{2} \sum_{i=0}^{N-1} \|p_{i+1} - p_i\|_2^2 \implies \nabla_{p_i} J_{\text{curv}} = 2p_i - p_{i-1} - p_{i+1} = -\Delta p_i$$

#### Obstacle Potential Field Gradient:
Let $d(p_i) = \min_{o \in \mathcal{O}} \|p_i - o\|_2$ and $o^*(p_i)$ be the nearest obstacle boundary point. If $d(p_i) < d_{\text{safe}}$:
$$U_{\text{obs}}(p_i) = \frac{1}{2} \left( \frac{1}{d(p_i)} - \frac{1}{d_{\text{safe}}} \right)^2$$
Applying the multivariate chain rule:
$$\nabla_{p_i} d(p_i) = \frac{p_i - o^*(p_i)}{\|p_i - o^*(p_i)\|_2} = \frac{p_i - o^*(p_i)}{d(p_i)}$$
$$\nabla_{p_i} U_{\text{obs}}(p_i) = \left( \frac{1}{d(p_i)} - \frac{1}{d_{\text{safe}}} \right) \left( -\frac{1}{d(p_i)^2} \right) \nabla_{p_i} d(p_i) = -\left( \frac{1}{d(p_i)} - \frac{1}{d_{\text{safe}}} \right) \frac{p_i - o^*(p_i)}{d(p_i)^3}$$

#### 2.4.3 Catmull-Rom Spline Curvature Formulation
For segment $i$ with parameter $u \in [0, 1]$ and tangents $T_i = \frac{1}{2}(P_{i+1} - P_{i-1})$:
$$C(u) = (2u^3 - 3u^2 + 1) P_i + (u^3 - 2u^2 + u) T_i + (-2u^3 + 3u^2) P_{i+1} + (u^3 - u^2) T_{i+1}$$
The analytical curvature of the parameterized curve is:
$$\kappa(u) = \frac{\|\dot{C}(u) \times \ddot{C}(u)\|}{\|\dot{C}(u)\|_2^3} = \frac{|\dot{x}(u) \ddot{y}(u) - \dot{y}(u) \ddot{x}(u)|}{\left( \dot{x}(u)^2 + \dot{y}(u)^2 \right)^{3/2}}$$

---

### 2.5 Finite-Difference Numerical Verification Theory

To verify analytical gradient expressions without external autograd engines, we compute symmetric difference quotients:
$$[\nabla_{\text{num}} f(x)]_i = \frac{f(x + \epsilon e_i) - f(x - \epsilon e_i)}{2\epsilon}$$

#### Taylor Expansion Truncation Error Proof:
Expanding $f(x \pm \epsilon e_i)$ around $x$ via third-order Taylor series:
$$f(x + \epsilon e_i) = f(x) + \epsilon \frac{\partial f}{\partial x_i} + \frac{\epsilon^2}{2} \frac{\partial^2 f}{\partial x_i^2} + \frac{\epsilon^3}{6} \frac{\partial^3 f}{\partial x_i^3} + O(\epsilon^4)$$
$$f(x - \epsilon e_i) = f(x) - \epsilon \frac{\partial f}{\partial x_i} + \frac{\epsilon^2}{2} \frac{\partial^2 f}{\partial x_i^2} - \frac{\epsilon^3}{6} \frac{\partial^3 f}{\partial x_i^3} + O(\epsilon^4)$$
Subtracting the two equations cancels the quadratic even terms:
$$f(x + \epsilon e_i) - f(x - \epsilon e_i) = 2\epsilon \frac{\partial f}{\partial x_i} + \frac{\epsilon^3}{3} \frac{\partial^3 f}{\partial x_i^3} + O(\epsilon^5)$$
Dividing by $2\epsilon$:
$$\frac{f(x + \epsilon e_i) - f(x - \epsilon e_i)}{2\epsilon} = \frac{\partial f}{\partial x_i} + \frac{\epsilon^2}{6} \frac{\partial^3 f}{\partial x_i^3} + O(\epsilon^4) = \frac{\partial f}{\partial x_i} + O(\epsilon^2)$$

#### Optimal Perturbation Step Size ($\epsilon^*$):
In IEEE 754 double precision floating-point arithmetic ($\epsilon_{\text{mach}} \approx 2.22 \times 10^{-16}$), total numerical error is the sum of truncation error and roundoff cancellation error:
$$E_{\text{total}}(\epsilon) \approx \frac{\epsilon^2}{6} M_3 + \frac{\epsilon_{\text{mach}} M_0}{\epsilon}$$
Setting $\frac{d E_{\text{total}}}{d \epsilon} = 0$:
$$\frac{\epsilon}{3} M_3 - \frac{\epsilon_{\text{mach}} M_0}{\epsilon^2} = 0 \implies \epsilon^* = \left( \frac{3 \epsilon_{\text{mach}} M_0}{M_3} \right)^{1/3} \approx 10^{-6}$$
With $\epsilon = 10^{-6}$, relative error satisfies:
$$\text{rel\_error} = \frac{\|\nabla_{\text{analytical}} - \nabla_{\text{num}}\|_2}{\max(\|\nabla_{\text{analytical}}\|_2 + \|\nabla_{\text{num}}\|_2, 10^{-15})} < 10^{-7}$$

---

## 3. Mathematical Complexity & Resource Analysis

| Component | Sub-Algorithm | Time Complexity | Auxiliary Space Complexity | Dominant Mathematical Operation |
| :--- | :--- | :--- | :--- | :--- |
| **Optimizers** | SGD (Vanilla) | $O(d)$ | $O(1)$ | Vector scalar-multiply & subtract |
| | SGD (Momentum) | $O(d)$ | $O(d)$ | Velocity recurrence buffer |
| | RMSProp | $O(d)$ | $O(d)$ | Exponentially weighted square average |
| | Adam | $O(d)$ | $2 \cdot O(d)$ | Moment updates & bias-correction quotients |
| **Regression** | Polynomial Expansion | $O(n \cdot D)$ | $O(n \cdot D)$ | Vectorized monomial column product ($D = \binom{d+p}{p}$) |
| | Normal Eq (Cholesky) | $O(n D^2 + \frac{1}{3} D^3)$ | $O(D^2)$ | Gram matrix multiplication $\Phi^T \Phi$ & triangular solve |
| | Gradient Descent (ISTA)| $O(K \cdot n D)$ | $O(D)$ | Matrix-vector multiplication & soft-thresholding |
| | Coordinate Descent | $O(K \cdot n D)$ | $O(D)$ | Cyclical coordinate correlation updates |
| **Path Planning** | A* Graph Search | $O(\|V\| \log \|V\| + \|E\|)$ | $O(\|V\|)$ | Min-heap priority queue operations ($\|V\| = H \times W$) |
| | Trajectory Smoothing | $O(K \cdot N \cdot M)$ | $O(N)$ | Biharmonic finite difference & obstacle field gradients |
| | Catmull-Rom Spline | $O(M_{\text{dense}})$ | $O(M_{\text{dense}})$ | Cubic Hermite basis function evaluations |
| **Verification** | Finite Differences | $2 \cdot d \cdot O(f)$ | $O(d)$ | Central difference objective evaluations |

---

## 4. Execution & Verification Guide

Depending on your terminal's current working directory, run the commands below:

### 4.1 When your terminal is inside `scratch_optimization_engine/`:
```bash
# 1. Run unit test suite verifying analytical gradients to < 1e-7 tolerance
python -m unittest tests.test_gradients -v
# (or directly: python tests/test_gradients.py)

# 2. Run convergence, timing, and memory profiling benchmark
python benchmarks/compare_convergence.py

# 3. Run 2D path planning and trajectory smoothing benchmark
python benchmarks/visualize_path.py
```

### 4.2 When your terminal is in the project parent directory:
```bash
# 1. Run unit test suite
python -m unittest scratch_optimization_engine.tests.test_gradients -v

# 2. Run convergence benchmark
python scratch_optimization_engine/benchmarks/compare_convergence.py

# 3. Run path planning benchmark
python scratch_optimization_engine/benchmarks/visualize_path.py
```
