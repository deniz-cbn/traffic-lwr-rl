# Mathematics of the project

Every formula here is implemented in `src/trafficrl/`; the code location is given in brackets.
GitHub renders the LaTeX. Notation: $\rho$ density [veh/m], $q$ flow [veh/s], $v_f$ free-flow speed,
$\rho_{\max}$ jam density, $\Delta x$ cell length, $\Delta t$ time step.

---

## 1. The LWR model  `[lwr.py: FundamentalDiagram]`

### 1.1 Conservation law
Let $N(x,t)$ be the cumulative number of vehicles that have passed position $x$ by time $t$, so that
$\rho = -\partial_x N$ and $q = \partial_t N$. Vehicles are neither created nor destroyed on a
road without ramps. Over an interval $[a,b]$:

$$\frac{d}{dt}\int_a^b \rho(x,t)\,dx = q(a,t) - q(b,t).$$

For smooth $\rho$ this gives the **Lighthill–Whitham–Richards** equation

$$\boxed{\;\partial_t \rho + \partial_x q(\rho) = 0,\qquad q(\rho)=\rho\, v(\rho)\;}$$

The model closes the system by assuming speed is an instantaneous function of density only (the
*fundamental diagram*). We use Greenshields:

$$v(\rho)=v_f\Big(1-\frac{\rho}{\rho_{\max}}\Big),\qquad
q(\rho)=v_f\,\rho\Big(1-\frac{\rho}{\rho_{\max}}\Big).$$

Properties used throughout:

* $q$ is **strictly concave**: $q''(\rho)=-2v_f/\rho_{\max}<0$.
* Critical density $\rho_c=\rho_{\max}/2$ maximises $q$; capacity $q_{\max}=v_f\rho_{\max}/4$.
* Characteristic (kinematic wave) speed $\;q'(\rho)=v_f(1-2\rho/\rho_{\max})$, positive in free flow
  ($\rho<\rho_c$), negative in congestion ($\rho>\rho_c$). Information travels at speed $|q'|\le v_f$.

With the default parameters $v_f=15$ m/s, $\rho_{\max}=0.15$ veh/m: $q_{\max}=0.5625$ veh/s $=2025$ veh/h.

### 1.2 Characteristics
Along curves $dx/dt=q'(\rho)$ the density is constant, because
$\frac{d}{dt}\rho(x(t),t)=\rho_t+q'(\rho)\rho_x=0$. Characteristics can cross (faster cars
catching slower traffic), so classical solutions break down in finite time even for smooth
initial data. We therefore need weak solutions.

### 1.3 Weak solutions, shocks, entropy condition
$\rho$ is a weak solution if for all smooth compactly supported test functions $\varphi$:
$\iint (\rho\varphi_t+q(\rho)\varphi_x)\,dx\,dt+\int \rho_0\varphi(x,0)\,dx=0.$

Across a discontinuity moving at speed $s$, conservation of vehicles forces the
**Rankine–Hugoniot condition**

$$s=\frac{q(\rho_R)-q(\rho_L)}{\rho_R-\rho_L}.$$

For Greenshields this simplifies to $s=v_f\big(1-(\rho_L+\rho_R)/\rho_{\max}\big)$ — implemented as a
unit test (`test_shock_speed_rankine_hugoniot`, which checks the numerical shock position against $sT$).

Weak solutions are non-unique, so we pick the physically relevant one with the **Lax entropy
condition** $q'(\rho_L)>s>q'(\rho_R)$. For concave $q$ this means a shock is admissible only when
$\rho_L<\rho_R$ (vehicles run into slower, denser traffic = the back of a queue); a jump from dense
to light traffic is instead a smooth **rarefaction fan** (a queue dissolving).

### 1.4 The Riemann problem
Initial data $\rho_0(x)=\rho_L$ for $x<0$, $\rho_R$ for $x>0$ has the entropy solution

* $\rho_L<\rho_R$: a shock with speed $s$ above.
* $\rho_L>\rho_R$: a rarefaction fan $\rho(x,t)=(q')^{-1}(x/t)$ between $q'(\rho_L)$ and $q'(\rho_R)$.

The flux at $x=0$ (the quantity a finite-volume scheme needs) is
$q^*(\rho_L,\rho_R)=q(\rho^*)$ where $\rho^*$ is the value of the solution along $x/t=0$:

| $\rho_L$ | $\rho_R$ | wave | $q^*$ |
|---|---|---|---|
| $\le\rho_c$ | $\le\rho_c$ | shock/rarefaction moving right | $q(\rho_L)$ |
| $\le\rho_c$ | $>\rho_c$ | shock | $\min\{q(\rho_L),q(\rho_R)\}$ |
| $>\rho_c$ | $\le\rho_c$ | fan containing $x/t=0$ ($q'=0$) | $q_{\max}$ |
| $>\rho_c$ | $>\rho_c$ | wave moving left | $q(\rho_R)$ |

---

## 2. Demand and supply functions  `[lwr.py: demand, supply]`

Define the **sending (demand)** and **receiving (supply)** functions

$$D(\rho)=\begin{cases}q(\rho)&\rho\le\rho_c\\ q_{\max}&\rho>\rho_c\end{cases},\qquad
S(\rho)=\begin{cases}q_{\max}&\rho\le\rho_c\\ q(\rho)&\rho>\rho_c\end{cases}.$$

$D$ is non-decreasing, $S$ non-increasing, and $q=\min(D,S)$. Intuition: $D$ is the most a
road section can *send* downstream (limited by how many vehicles are there, up to capacity);
$S$ is the most it can *receive* from upstream (limited by free space, up to capacity).

**Theorem (Godunov flux in demand–supply form).** For concave $q$ with a single maximum, the exact
Riemann flux is
$$\boxed{F(\rho_L,\rho_R)=\min\{D(\rho_L),\,S(\rho_R)\}}.$$

*Proof by cases* (matching the table in 1.4):
1. $\rho_L,\rho_R\le\rho_c$: $D(\rho_L)=q(\rho_L)\le q_{\max}=S(\rho_R)$, so $F=q(\rho_L)$. ✓
2. $\rho_L\le\rho_c<\rho_R$: $D=q(\rho_L)$, $S=q(\rho_R)$, so $F=\min\{q(\rho_L),q(\rho_R)\}$. ✓
3. $\rho_L>\rho_c\ge\rho_R$: $D=q_{\max}=S$, so $F=q_{\max}$. ✓
4. $\rho_L,\rho_R>\rho_c$: $D=q_{\max}\ge q(\rho_R)=S$, so $F=q(\rho_R)$. ✓ $\blacksquare$

The test `test_godunov_flux_equals_exact_riemann_flux` checks this identity against an independent
implementation of the exact Riemann solution on 2000 random state pairs.

---

## 3. Finite-volume discretisation  `[lwr.py: godunov_step]`

Divide a link into cells $C_i=[x_{i-1/2},x_{i+1/2}]$ of length $\Delta x$ and let
$\rho_i^n\approx\frac1{\Delta x}\int_{C_i}\rho(x,t_n)\,dx$. Integrating the conservation law over
$C_i\times[t_n,t_{n+1}]$ gives the *exact* identity

$$\rho_i^{n+1}=\rho_i^n-\frac{1}{\Delta x}\int_{t_n}^{t_{n+1}}\!\big[q(x_{i+1/2},t)-q(x_{i-1/2},t)\big]dt .$$

Godunov's idea: at each interface, the solution over one step is the self-similar Riemann solution
between the two neighbouring cell averages, so the interface flux is **constant in time** and equals
$q^*(\rho_i^n,\rho_{i+1}^n)=F(\rho_i^n,\rho_{i+1}^n)$. Hence

$$\boxed{\rho_i^{n+1}=\rho_i^n-\frac{\Delta t}{\Delta x}\Big(F_{i+1/2}^n-F_{i-1/2}^n\Big),\quad
F_{i+1/2}^n=\min\{D(\rho_i^n),S(\rho_{i+1}^n)\}.}$$

This is valid while waves from neighbouring interfaces do not interact within a step, i.e. under the
**CFL condition** $\;v_f\,\Delta t/\Delta x\le1$ (asserted in the code; defaults give $0.6$).

### 3.1 Properties
Write $\lambda=\Delta t/\Delta x$ and $H(a,b,c)=b-\lambda\big(F(b,c)-F(a,b)\big)$.

* **Conservation (discrete).** Summing over cells, interior fluxes telescope:
  $\Delta x\sum_i\rho_i^{n+1}=\Delta x\sum_i\rho_i^n+\Delta t\,(F_{\text{in}}-F_{\text{out}})$.
  Tested to $10^{-6}$ veh in `test_node_mass_conservation_and_capacity`.
* **Monotonicity ⇒ invariant region.** Write $F(a,b)=\min\{D(a),S(b)\}$, which is non-decreasing in $a$
  and non-increasing in $b$. For the update $H(a,b,c)=b-\lambda\big(F(b,c)-F(a,b)\big)$:
  $$\partial_aH=\lambda\,\partial_1F(a,b)\ge0,\qquad \partial_cH=-\lambda\,\partial_2F(b,c)\ge0,$$
  $$\partial_bH=1-\lambda\,\partial_1F(b,c)+\lambda\,\partial_2F(a,b).$$
  The two terms in $\partial_bH$ that could be negative never act together: $\partial_1F(b,c)\ne0$ needs
  $D'(b)\ne0$, i.e. $b<\rho_c$, while $\partial_2F(a,b)\ne0$ needs $S'(b)\ne0$, i.e. $b>\rho_c$. Each has
  magnitude at most $v_f$, so $\partial_bH\ge1-\lambda v_f\ge0$ under the CFL condition. Thus $H$ is
  **monotone non-decreasing in all three arguments**. The constants $0$ and $\rho_{\max}$ are fixed
  points of $H$ (zero flux everywhere), so by monotonicity $0\le\rho_i^n\le\rho_{\max}$ for all $n$
  whenever the initial data are — densities cannot go negative or exceed jam density (checked in tests).
* **Convergence.** A conservative, monotone scheme converges to the unique entropy solution as
  $\Delta x\to0$ with $\lambda$ fixed (Lax–Wendroff theorem plus Crandall–Majda / Harten–Hyman–Lax).
  It is only **first-order accurate** (numerical diffusion smears shocks over a few cells; the
  Kuznetsov bound gives $\|\rho_{\Delta x}-\rho\|_{L^1}=O(\sqrt{\Delta x})$ for BV data, typically
  $O(\Delta x)$ observed for piecewise smooth solutions).

---

## 4. Boundary conditions  `[network.py: step]`

A boundary acts like a ghost cell with prescribed flux, limited by what the adjacent cell can
physically do:

* **Entry link:** admitted flow $=\min\{\lambda_{\text{arr}}(t)+Q/\Delta t,\;S(\rho_1)\}$, where $Q$ is the
  virtual queue outside the network. The queue evolves as $\dot Q=\lambda_{\text{arr}}-\text{admitted}$ and
  **counts towards total time spent**, so a controller cannot "win" by blocking entry.
* **Exit link:** free outflow $D(\rho_n)$ (no downstream restriction).
* **Signalised stop line:** outflow equals the node-model flux $f_i\le D(\rho_n)$ if green, $0$ if red.

---

## 5. Intersection (node) model  `[network.py: node_flows]`

Each node has inbound links $i$ (arriving moving direction $i\in\{E,W,S,N\}$) and outbound links $j$.
$T_{ij}\ge0$ with $\sum_jT_{ij}=1$ is the turning matrix (default through 0.7, left 0.15, right 0.15).
Let $g_i\in\{0,1\}$ indicate a green signal for inbound $i$ (phase 0 serves E/W, phase 1 serves N/S;
all-red during the clearance period).

1. **Sending.** $\hat D_i=g_i\,D(\rho_{n}^{(i)})$ — red approaches send nothing.
2. **Demand on each outbound link.** $d_j=\sum_i T_{ij}\hat D_i$.
3. **Proportional supply scaling.** $\alpha_j=\min\{1,\;S_j/d_j\}$ where $S_j=S(\rho_1^{(j)})$.
4. **FIFO restriction.** A queue is first-in-first-out, so an inbound link is throttled by its most
   restricted turning movement: $f_i=\hat D_i\cdot\min_{j:\,T_{ij}>0}\alpha_j$.
5. **Outbound inflow.** $\phi_j=\sum_iT_{ij}f_i$.

**Conservation:** what leaves inbound links $\sum_i f_i$ equals what enters outbound links $\sum_j\phi_j$
because rows of $T$ sum to one. **Admissibility:**
$\phi_j=\sum_iT_{ij}\hat D_i\min_k\alpha_k\le\alpha_j d_j\le S_j$, so outbound links are never
over-filled and the Godunov update in section 3 stays in its invariant region. Spillback emerges
naturally: a full outbound link has small $S_j$, which lowers $\alpha_j$ and throttles every inbound
link that turns into it — blocking the intersection (FIFO blocking), exactly the mechanism that causes
gridlock in real networks and that an RL controller can learn to avoid.

---

## 6. Network quantities and the reward

Let $M(t)=\Delta x\sum\rho+\sum Q$ be the number of vehicles in the system (links plus waiting
outside), $A(t)$ the cumulative arrivals and $D_{\text{ex}}(t)$ the cumulative exits. Then

$$M(t)=A(t)-D_{\text{ex}}(t),\qquad
\text{TTS}=\int_0^TM(t)\,dt=\int_0^T\big(A(t)-D_{\text{ex}}(t)\big)\,dt ,$$

(the area between the cumulative curves — Newell's formulation). Since $A$ is exogenous (demand),

$$\min \text{TTS}\iff \max\int_0^T D_{\text{ex}}(t)\,dt,$$

i.e. controllers should discharge vehicles as early as possible. Instantaneous network flux is a
dense, local proxy of the exit flux, and the **cumulative flux telescopes**:
$\sum_t \text{(exits)}=A(T)-M(T)$ — a controller that keeps more vehicles in the network at the end
has by construction lower cumulative throughput. This motivates the reward below, and why the
flux reward is aligned with the headline metric (TTS).

---

## 7. Signal control as a Markov decision process  `[qlearning.py]`

Each intersection $k$ is an agent that decides every $\Delta t_c=5$ s whether to **keep** ($a=0$) or
**switch** ($a=1$) its phase. A switch costs $L=2$ s of all-red (lost time).

**State compression.** The local observation is phase-relative (so the two phases share statistics):

* $g$ = detector density of the busiest *green* approach, $r$ = of the busiest *red* approach
  (normalised by $\rho_{\max}$), each binned with edges $\{0.12,0.25,0.45\}$ → 4 levels,
* green time bin $\{<20\,\text{s},\,20\text{–}40\,\text{s},\ge40\,\text{s}\}$ → 3 levels,
* blocked flag: some outbound link's first cell above $0.6\rho_{\max}$ (downstream spillback) → 2 levels.

$|\mathcal S|=4\cdot4\cdot3\cdot2=\mathbf{96}$ states, $|\mathcal A|=2$. Quotienting out
the identity of the green axis (E/W vs N/S) halves the table relative to using absolute axes,
and the table is shared by all agents.

**Reward.** Let $\phi_k(t)$ be vehicles discharged through node $k$ in the control interval,
normalised by $2q_{\max}\Delta t_c$ (max possible), and $\bar\phi=\frac1K\sum_k\phi_k$:

$$r_k=\mu\,\bar\phi+(1-\mu)\,\phi_k,\qquad\mu=0.5 .$$

$\mu=1$ is the pure network-wide reward (cooperative, but it contains the flux of $K-1$ other
agents, so the share of the signal attributable to any one agent's action shrinks roughly like $1/K$ —
the credit-assignment problem); $\mu=0$ is purely local (low noise, but ignores neighbours).
$\mu$ is an ablation parameter.

**Bellman optimality.** The optimal action-value function satisfies
$$Q^*(s,a)=\mathbb E\big[r+\gamma\max_{a'}Q^*(s',a')\,\big|\,s,a\big],\quad\gamma=0.9 .$$

**Q-learning update** (applied for every agent transition to the one shared table):
$$Q(s,a)\leftarrow Q(s,a)+\alpha\big[r+\gamma\max_{a'}Q(s',a')-Q(s,a)\big].$$
For a stationary finite MDP, Watkins–Dayan guarantee $Q\to Q^*$ w.p. 1 if every $(s,a)$ is visited
infinitely often and the step sizes satisfy Robbins–Monro ($\sum\alpha_n=\infty,\ \sum\alpha_n^2<\infty$).
**Honest caveats** (state them in the report): (i) we use a constant $\alpha$ (tracking, not
converging exactly); (ii) the compressed state is partially observable (aliasing); (iii) other
intersections are simultaneously learning, so each agent sees a non-stationary environment.
So the guarantee is a motivation, not a theorem for this system — which is why we evaluate empirically.

**Value scale.** With per-interval reward $r\in[0,1]$, $Q^*\approx\bar r/(1-\gamma)$. Measured $\bar r$
(0.4–0.9) gives $Q\approx4\text{–}9$. Initialising $Q$ far *below* this scale makes any tried action look
better than untried ones and freezes exploration; far above gives long optimistic exploration. We
initialise at $6$ (found by a small sweep: see report).

**Parameter sharing.** $K$ agents write to one table, so each $(s,a)$ receives $\sim K\times$ more
samples per episode, which is what makes cold-start training quick on large grids and allows the
policy trained on a small grid to be **transferred** unchanged to larger ones (the state is local and
phase-relative, hence topology-independent).

---

## 8. Sensor-informed prior  `[qlearning.py: sensor_prior_actions, controllers.py: actuated_rule]`

The classic **actuated** rule is: switch iff $t_g\ge t_{\min}$, a vehicle is calling on red
($r\ge r_{\text{call}}$) and (the green side has gapped out, $g<g_{\text{gap}}$, or $t_g\ge t_{\max}$).
Call this deterministic map $\pi_s(s)$ after evaluating it at representative values of each bin.

1. **Prior initialisation.** $Q_0(s,a)=q_0+b\,\mathbb 1[a=\pi_s(s)]$. Initially, the greedy policy of the
   agent *is* the sensor policy, so the cold-start performance is that of the incumbent controller instead of
   random switching.
2. **Guided exploration.** Behaviour policy
   $$\mu(a|s)=(1-\varepsilon)\,\mathbb 1[a=\arg\max Q]+\varepsilon\big[p_g\,\mathbb 1[a=\pi_s(s)]+(1-p_g)\tfrac12\big].$$
   Q-learning is off-policy, so its fixed point $Q^*$ is **unchanged** by this choice; it only changes
   *which* samples are collected. Every action still has probability $\ge\varepsilon(1-p_g)/2>0$,
   preserving the "visit every pair infinitely often" condition ($p_g=0.7$).
3. **Safety layer.** Minimum green and maximum green (when red has a call) are enforced for the RL agent
   exactly as for the baseline — the agent only decides in the free region.

The "RL pure" ablation uses $b=0$, $p_g=0$ (and a flat table), so the difference between the two
isolates the value of the sensor prior for cold-start convergence.

---

## 9. Evaluation metrics

* **TTS** [veh·s] as in section 6 (primary; lower is better).
* Throughput $D_{\text{ex}}(T)$ and residual vehicles $M(T)$.
* Relative change versus the actuated controller on identical demand seeds:
  $\Delta=100\,(\overline{\text{TTS}}_{\text{method}}-\overline{\text{TTS}}_{\text{act}})/\overline{\text{TTS}}_{\text{act}}$.
  RL results are reported as mean ± std over independent **training seeds** (the dominant source of
  variance), alongside how many seeds beat the baseline.

## 10. Known limitations of the model

* LWR has no inertia/stop-and-go dynamics and a single-class, single-lane-equivalent flow; Greenshields has
  a symmetric, unrealistic capacity drop shape (no capacity drop at queue discharge).
* Routing is by fixed turning ratios, not origin–destination paths.
* First-order Godunov diffuses shocks; with $\Delta x=25$ m queues are resolved to $\pm$ a few cells.
* Tabular Q-learning with an 96-state compression cannot express long-range coordination (green waves);
  a natural extension is neighbour-pressure features or function approximation (DQN).
