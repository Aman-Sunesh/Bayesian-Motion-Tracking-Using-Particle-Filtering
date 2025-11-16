# Bayesian Motion Tracking Using Particle Filtering

This repository implements a **particle filter** for Bayesian motion tracking of a simulated rat.  
We model the rat’s hidden (latent) position over time and noisy sensor observations, then use a
Sequential Monte Carlo (SMC) / particle filtering algorithm to infer the latent trajectory and
predict future motion.

The core work is contained in the notebook:

- `particle_filtering.ipynb`

---

## 1. Problem Overview

We consider a 2D latent state $z_t \in \mathbb{R}^2$ describing the rat’s position and a
2D observation $x_t \in \mathbb{R}^2$ produced by noisy sensors. The dynamics are given by:

- **Latent transition**: `latent_sample(z_t) -> z_{t+1}`  
- **Observation model**: `observation_sample(z_t) -> x_t`  
- **Observation likelihood**: `observation_probability(z_t, x_t) = p(x_t \mid z_t)`

Both the transition and observation distributions are **non-Gaussian**, built from mixtures of
Gaussians and nonlinear transformations. This makes Kalman filtering inapplicable and motivates
the use of a particle filter.

The goal is to:
1. Reconstruct the latent trajectory from noisy observations.
2. Study the filtering error over time.
3. Predict future latent states and observations beyond the last measurement.

---

## 2. Implementation Details

All logic lives in `particle_filtering.ipynb`.

### 2.1 Generative Model

We implement:

- `observation_probability(latent, observation)`:  
  Computes the likelihood $p(x_t \mid z_t)$ using a mixture of 3 Gaussians with different
  means and covariances.

- `observation_sample(latent)`:  
  Samples an observation from the same mixture model.

- `latent_sample(latent)`:  
  Samples the next latent state using a nonlinear update involving sine/cosine terms plus
  Gaussian noise and a drift based on the previous state.

These functions are used both to **simulate data** (the “true” rat path and observations) and as
the generative model inside the particle filter.

### 2.2 `ParticleFiltering` Class

The class encapsulates a standard particle filtering pipeline.

**Constructor**

```python
pf = ParticleFiltering(
    dim_z=2,            # latent dimension
    dim_x=2,            # observation dimension
    sigma_w_zero=1.1,   # prior std dev for initial state
    mu_zero=np.array([0.0, 0.0])  # prior mean for initial state
)
```

**Key methods**

- `particle_filter(observations, n_samples)`  
  - Initializes particles from a Gaussian prior $\mathcal{N}(\mu_0, \Sigma_0)$.  
  - For each time step:
    1. **Resample** particles according to their weights.
    2. **Propagate** them through `latent_sample`.
    3. **Reweight** using `compute_w` and normalize.
  - Returns all latent samples and their weights over time.

- `compute_w(observation_t, z_samples_t)`  
  - Computes importance weights
    $w_i \propto p(x_t \mid z_t^{(i)})$
    using `observation_probability`, and normalizes so that the weights sum to 1.

- `predict(final_latent_samples, final_weights, n_future)`  
  - Starts from the final filtered particle set at time $T$.
  - Repeatedly:
    1. Resamples according to the final weights.
    2. Propagates latent states forward with `latent_sample`.
    3. Generates pseudo-observations with `observation_sample`.
  - Since no new observations arrive, all future weights are uniform.
  - Returns latent and observation samples for future time steps.

---

## 3. Experiments

### 3.1 Data Simulation

We first simulate:

- `latent_states`: the true latent trajectory for 101 time steps.
- `observation_states`: the corresponding noisy observations for 100 time steps.

We visualize:

- Observed rat positions in the plane.
- Filtered latent means vs. true latent states.
- Absolute error over time in each coordinate (X and Y).

### 3.2 Filtering Performance (Part 2)

Configuration:

- Prior: `sigma_w_zero = 1.1`, `mu_zero = [1, 1]`
- Number of particles: `N = 100`

We run `particle_filter` on the full observation sequence and compute the weighted mean of
particles at each time step to estimate $\mathbb{E}[z_t \mid x_{1:t}]$.

**Qualitative answer**

> The later latent states have smaller and more stable errors than the early ones. Early on, the
> filter starts from a prior that may not match the true state well and has seen only a few
> observations, so many particles are in the wrong region and the posterior mean can be far from
> the truth. As more observations arrive, the resample–propagate–reweight cycle gradually
> concentrates particles around states that consistently explain the data, sharpening the posterior
> and reducing average error over time.

### 3.3 Prediction Performance (Part 3)

Configuration:

- Prior: `sigma_w_zero = 1.1`, `mu_zero = [0, 0]`
- Number of particles: `N = 200`
- We use only the **first 80 observations** for filtering.
- Then we use `predict` to simulate **20 future time steps** of latent states and observations.

We compute the mean and standard deviation of the predicted latent and observed position in X,
and overlay the true trajectory with $\pm 2$-standard-deviation prediction bands.

**Qualitative answer**

> The particle filter predicts future observations reasonably well but with growing uncertainty
> over time. In both the latent-state and observation plots, the true trajectory stays mostly
> within the $\pm 2$-standard-deviation prediction bands, showing that the filter captures the general
> direction and variability of the rat’s motion. However, the confidence intervals widen as we
> move further into the future because there are no new observations after time 80, so the filter
> relies only on the stochastic latent dynamics, causing uncertainty to accumulate. Overall, the
> model’s predictions remain consistent with the true values, and the true observations do fall
> within the predicted distribution for nearly all time steps.

---

## 4. Setup and Usage

### 4.1 Requirements

The notebook expects the following Python packages:

- `numpy`
- `matplotlib`
- `scipy`

You can install them with:

```bash
pip install numpy matplotlib scipy
```

### 4.2 Running the Notebook

1. Clone the repository:

   ```bash
   git clone https://github.com/Aman-Sunesh/Bayesian-Motion-Tracking-Using-Particle-Filtering.git
   cd Bayesian-Motion-Tracking-Using-Particle-Filtering
   ```

2. Start Jupyter (or VS Code / any notebook environment) and open:

   ```text
   particle_filtering.ipynb
   ```

3. Run all cells from top to bottom to:
   - Simulate the rat’s trajectory and observations.
   - Run particle filtering on the observed data.
   - Visualize reconstruction quality and absolute errors.
   - Predict future latent states and observations and visualize uncertainty bands.

---

## 5. Repository Structure (minimal)

```text
.
├── particle_filtering.ipynb   # Main notebook with implementation and experiments
└── README.md                  # Project overview and instructions
```

You can extend this structure with additional modules, figures, or reports as the project grows.

---

## 6. License

You may choose a license appropriate for coursework, personal projects, or open-source sharing.
If unspecified, this repository currently has **no explicit license**, meaning all rights are
reserved by default.
