#%%
import numpy as np
import scipy
import matplotlib.pyplot as plt
import seaborn as sns

#%%
data = np.genfromtxt('stylos.csv', delimiter=',', skip_header=1)

#%%
var_a = (np.log(2) / 1.96)**2
# posterior distribution
def posterior(theta, data):
  x = data[:,0][:,np.newaxis]
  y = data[:,1][:,np.newaxis]
  ln_a, b, var = theta
  n = data.shape[0]

  # support: 0 < beta < 1 and sigma^2 > 0
  in_support = (b > 0) & (b < 1) & (var > 0)
  # placeholder variance outside the support, so the sqrt below stays valid
  var = np.where(var > 0, var, 1.0)

  pi_ln_a = np.exp(-ln_a**2 / (2*var_a))
  pi_b = b**4 * (1-b)**4
  pi_var = 1/var

  L = np.prod(
    np.exp(-(np.log(y) - ln_a - b*np.log(x))**2 / (2*var))
      / (y * np.sqrt(2*np.pi*var)),
    axis=0
  )

  return np.where(in_support, pi_ln_a * pi_b * pi_var * L, 0.0)

#%%
def beta_proposal_params(mode, var):
  # Beta(a, b) with the given mode and variance (requires var < 1/12).
  # With a = 1 + mode*k and b = 1 + (1-mode)*k, the mode is fixed for any k > 0.
  # Setting the variance to var gives a cubic in k with exactly one positive root.
  roots = np.roots([var, 7*var - mode*(1-mode), 16*var - 1, 12*var - 1])
  k = roots[np.isreal(roots) & (roots.real > 0)].real[0]
  return 1 + mode*k, 1 + (1-mode)*k

def sample_posterior_mh(data, proposal_dist='rw',
  proposal_params=(0.03, 0.0004, 0.46), size=1000, init=(0, 0.5, 0.01)):
  sample = np.empty((size, 3))
  u = scipy.stats.uniform.rvs(size=size)

  # defaults to the prior means
  sample[0,:] = init

  match proposal_dist:
    case 'baseline':
      q = lambda theta, _: 1
      # Proposals with sigma^2 <= 0 have zero posterior density and are rejected,
      # so no clipping is needed (clipping would make the walk asymmetric)
      def sample_q(t):
        return sample[t,:] + scipy.stats.norm.rvs(scale=[0.1, 0.1, 0.01], size=3)
    case 'rw':
      # ln_a_step = 0.03     # sd of the Gaussian walk on ln(alpha)
      # b_var = 0.0004       # variance of the Beta proposal for beta
      # log_var_step = 0.46  # sd of the Gaussian walk on ln(sigma^2)
      ln_a_step, b_var, log_var_step = proposal_params

      # Density of proposing theta from given. The ln(alpha) walk is symmetric,
      # so its term cancels in the acceptance ratio and is left out.
      def q(theta, given):
        return (
            scipy.stats.beta.pdf(theta[1], *beta_proposal_params(given[1], b_var))
            * scipy.stats.lognorm.pdf(theta[2], s=log_var_step, scale=given[2])
        )

      def sample_q(t):
        ln_a, b, var = sample[t,:]
        return np.array([
            ln_a + scipy.stats.norm.rvs(scale=ln_a_step),
            scipy.stats.beta.rvs(*beta_proposal_params(b, b_var)),
            var * np.exp(scipy.stats.norm.rvs(scale=log_var_step)),
        ])

  for t in range(size-1):
      proposal = sample_q(t)
      ratio = (
          posterior(proposal, data) * q(sample[t,:], proposal)
          / (posterior(sample[t,:], data) * q(proposal, sample[t,:]))
      )
      accept_prob = min(ratio, 1.0)

      if u[t] < accept_prob:
        sample[t+1,:] = proposal
      else:
        sample[t+1,:] = sample[t,:]

  return sample

#%%
# Evaluate the true posterior on a 3D grid, used for the true marginals below.
# The grids must cover essentially all of the posterior mass.
ln_a_grid = np.linspace(0.04, 0.30, 121)
b_grid = np.linspace(0.30, 0.54, 121)
var_grid = np.linspace(0.002, 0.045, 121)
grids = [ln_a_grid, b_grid, var_grid]

# Unnormalised posterior, indexed as [ln_a, b, var]
B, V = np.meshgrid(b_grid, var_grid, indexing='ij')
density = np.empty((len(ln_a_grid), len(b_grid), len(var_grid)))
for i, ln_a in enumerate(ln_a_grid):
  density[i] = posterior((ln_a, B.ravel(), V.ravel()), data).reshape(B.shape)

trapezoid = scipy.integrate.trapezoid

# Each contour encloses this much of the probability, in the increasing level order contour() needs
contour_masses = [0.99, 0.9, 0.5]

def density_levels(z, masses):
  # Density values whose highest-density regions enclose the given probability masses
  z_sorted = np.sort(z.ravel())[::-1]
  cumulative = np.cumsum(z_sorted) / z_sorted.sum()
  return [z_sorted[np.searchsorted(cumulative, mass)] for mass in masses]

#%%
x_samples = sample_posterior_mh(data, size=5000)
labels = ['$\\ln\\alpha$', '$\\beta$', '$\\sigma^2$']

# Transpose xs_samples to iterate over columns (each parameter's history of samples)
x_cols = x_samples.T

# Small, faint points and thin lines, so that thousands of samples stay readable
plot_style = dict(markersize=3, markeredgewidth=0, linewidth=1.5, alpha=0.6)

for i, (label, x) in enumerate(zip(labels, x_cols)):
  plt.plot(x, 'o-', **plot_style)
  plt.title(label)
  plt.show()

  # bivariates
  for j, (label2, x2) in enumerate(zip(labels[i+1:], x_cols[i+1:])):
    plt.figure()
    plt.plot(x, x2, 'o-', **plot_style)

    # True bivariate marginal: integrate out the third parameter
    k = i + 1 + j
    third = 3 - i - k
    bivariate = trapezoid(density, grids[third], axis=third)
    levels = density_levels(bivariate, contour_masses)
    contours = plt.contour(grids[i], grids[k], bivariate.T, levels=levels, colors='black', linewidths=1)
    plt.clabel(contours, fmt={level: f'{mass:.0%}' for level, mass in zip(levels, contour_masses)})

    plt.title(f'{label} vs {label2}')
    plt.xlabel(label)
    plt.ylabel(label2)
    plt.show()

#%%
n = 5000
m = 5
# Gelman-Rubin needs starting points overdispersed relative to the posterior
inits = [
  (0, 0.5, 0.01),
  (0.4, 0.2, 0.05),
  (-0.1, 0.7, 0.05),
  (0.4, 0.7, 0.02),
  (-0.1, 0.2, 0.02),
]
# Samplers to compare: name -> keyword arguments for sample_posterior_mh
samplers = {
  'Baseline Gaussian walk': dict(proposal_dist='baseline'),
  'RW, default steps': dict(proposal_dist='rw', proposal_params=(0.03, 0.0004, 0.46)),
  'RW, larger steps': dict(proposal_dist='rw', proposal_params=(0.045, 0.0009, 0.7)),
}
# One colour per sampler, in a fixed order
sampler_colors = ['#2a78d6', '#eb6834', '#1baf7a']

# Each indexed as [iteration, chain, parameter]
samples = {
  name: np.stack(
    [sample_posterior_mh(data, size=2*n, init=init, **kwargs) for init in inits],
    axis=1
  )
  for name, kwargs in samplers.items()
}
for name, chains in samples.items():
  # A rejected proposal repeats the previous state
  accept_rate = (chains[1:] != chains[:-1]).any(axis=2).mean()
  print(f'{name}: acceptance rate {accept_rate:.2f}')

# Thin the default RW chains, keeping every k-th draw. The default chain's integrated
# autocorrelation time is roughly 11-15 iterations, so k = 10 leaves kept draws nearly uncorrelated.
k = 10
samples[f'RW, default steps, thinned (k={k})'] = samples['RW, default steps'][::k]
# The thinned chains reuse the default RW colour, dashed
sampler_colors.append(sampler_colors[1])
sampler_styles = ['-', '-', '-', '--']

#%%
def gelman_rubin(chains):
  # Potential scale reduction factor of each parameter, for chains indexed as above
  length = chains.shape[0]
  within = chains.var(axis=0, ddof=1).mean(axis=0)
  between = length * chains.mean(axis=0).var(axis=0, ddof=1)
  pooled = (length - 1) / length * within + between / length
  return np.sqrt(pooled / within)

def acf(x, max_lag):
  x = x - x.mean()
  autocovariance = np.correlate(x, x, mode='full')[len(x) - 1:]
  return autocovariance[:max_lag + 1] / autocovariance[0]

# R-hat as the chains grow, discarding the first half of the iterations at each point.
# Iterations count every draw of the sampler, kept or not, so thinned chains line up.
iterations = np.arange(100, 2*n + 1, 100)
r_hat = {}
for name, chains in samples.items():
  thin = 2*n // len(chains)
  r_hat[name] = np.array([gelman_rubin(chains[t // 2 // thin:t // thin]) for t in iterations])
for name, values in r_hat.items():
  print(f'{name}: final R-hat (ln_a, b, var) {values[-1].round(3)}')

fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
for j, (ax, label) in enumerate(zip(axes, labels)):
  for (name, values), color, style in zip(r_hat.items(), sampler_colors, sampler_styles):
    ax.plot(iterations, values[:, j], style, linewidth=2, color=color, label=name)
  ax.axhline(1.1, color='gray', linestyle='--', linewidth=1, label='1.1 threshold')
  ax.set_title(f'Gelman-Rubin $\\hat{{R}}$ of {label}')
  ax.set_xlabel('Iteration')
  ax.set_ylabel('$\\hat{R}$')
  ax.spines[['top', 'right']].set_visible(False)
# One shared legend above the panels
fig.legend(*axes[0].get_legend_handles_labels(), loc='upper center', ncol=3, frameon=False)
plt.tight_layout(rect=(0, 0, 1, 0.84))
plt.show()

# Autocorrelation after burn-in, averaged over the chains
max_lag = 100
lags = np.arange(max_lag + 1)

fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
for j, (ax, label) in enumerate(zip(axes, labels)):
  for (name, chains), color, style in zip(samples.items(), sampler_colors, sampler_styles):
    # Second half of each chain, so thinned chains lose the same burn-in
    kept = chains[len(chains) // 2:]
    mean_acf = np.mean([acf(kept[:, chain, j], max_lag) for chain in range(m)], axis=0)
    # ax.plot(lags, mean_acf, style, linewidth=2, color=color, label=name)
    ax.bar(lags, mean_acf, color=color, label=name)
  ax.axhline(0, color='gray', linewidth=1)
  ax.set_title(f'Autocorrelation of {label}')
  # Lags count kept draws, so one thinned lag spans k sampler iterations
  ax.set_xlabel('Lag (kept draws)')
  ax.set_ylabel('ACF')
  ax.spines[['top', 'right']].set_visible(False)
fig.legend(*axes[0].get_legend_handles_labels(), loc='upper center', ncol=2, frameon=False)
plt.tight_layout(rect=(0, 0, 1, 0.84))
plt.show()

#%%
ln_a_pos_mean = np.mean(x_samples[:,0])
b_pos_mean = np.mean(x_samples[:,1])
def model(x, ln_a, b):
  return np.exp(ln_a + b*np.log(x))

x = np.linspace(data[:,0].min(), data[:,0].max(), 100)
plt.plot(x, model(x, ln_a_pos_mean, b_pos_mean), label='Posterior mean fit')
plt.plot(x, model(x, ln_a_pos_mean, 0.1), label='Low beta')
plt.plot(x, model(x, ln_a_pos_mean, 0.7), label='High beta')
plt.plot(data[:,0], data[:,1], label='Data')
plt.legend()

#%%
# True marginals: integrate the posterior numerically over the other two parameters.
ln_a_marginal = trapezoid(trapezoid(density, var_grid, axis=2), b_grid, axis=1)
b_marginal = trapezoid(trapezoid(density, var_grid, axis=2), ln_a_grid, axis=0)
var_marginal = trapezoid(trapezoid(density, b_grid, axis=1), ln_a_grid, axis=0)

marginals = [ln_a_marginal, b_marginal, var_marginal]

# Iterations to discard while the chain moves away from its starting point
burn_in = 200

fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
for j, (ax, grid, marginal, label) in enumerate(zip(axes, grids, marginals, labels)):
  ax.hist(x_samples[burn_in:, j], bins=30, density=True, color='lightgray', label='MCMC samples')
  # Normalise so that each marginal integrates to 1
  ax.plot(grid, marginal / trapezoid(marginal, grid), linewidth=2, label='True marginal')
  ax.set_title(f'Marginal posterior of {label}')
  ax.set_xlabel(label)
  ax.set_ylabel('Density')
  # Headroom so the legend clears the tallest bars
  ax.set_ylim(0, ax.get_ylim()[1] * 1.25)
  ax.spines[['top', 'right']].set_visible(False)
  ax.legend(frameon=False)

plt.tight_layout()
plt.show()
