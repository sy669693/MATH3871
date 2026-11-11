#%%
import numpy as np
import scipy
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import time

#%%
LINEWIDTH = 469.75502 / 72

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

def mh_chain(data, proposal_dist, proposal_params, size, init):
  # One Metropolis-Hastings chain, indexed as [iteration, parameter]
  sample = np.empty((size, 3))
  sample[0,:] = init
  u = np.random.uniform(size=size-1)

  match proposal_dist:
    case 'baseline':
      # Proposals with sigma^2 <= 0 have zero posterior density and are rejected,
      # so no clipping is needed (clipping would make the walk asymmetric)
      steps = np.random.normal(scale=[0.1, 0.1, 0.01], size=(size-1, 3))

      # Returns the proposal and q(current | proposal) / q(proposal | current),
      # which is 1 for a symmetric walk
      def propose(current, t):
        return current + steps[t], 1.0
    case 'rw':
      # ln_a_step = 0.03     # sd of the Gaussian walk on ln(alpha)
      # b_var = 0.0004       # variance of the Beta proposal for beta
      # log_var_step = 0.46  # sd of the Gaussian walk on ln(sigma^2)
      ln_a_step, b_var, log_var_step = proposal_params
      # Gaussian steps of the walks on ln(alpha) and ln(sigma^2), drawn up front
      steps = np.random.normal(scale=[ln_a_step, log_var_step], size=(size-1, 2))

      def propose(current, t):
        ln_a, b, var = current
        forward = beta_proposal_params(b, b_var)
        b_new = scipy.stats.beta.rvs(*forward)
        proposal = np.array([ln_a + steps[t,0], b_new, var * np.exp(steps[t,1])])
        # The ln(alpha) walk is symmetric, so its term cancels. The log-normal walk on
        # sigma^2 has q(var | var') / q(var' | var) = var' / var.
        q_ratio = (
            scipy.stats.beta.pdf(b, *beta_proposal_params(b_new, b_var))
            / scipy.stats.beta.pdf(b_new, *forward)
            * proposal[2] / var
        )
        return proposal, q_ratio
    case 'independence':
      # proposal_params is a proposal object (see the independence proposal cell) with
      # rvs(size) and logpdf(theta), both for rows theta = (ln_a, b, var)
      proposal_density = proposal_params

      def log_weight(theta):
        # Log importance weight ln(posterior / proposal density), for rows theta.
        # Proposals outside the support get -inf, so they are never accepted.
        theta = np.atleast_2d(theta)
        with np.errstate(divide='ignore'):
          return np.log(posterior(theta.T, data)) - proposal_density.logpdf(theta)

      # Proposals ignore the current state, so all of them are drawn, and their
      # weights computed in one vectorised call, before the loop
      proposals = proposal_density.rvs(size-1)
      proposal_log_weights = log_weight(proposals)
      log_u = np.log(u)

      # Acceptance ratio is w(proposal) / w(current), compared on the log scale
      current_log_weight = log_weight(sample[0,:])[0]
      for t in range(size-1):
        if log_u[t] + current_log_weight < proposal_log_weights[t]:
          sample[t+1,:] = proposals[t]
          current_log_weight = proposal_log_weights[t]
        else:
          sample[t+1,:] = sample[t,:]
      return sample

  # The current state's posterior is cached, so each iteration evaluates it once
  current_post = posterior(sample[0,:], data)
  for t in range(size-1):
    proposal, q_ratio = propose(sample[t,:], t)
    proposal_post = posterior(proposal, data)

    # Accept with probability min(1, ratio), written without the division
    if u[t] * current_post < proposal_post * q_ratio:
      sample[t+1,:] = proposal
      current_post = proposal_post
    else:
      sample[t+1,:] = sample[t,:]

  return sample

def sample_posterior_mh(data, proposal_dist='rw',
  proposal_params=(0.03, 0.0004, 0.46), size=1000, init=(0, 0.5, 0.01), n_chains=1):
  # Runs n_chains independent chains. init is either one starting point shared by
  # every chain (defaults to the prior means) or one starting point per chain.
  # Returns [iteration, parameter] for one chain, else [iteration, chain, parameter].
  inits = np.broadcast_to(np.asarray(init, dtype=float), (n_chains, 3))
  chains = np.stack(
    [mh_chain(data, proposal_dist, proposal_params, size, start) for start in inits],
    axis=1
  )
  return chains[:, 0] if n_chains == 1 else chains

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
contour_masses = [0.99, 0.9, 0.75, 0.5, 0.25]

def density_levels(z, masses):
  # Density values whose highest-density regions enclose the given probability masses
  z_sorted = np.sort(z.ravel())[::-1]
  cumulative = np.cumsum(z_sorted) / z_sorted.sum()
  return [z_sorted[np.searchsorted(cumulative, mass)] for mass in masses]

#%%
# Independence proposal: a trivariate t directly on (ln_a, b, var) matched to the posterior.
# Its mean and covariance are the moments of the gridded posterior, so its contours line up
# with the bivariate contours: ln_a and b negatively correlated, sigma^2 nearly independent.
LN_A, B, V = np.meshgrid(*grids, indexing='ij')
grid_points = np.column_stack([LN_A.ravel(), B.ravel(), V.ravel()])
# Grid weights; the grid is uniform in every parameter, so these are posterior cell masses
grid_weights = density.ravel() / density.sum()
indep_mean = grid_weights @ grid_points
indep_cov_fit = np.cov(grid_points, rowvar=False, aweights=grid_weights, ddof=0)
# The fitted correlations of var with ln_a and b are close to zero, so they are set to
# zero: only ln_a and b are correlated in the proposal
indep_cov_fit[2, :2] = indep_cov_fit[:2, 2] = 0

# An independence sampler gets stuck wherever the proposal's tails are lighter than the
# posterior's, so the covariance is inflated a little to make the proposal over-dispersed
indep_inflation = 1.2**2
indep_cov = indep_inflation * indep_cov_fit

# A t rather than a normal: a normal's tails fall off much faster than the posterior's, and
# far from the mode posterior / proposal was e^55 to e^90 times its typical size, so chains
# started there never accepted a move. The t's polynomial tails keep this ratio bounded.
indep_df = 4

indep_sd = np.sqrt(np.diag(indep_cov_fit))
print('Proposal mean (ln_a, b, var):', indep_mean.round(4))
print('Fitted sd:', indep_sd.round(4))
print('Proposal correlation matrix:\n', (indep_cov_fit / np.outer(indep_sd, indep_sd)).round(3))

class TProposal:
  # Proposal for sample_posterior_mh(proposal_dist='independence'): a trivariate t on
  # (ln_a, b, var), with rvs(size) and logpdf(theta) for rows theta = (ln_a, b, var).
  # It ignores that b and var are bounded: proposals outside 0 < b < 1 or var > 0 have zero
  # posterior density, so they are simply rejected, at the cost of a wasted proposal
  # (about 1% of them, nearly all with var <= 0).
  def __init__(self, mean, cov, df):
    # A t's covariance is shape * df / (df - 2), so rescale to match cov
    self.dist = scipy.stats.multivariate_t(mean, cov * (df - 2) / df, df=df)

  def rvs(self, size):
    return self.dist.rvs(size=size).reshape(-1, 3)

  def logpdf(self, theta):
    return self.dist.logpdf(theta)

t_proposal = TProposal(indep_mean, indep_cov, indep_df)

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
  f'Independence, t (df={indep_df})': dict(proposal_dist='independence',
                                          proposal_params=t_proposal),
}
# One colour per sampler, in a fixed order
sampler_colors = ['#2a78d6', '#eb6834', '#1baf7a', '#9b59d0']

# Each indexed as [iteration, chain, parameter], with the wall-clock time of each run
samples = {}
run_times = {}
for name, kwargs in samplers.items():
  start = time.perf_counter()
  samples[name] = sample_posterior_mh(data, size=2*n, init=inits, n_chains=m, **kwargs)
  run_times[name] = time.perf_counter() - start
for name, chains in samples.items():
  # A rejected proposal repeats the previous state
  accept_rate = (chains[1:] != chains[:-1]).any(axis=2).mean()
  print(f'{name}: acceptance rate {accept_rate:.2f}')

# Thin the default RW chains, keeping every k-th draw. The default chain's integrated
# autocorrelation time is roughly 11-15 iterations, so k = 10 leaves kept draws nearly uncorrelated.
k = 10
samples[f'RW, default steps, thinned (k={k})'] = samples['RW, default steps'][::k]
# Thinning reuses the default RW run, so it costs the same time
run_times[f'RW, default steps, thinned (k={k})'] = run_times['RW, default steps']
# The thinned chains reuse the default RW colour, dashed
sampler_colors.append(sampler_colors[1])
sampler_styles = ['-'] * len(samplers) + ['--']

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
    ax.plot(lags, mean_acf, style, linewidth=1.5, color=color, label=name)
  ax.axhline(0, color='gray', linewidth=1)
  ax.set_title(f'Autocorrelation of {label}')
  # Lags count kept draws, so one thinned lag spans k sampler iterations
  ax.set_xlabel('Lag (kept draws)')
  ax.set_ylabel('ACF')
  ax.spines[['top', 'right']].set_visible(False)
fig.legend(*axes[0].get_legend_handles_labels(), loc='upper center', ncol=3, frameon=False)
plt.tight_layout(rect=(0, 0, 1, 0.84))
plt.show()

#%%
# Summary table of each sampler's efficiency, computed on the second half of every chain

def autocorr_time(chains, c=5):
  # Integrated autocorrelation time tau = 1 + 2 * sum of the autocorrelations, for chains
  # indexed as [iteration, chain]. The sum is cut at Sokal's window, the first lag M with
  # M >= c * tau(M), since summing the noisy far tail would swamp the estimate.
  rho = np.mean([acf(chains[:, chain], len(chains) - 1) for chain in range(chains.shape[1])], axis=0)
  taus = 2 * np.cumsum(rho) - 1
  window = np.argmax(np.arange(len(taus)) >= c * taus)
  return taus[window]

# R-hat at a fixed point, early enough to separate the samplers: by 10,000 iterations every
# sampler is already within about 0.02 of 1. Like the R-hat plot, it uses the second half
# of the iterations up to that point.
r_hat_at = 1000
r_hat_index = np.searchsorted(iterations, r_hat_at)

rows = {}
for name, chains in samples.items():
  kept = chains[len(chains) // 2:]
  # A thinned chain's acceptance rate is that of the chain it was thinned from
  source = samples['RW, default steps'] if 'thinned' in name else chains
  rows[name] = {
    'Acceptance rate': (source[1:] != source[:-1]).any(axis=2).mean(),
    **{f'R-hat {label} ({r_hat_at} its)': value
       for label, value in zip(['ln_a', 'b', 'var'], r_hat[name][r_hat_index])},
    **{f'tau {label}': autocorr_time(kept[:, :, j]) for j, label in enumerate(['ln_a', 'b', 'var'])},
    'Time (s)': run_times[name],
  }

summary = pd.DataFrame(rows).T
display(summary.style.format(
  {column: '{:.3f}' if column.startswith('R-hat') else '{:.2f}' if column == 'Acceptance rate'
   else '{:.1f}' for column in summary.columns}
))

#%%
# True and proposal bivariate marginals for each independence sampler, near the mode (top
# row) and on a log scale far out in the tails where the chains start (bottom row). The
# proposal must cover the posterior's tails too, or chains started there cannot escape.

# Wide grid covering every starting point. sigma^2 is log-spaced out to 1, because far from
# the mode the posterior moves to large variances to explain the large residuals.
wide_grids = [np.linspace(-0.15, 0.45, 121), np.linspace(0.15, 0.75, 121), np.geomspace(1e-3, 1, 161)]
WB, WV = np.meshgrid(wide_grids[1], wide_grids[2], indexing='ij')
wide_density = np.empty([len(grid) for grid in wide_grids])
for i, ln_a in enumerate(wide_grids[0]):
  wide_density[i] = posterior((ln_a, WB.ravel(), WV.ravel()), data).reshape(WB.shape)

# Each proposal with its colour from the comparison plots
indep_proposals = {
  f't (df={indep_df})': (t_proposal, sampler_colors[3]),
}

def grid_density(proposal, grids):
  # Proposal density on the 3D grid, indexed as [ln_a, b, var] like the posterior grids
  points = np.stack(np.meshgrid(*grids, indexing='ij'), axis=-1).reshape(-1, 3)
  return np.exp(proposal.logpdf(points)).reshape([len(grid) for grid in grids])

def log_relative(log_z):
  # Log density relative to its own peak, so curves of different heights line up
  return log_z - log_z.max()

# Contours this many log units below each density's peak (e^-80 down to e^-1)
tail_levels = [-80, -40, -20, -5, -1]
pairs = [(0, 1), (0, 2), (1, 2)]
starts = np.array(inits)

# Each proposal on both grids; the bivariate marginals integrate out the third parameter
# numerically, exactly as for the true posterior
proposal_grids = {name: (grid_density(proposal, grids), grid_density(proposal, wide_grids))
                  for name, (proposal, _) in indep_proposals.items()}

fig, axes = plt.subplots(2, 3, figsize=(12, 7.5))
for col, (i, k) in enumerate(pairs):
  third = 3 - i - k

  # Top row: the highest-density contours used before, on the original grid
  ax = axes[0, col]
  true_biv = trapezoid(density, grids[third], axis=third)
  ax.contour(grids[i], grids[k], true_biv.T, levels=density_levels(true_biv, contour_masses),
             colors='black', linewidths=1.5)
  for name, (_, color) in indep_proposals.items():
    proposal_biv = trapezoid(proposal_grids[name][0], grids[third], axis=third)
    ax.contour(grids[i], grids[k], proposal_biv.T, levels=density_levels(proposal_biv, contour_masses),
               colors=color, linewidths=1, linestyles='--')
  ax.set_title(f'Near the mode: {labels[k]} vs {labels[i]}')

  # Bottom row: log density on the wide grid, with the chains' starting points
  ax = axes[1, col]
  true_biv = trapezoid(wide_density, wide_grids[third], axis=third)
  true_log = log_relative(np.log(np.maximum(true_biv, 1e-300)))
  # Solid, since matplotlib otherwise dashes negative levels
  contours = ax.contour(wide_grids[i], wide_grids[k], true_log.T, levels=tail_levels,
                        colors='black', linewidths=1.5, linestyles='solid')
  ax.clabel(contours, fmt=lambda level: f'{level:.0f}', fontsize='small')
  for name, (_, color) in indep_proposals.items():
    proposal_biv = trapezoid(proposal_grids[name][1], wide_grids[third], axis=third)
    proposal_log = log_relative(np.log(np.maximum(proposal_biv, 1e-300)))
    contours = ax.contour(wide_grids[i], wide_grids[k], proposal_log.T, levels=tail_levels,
                          colors=color, linewidths=1, linestyles='--')
    ax.clabel(contours, fmt=lambda level: f'{level:.0f}', fontsize='x-small')
  ax.plot(starts[:, i], starts[:, k], 'x', color='black', markersize=8, markeredgewidth=2)
  if k == 2:
    ax.set_yscale('log')
  ax.set_title(f'Tails (log density below peak): {labels[k]} vs {labels[i]}', fontsize='medium')

for ax, (i, k) in zip(axes.T, pairs):
  for a in ax:
    a.set_xlabel(labels[i])
    a.set_ylabel(labels[k])
    a.spines[['top', 'right']].set_visible(False)

fig.legend(
  handles=[plt.Line2D([], [], color='black', linewidth=1.5, label='True posterior')]
    + [plt.Line2D([], [], color=color, linestyle='--', label=f'{name} proposal')
       for name, (_, color) in indep_proposals.items()]
    + [plt.Line2D([], [], marker='x', linestyle='', color='black', markeredgewidth=2,
                  label='Chain starting points')],
  loc='upper center', ncol=len(indep_proposals) + 2, frameon=False, fontsize='small',
)
plt.tight_layout(rect=(0, 0, 1, 0.95))
plt.show()

# Log importance weight ln(posterior / proposal) at each start, against its typical size
# near the mode. A start whose weight is far above typical would almost never be left.
mode_point = indep_mean[np.newaxis]
for name, (proposal, _) in indep_proposals.items():
  log_w = lambda theta: np.log(posterior(theta.T, data)) - proposal.logpdf(theta)
  print(f'{name}: log weight at starts {log_w(starts).round(1)}, at the mode {log_w(mode_point)[0]:.1f}')

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

#%%
# Trace of each parameter, with its marginal posterior as a sideways histogram.
# Three independent chains from different starting points, indexed as [iteration, chain, parameter]
trace_inits = [(0, 0.5, 0.01), (0.25, 0.35, 0.03), (0.1, 0.5, 0.005)]
trace_chains = sample_posterior_mh(data, size=5000, init=trace_inits, n_chains=len(trace_inits))
chain_colors = ['#2a78d6', '#eb6834', '#1baf7a']
# Post-burn-in draws of every chain pooled together, indexed as [draw, parameter]
pooled = trace_chains[burn_in:].reshape(-1, 3)

fig, axes = plt.subplots(3, 2, figsize=(LINEWIDTH, 7/10*LINEWIDTH),
                         gridspec_kw=dict(width_ratios=[4, 1], wspace=0.05))
for j, (label, grid, marginal) in enumerate(zip(labels, grids, marginals)):
  trace_ax, hist_ax = axes[j]
  # Traces share iterations; each histogram keeps its own density scale
  trace_ax.sharex(axes[0, 0])
  for chain, color in enumerate(chain_colors):
    trace_ax.plot(trace_chains[:, chain, j], linewidth=0.5, alpha=0.7, color=color)
  trace_ax.axvspan(0, burn_in, color='gray', alpha=0.2, linewidth=0)
  trace_ax.set_ylabel(label)
  if j < 2:
    trace_ax.tick_params(labelbottom=False)

  # Sideways histogram of the pooled chains, sharing the trace's y-axis
  hist_ax.sharey(trace_ax)
  hist_ax.hist(pooled[:, j], bins=40, density=True, orientation='horizontal',
               color='gray', alpha=0.4)
  hist_ax.plot(marginal / trapezoid(marginal, grid), grid, color='black', linewidth=1.5)
  hist_ax.tick_params(labelleft=False)
  hist_ax.set_xticks([])
  for ax in axes[j]:
    ax.spines[['top', 'right']].set_visible(False)

axes[-1, 0].set_xlabel('Iteration')
axes[-1, 1].set_xlabel('Density')
# One shared legend under the title, with thicker proxy lines so the chain colours read clearly
handles = [plt.Line2D([], [], color=color, linewidth=2, label=f'Chain {chain + 1}')
           for chain, color in enumerate(chain_colors)]
handles += [plt.Rectangle((0, 0), 1, 1, color='gray', alpha=0.2, label='Burn-in'),
            plt.Rectangle((0, 0), 1, 1, color='gray', alpha=0.4, label='Pooled samples'),
            plt.Line2D([], [], color='black', linewidth=1.5, label='True marginal')]
fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, 0.95), ncol=6,
           frameon=False, fontsize='small', handlelength=1.5, columnspacing=1)
fig.suptitle('Trace and marginal posterior of each parameter')
plt.tight_layout(rect=(0, 0, 1, 0.92))
plt.show()

#%%
# Bivariate marginals: post-burn-in samples against the true highest-density contours
pairs = [(0, 1), (0, 2), (1, 2)]

fig, axes = plt.subplots(1, 3, figsize=(LINEWIDTH, 4/12*LINEWIDTH))
for ax, (i, k) in zip(axes, pairs):
  ax.plot(x_samples[burn_in:, i], x_samples[burn_in:, k], 'o', color='#2a78d6',
          markersize=2, markeredgewidth=0, alpha=0.3)

  # True bivariate marginal: integrate out the third parameter
  third = 3 - i - k
  bivariate = trapezoid(density, grids[third], axis=third)
  levels = density_levels(bivariate, contour_masses)
  contours = ax.contour(grids[i], grids[k], bivariate.T, levels=levels, colors='black', linewidths=1)
  ax.clabel(contours, fmt={level: f'{mass:.0%}' for level, mass in zip(levels, contour_masses)},
            fontsize='small')

  ax.set_title(f'{labels[k]} vs {labels[i]}')
  ax.set_xlabel(labels[i])
  ax.set_ylabel(labels[k])
  ax.spines[['top', 'right']].set_visible(False)

# Proxy handles, since contour sets have no legend entry of their own
fig.legend(
  handles=[
    plt.Line2D([], [], marker='o', linestyle='', color='#2a78d6', alpha=0.5, label='MCMC samples'),
    plt.Line2D([], [], color='black', linewidth=1, label='True highest-density regions'),
  ],
  loc='upper center', ncol=2, frameon=False,
)
plt.tight_layout(rect=(0, 0, 1, 0.92))
plt.show()

#%%
def predict_leave_one_out(data, i, N=5000):
  # simulate predictive distribution
  x_act, y_act = data[i,:]

  data = np.delete(data, i, axis=0)
  samples = sample_posterior_mh(data, size=N+burn_in)[burn_in:,:]

  ln_a = samples[:,0]
  b = samples[:,1]
  var = samples[:,2]
  eps = scipy.stats.norm.rvs(scale=np.sqrt(var), size=N)

  ln_y_pred = ln_a + b*np.log(x_act) + eps
  y_pred = np.exp(ln_y_pred)

  ci = np.quantile(y_pred, [0.025, 0.975])

  return ci, y_pred, y_act

#%%
ci, y_pred, y_act = predict_leave_one_out(data, 0)
print(f'95% CI: {ci}')

x_1, y_1 = data[0]
fig, ax = plt.subplots()#figsize=(6, 3.5))
ax.axvspan(*ci, color='#2a78d6', alpha=0.15, linewidth=0, label=f'95% CI [{ci[0]:.2f}, {ci[1]:.2f}]')
ax.hist(y_pred, bins=50, density=True, color='#2a78d6', alpha=0.7, label='Predictive samples')
ax.axvline(y_1, color='#eb6834', linewidth=2, label=f'Observed $y_1 = {y_1}$')
ax.set_xlabel('Radius $y_1^*$ (m)')
ax.set_ylabel('Density')
ax.set_title(f'Leave-one-out predictive distribution at $x_1 = {x_1}$ m')
ax.spines[['top', 'right']].set_visible(False)
ax.legend(frameon=False)
plt.tight_layout()
plt.show()

# plt.hist(y_pred, bins=20, density=True)
# plt.axvline(y_act, color='red', label='Observed $y_1$')
# plt.axvline(ci[0], color='orange')
# plt.axvline(ci[1], color='orange')
# plt.legend()

#%%
cis = np.array([predict_leave_one_out(data, i)[0] for i in range(24)])

#%%
# x = data[:,0]
# plt.plot(x, cis[:,0], '--', color='tab:blue', label='Predicted 95% CI')
# plt.plot(x, cis[:,1], '--', color='tab:blue')
# plt.plot(x, data[:,1], 'o', label='Observed data')
# plt.xlabel('Depth (m)')
# plt.ylabel('Radius (m)')

#%%
var_a = (np.log(2) / 1.96)**2
# posterior distribution
def posterior_linear(theta, data):
  x = data[:,0][:,np.newaxis]
  y = data[:,1][:,np.newaxis]
  ln_a, var = theta
  n = data.shape[0]

  # support: 0 < beta < 1 and sigma^2 > 0
  in_support = (var > 0)
  # placeholder variance outside the support, so the sqrt below stays valid
  var = np.where(var > 0, var, 1.0)

  pi_ln_a = np.exp(-ln_a**2 / (2*var_a))
  pi_var = 1/var

  L = np.prod(
    np.exp(-(np.log(y) - ln_a - np.log(x))**2 / (2*var))
      / (y * np.sqrt(2*np.pi*var)),
    axis=0
  )

  return np.where(in_support, pi_ln_a * pi_var * L, 0.0)

def sample_posterior_linear(data, proposal_params=(0.15, 0.46), size=1000,
  init=(0, 0.5)):
  sample = np.empty((size, 2))

  # defaults to the prior means
  sample[0,:] = init

  # ln_a_step = 0.15     # sd of the Gaussian walk on ln(alpha)
  # log_var_step = 0.46  # sd of the Gaussian walk on ln(sigma^2)
  ln_a_step, log_var_step = proposal_params

  # All random numbers are drawn up front: the Gaussian steps of both walks,
  # and the uniforms for the accept/reject decisions
  steps = np.random.normal(scale=[ln_a_step, log_var_step], size=(size-1, 2))
  u = np.random.uniform(size=size-1)

  # The current state's posterior is cached, so each iteration evaluates it once
  current_post = posterior_linear(sample[0,:], data)
  for t in range(size-1):
    ln_a, var = sample[t,:]
    proposal = np.array([ln_a + steps[t,0], var * np.exp(steps[t,1])])
    proposal_post = posterior_linear(proposal, data)

    # The ln(alpha) walk is symmetric, so its term cancels. The log-normal walk on
    # sigma^2 has q(var | var') / q(var' | var) = var' / var, so no pdf calls are needed.
    # Accept with probability min(1, ratio), written without the division.
    if u[t] * current_post < proposal_post * proposal[1] / var:
      sample[t+1,:] = proposal
      current_post = proposal_post
    else:
      sample[t+1,:] = sample[t,:]

  return sample

def predict_leave_one_out_linear(data, i, N=5000):
  # simulate predictive distribution
  x_act, y_act = data[i,:]

  data = np.delete(data, i, axis=0)
  samples = sample_posterior_linear(data, size=N+burn_in)[burn_in:,:]

  ln_a = samples[:,0]
  var = samples[:,1]
  eps = scipy.stats.norm.rvs(scale=np.sqrt(var), size=N)

  ln_y_pred = ln_a + np.log(x_act) + eps
  y_pred = np.exp(ln_y_pred)

  ci = np.quantile(y_pred, [0.025, 0.975])

  return ci, y_pred

#%%
cis_linear = np.array([predict_leave_one_out_linear(data, i)[0] for i in range(24)])

#%%
# x = data[:,0]
# plt.plot(x, cis[:,0], '--', color='tab:blue', label='Predicted 95% CI')
# plt.plot(x, cis[:,1], '--', color='tab:blue')
# plt.plot(x, data[:,1], 'o', label='Observed data')
# plt.xlabel('Depth (m)')
# plt.ylabel('Radius (m)')

x, y = data.T
fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
for ax, intervals, title in zip(axes, [cis, cis_linear], ['Full model', 'Simplified model ($\\beta = 1$)']):
  inside = (intervals[:, 0] <= y) & (y <= intervals[:, 1])
  ax.vlines(x, intervals[:, 0], intervals[:, 1], color='#2a78d6', linewidth=3, alpha=0.5, label='95% predictive CI')
  ax.plot(x[inside], y[inside], 'o', color='black', markersize=5, label='Observed, inside CI')
  ax.plot(x[~inside], y[~inside], 'o', color='#eb6834', markersize=6, label='Observed, outside CI')
  ax.set_title(f'{title}: {inside.sum()}/{len(y)} inside')
  ax.set_xscale('log')
  ax.set_yscale('log')
  ax.set_xlabel('Depth $x$ (m)')
  ax.spines[['top', 'right']].set_visible(False)
axes[0].set_ylabel('Radius $y$ (m)')
axes[0].legend(frameon=False, loc='upper left')
plt.tight_layout()
plt.show()

# %%
