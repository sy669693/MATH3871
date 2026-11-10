#%%
import numpy as np
import scipy
import matplotlib.pyplot as plt
import seaborn as sns

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

#%%
# Trace of each parameter, with its marginal posterior as a sideways histogram
fig, axes = plt.subplots(3, 2, figsize=(LINEWIDTH, 7/10*LINEWIDTH),
                         gridspec_kw=dict(width_ratios=[4, 1], wspace=0.05))
for j, (label, grid, marginal) in enumerate(zip(labels, grids, marginals)):
  trace_ax, hist_ax = axes[j]
  # Traces share iterations; each histogram keeps its own density scale
  trace_ax.sharex(axes[0, 0])
  trace_ax.plot(x_samples[:, j], linewidth=0.6, color='#2a78d6')
  trace_ax.axvspan(0, burn_in, color='gray', alpha=0.2, linewidth=0, label='Burn-in')
  trace_ax.set_ylabel(label)
  if j < 2:
    trace_ax.tick_params(labelbottom=False)

  # Sideways histogram sharing the trace's y-axis
  hist_ax.sharey(trace_ax)
  hist_ax.hist(x_samples[burn_in:, j], bins=40, density=True, orientation='horizontal',
               color='#2a78d6', alpha=0.4, label='MCMC samples')
  hist_ax.plot(marginal / trapezoid(marginal, grid), grid, color='black', linewidth=1.5,
               label='True marginal')
  hist_ax.tick_params(labelleft=False)
  hist_ax.set_xticks([])
  for ax in axes[j]:
    ax.spines[['top', 'right']].set_visible(False)

axes[-1, 0].set_xlabel('Iteration')
axes[-1, 1].set_xlabel('Density')
axes[0, 0].legend(frameon=False, loc='upper right')
axes[0, 1].legend(frameon=False, loc='upper right', fontsize='small')
fig.suptitle('Trace and marginal posterior of each parameter')
plt.tight_layout()
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
  u = scipy.stats.uniform.rvs(size=size)

  # defaults to the prior means
  sample[0,:] = init

  # ln_a_step = 0.03     # sd of the Gaussian walk on ln(alpha)
  # b_var = 0.0004       # variance of the Beta proposal for beta
  # log_var_step = 0.46  # sd of the Gaussian walk on ln(sigma^2)
  ln_a_step, log_var_step = proposal_params

  # Density of proposing theta from given. The ln(alpha) walk is symmetric,
  # so its term cancels in the acceptance ratio and is left out.
  def q(theta, given):
    return scipy.stats.lognorm.pdf(theta[1], s=log_var_step, scale=given[1])

  def sample_q(t):
    ln_a, var = sample[t,:]
    return np.array([
        ln_a + scipy.stats.norm.rvs(scale=ln_a_step),
        var * np.exp(scipy.stats.norm.rvs(scale=log_var_step)),
    ])

  for t in range(size-1):
      proposal = sample_q(t)
      ratio = (
          posterior_linear(proposal, data) * q(sample[t,:], proposal)
          / (posterior_linear(sample[t,:], data) * q(proposal, sample[t,:]))
      )
      accept_prob = min(ratio, 1.0)

      if u[t] < accept_prob:
        sample[t+1,:] = proposal
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
