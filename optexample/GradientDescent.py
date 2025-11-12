import numpy as np
import matplotlib.pyplot as plt

def f(x):               # f(x) = (x1-1)^2 + 2*(x2+2)^2
    return (x[0]-1)**2 + 2*(x[1]+2)**2

def grad_f(x):
    return np.array([2*(x[0]-1), 4*(x[1]+2)])

x = np.array([3.0, 3.0])     # initial guess
alpha = 0.5
xs = [x.copy()]
fs = [f(x)]

for k in range(50):
    x = x - alpha * grad_f(x)
    xs.append(x.copy())
    fs.append(f(x))

xs = np.array(xs)

# Create subplot figure
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Left plot: trajectory
axes[0].plot(xs[:,0], xs[:,1], 'o-')
axes[0].set_xlabel('$x_1$ (first coordinate)')
axes[0].set_ylabel('$x_2$ (second coordinate)')
axes[0].set_title('Trajectory of $x$')

# Right plot: cost vs iteration
axes[1].plot(fs, 'o-')
axes[1].set_xlabel('Iteration')
axes[1].set_ylabel('Cost $f(x)$')
axes[1].set_title('Cost Function Decrease')

plt.tight_layout()
plt.show()
