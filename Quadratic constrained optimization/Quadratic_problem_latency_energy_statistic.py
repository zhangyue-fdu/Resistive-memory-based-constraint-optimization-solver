import torch
import time
import pynvml
from typing import Callable, List

torch.set_printoptions(precision=4)

# === 1.2 Numerical simulation of ODEs ===
def runge_kutta_4(f: Callable, x: torch.Tensor, dt: float):
    k1 = f(x)
    k2 = f(x + dt/2 * k1)
    k3 = f(x + dt/2 * k2)
    k4 = f(x + dt * k3)
    s = (k1 + 2*k2 + 2*k3 + k4)/6
    return x + s * dt

def euler(f: Callable, x: torch.Tensor, dt: float):
    s = f(x)
    return x + s * dt

# === 1.3 Differentiable function wrapper ===
class Differentiable:
    def __init__(self, f: Callable):
        self.f = f

    def __call__(self, x: torch.Tensor, d=False):
        if d:
            x.requires_grad_(True)
            y = self.f(x)
            y.backward(torch.ones_like(y))
            grad = x.grad.clone()
            x.requires_grad_(False)
            return grad
        else:
            return self.f(x)

# === 2. Optimization solver ===
class Minimization:
    def __init__(self, f: Differentiable, gs: List[Differentiable],
                 tau=1.0, penalty_rate=1.0, device='cuda'):
        self.device = device
        self.f = f
        self.gs = gs
        self.penalty_rate = penalty_rate
        self.tau = tau

    def penalty(self, x):
        return torch.maximum(x, torch.tensor(0.0, device=self.device)) ** 2

    def sum_penalty(self, x):
        return sum(self.penalty(g(x)) for g in self.gs)

    def energy(self, x):
        return self.f(x) + self.penalty_rate * self.sum_penalty(x)

    def dynamics(self, x):
        x.requires_grad_(True)
        e = self.energy(x)
        e.backward()
        dx = -self.tau * x.grad
        x.requires_grad_(False)
        return dx

    def iterate(self, x0, dt=1e-4, T=0.2, simulator=runge_kutta_4, pr=None):
        if pr is not None:
            self.penalty_rate = pr

        x = x0.clone().to(self.device)
        t = 0.0
        k = 0
        while t < T:
            with torch.no_grad():
                p1 = self.penalty(self.gs[0](x)).item()
                p2 = self.penalty(self.gs[1](x)).item()
                p3 = self.penalty(self.gs[2](x)).item()
                p4 = self.penalty(self.gs[3](x)).item()
                p = self.sum_penalty(x).item()
                e = self.energy(x).item()

            xnext = simulator(self.dynamics, x, dt)
            
            yield (
                t,
                x.detach().cpu().numpy(),
                e,
                p,
                p1, p2, p3, p4,
                xnext.detach().cpu().numpy()
            )

            x = xnext
            t = k * dt
            k += 1

# === 3.1 Problem definition ===
f = Differentiable(
    lambda x: (
        x[0] * 0.5 +
        0.5 * (5 * x[0]**2 + 8 * x[1]**2 + 4 * x[2]**2) -
        4 * x[0] * x[1] - 4 * x[1] * x[2]
    )
)

g1 = Differentiable(lambda x: 1 - x[0] - x[1] - x[2])
g2 = Differentiable(lambda x: -x[0])
g3 = Differentiable(lambda x: -x[1])
g4 = Differentiable(lambda x: -x[2])

m = Minimization(f, [g1, g2, g3, g4], penalty_rate=2000.0, device='cuda')

# === 3.2 Solver (concise version) ===
def solve(*args, **kwargs):
     # Power measurements
    power_readings = []
    
    # Initialize NVML
    try:
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    except pynvml.NVMLError as err:
        print(f"Failed to initialize NVML: {err}")
        handle = None
    
    start_time = time.time()
    
    # Iteration loop
    for k, r in enumerate(m.iterate(*args, **kwargs)):
        t, x, e, p, p1, p2, p3, p4, xnext = r
        
        # Record power usage
        if handle is not None:
            try:
                power_mW = pynvml.nvmlDeviceGetPowerUsage(handle)
                power_W = power_mW / 1000.0
                current_time = time.time() - start_time
                power_readings.append((current_time, power_W))
            except pynvml.NVMLError as err:
                print(f"Failed to read power: {err}")
        
        # Console output
        print(f"[Step {k}] t={t:.6f}, "
              f"x=[{x[0]:.4f}, {x[1]:.4f}, {x[2]:.4f}], "
              f"Energy={e:.4f}, Penalty={p:.4f} "
              f"(P1={p1:.2f}, P2={p2:.2f}, P3={p3:.2f}, P4={p4:.2f})")

    # Energy statistics
    total_energy = 0.0
    if len(power_readings) > 1:
        for i in range(1, len(power_readings)):
            t1, p1 = power_readings[i-1]
            t2, p2 = power_readings[i]
            delta_t = t2 - t1
            avg_power = (p1 + p2) / 2
            total_energy += avg_power * delta_t
    
    # Shut down NVML
    if handle is not None:
        pynvml.nvmlShutdown()
    
    # Final report
    total_time = time.time() - start_time
    avg_power = total_energy / total_time if total_time > 0 else 0
    
    print("\n=== Final state ===")
    print(f"Final solution: x = {xnext.round(4)}")
    print(f"Total runtime: {total_time:.4f} seconds")
    print(f"Total energy consumption: {total_energy:.4f} Joules")
    print(f"Average power: {avg_power:.2f} Watts")

# === Main ===
if __name__ == "__main__":
    x0 = torch.rand(3, device='cuda', dtype=torch.float64)
    solve(x0=x0, dt=1e-4, T=0.0001, simulator=runge_kutta_4)