import numpy as np
import matplotlib.pyplot as plt

# 1) Generate input
N = 2000
t = np.linspace(0, 2 * np.pi, N, endpoint=False)
I = np.sin(t) 

# 2) Penalty function
R = 1.0  
v = R * np.maximum(0.0, -I)

# 3) Plot input and output
plt.figure(figsize=(10, 5), dpi=140)

plt.plot(t, I, label='Input I(t) = sin(t)', color='#1f77b4', linewidth=2)
plt.plot(t, v, label='Output v(t) = R·max(0, -I)', color='#d62728', linewidth=2)

# Visual aids
plt.axhline(0, color='black', linewidth=0.8)
plt.title('Penalty Function (Eq. 23) Response to Sine Input')
plt.xlabel('t')
plt.ylabel('Amplitude')
plt.legend()
plt.tight_layout()
plt.show()
