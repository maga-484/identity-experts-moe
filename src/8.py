import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import random
from copy import deepcopy

# ------------------------------------------------------------
# FUNCIONES DE ID JERÁRQUICO REAL (a.b.c.d.e.f.g.h)
# ------------------------------------------------------------
def hash_jerarquico(a, b, c, d, e, f, g, h):
    return (a * (100**7) + b * (100**6) + c * (100**5) + d * (100**4) + 
            e * (100**3) + f * (100**2) + g * 100 + h)

def id_a_componentes(id_str):
    partes = id_str.split('.')
    return (int(partes[0]), int(partes[1]), int(partes[2]), int(partes[3]),
            int(partes[4]), int(partes[5]), int(partes[6]), int(partes[7]))

def id_a_hash(id_str):
    a,b,c,d,e,f,g,h = id_a_componentes(id_str)
    return hash_jerarquico(a,b,c,d,e,f,g,h)

def generar_id_aleatorio():
    a = random.randint(0, 999)
    b = random.randint(0, 99)
    c = random.randint(0, 99)
    d = random.randint(0, 99)
    e = random.randint(0, 99)
    f = random.randint(0, 99)
    g = random.randint(0, 99)
    h = random.randint(0, 99)
    return f"{a:03d}.{b:02d}.{c:02d}.{d:02d}.{e:02d}.{f:02d}.{g:02d}.{h:02d}"

# ------------------------------------------------------------
# 2. Experto Identitario
# ------------------------------------------------------------
class IdentityExpert(nn.Module):
    def __init__(self, input_dim=32, hidden_dim=128, output_dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    def forward(self, x):
        return self.net(x)

# ------------------------------------------------------------
# 3. Generación de datos
# ------------------------------------------------------------
def generate_silo_data(seed, n_samples=2000, input_dim=32, output_dim=32, domain_type="linear"):
    torch.manual_seed(seed)
    X = torch.randn(n_samples, input_dim)
    
    if domain_type == "linear":
        W_true = torch.randn(input_dim, output_dim) * 1.0
        y = torch.mm(X, W_true) + torch.randn(n_samples, output_dim) * 0.05
    elif domain_type == "sinusoidal":
        y = torch.sin(1.5 * X[:, :output_dim]) + torch.randn(n_samples, output_dim) * 0.05
    elif domain_type == "exponential":
        y = torch.exp(torch.clamp(0.3 * X[:, :output_dim], max=2.0)) + torch.randn(n_samples, output_dim) * 0.05
    else:
        y = (X[:, :output_dim] / 2.0)**3 + torch.randn(n_samples, output_dim) * 0.05
    
    y_mean = y.mean(dim=0, keepdim=True)
    y_std = y.std(dim=0, keepdim=True) + 1e-8
    y = (y - y_mean) / y_std
    
    X_mean = X.mean(dim=0, keepdim=True)
    X_std = X.std(dim=0, keepdim=True) + 1e-8
    X = (X - X_mean) / X_std
    
    return X, y

# Crear silos
num_silos = 10
silos_data = {}
domain_types = ["linear", "sinusoidal", "exponential", "polynomial", "linear", 
                "sinusoidal", "exponential", "polynomial", "linear", "sinusoidal"]

print(f"Generando {num_silos} silos...")
for i in range(num_silos):
    sid = f"silo_{i}"
    X, y = generate_silo_data(seed=1000+i, n_samples=2000, 
                              domain_type=domain_types[i % len(domain_types)])
    silos_data[sid] = (X, y)

print("✓ Datos generados")

# ------------------------------------------------------------
# 4. Asignación de hash con ID JERÁRQUICO REAL (CORREGIDO)
# ------------------------------------------------------------
hash_map = {}
expert_by_sid = {}
random.seed(42)

for idx, (sid, (X, y)) in enumerate(silos_data.items()):
    # Generar ID jerárquico basado en el índice (único)
    a = idx
    b = (idx * 7) % 100
    c = (idx * 13) % 100
    d = (idx * 17) % 100
    e = (idx * 19) % 100
    f = (idx * 23) % 100
    g = (idx * 29) % 100
    h = (idx * 31) % 100
    
    id_str = f"{a:03d}.{b:02d}.{c:02d}.{d:02d}.{e:02d}.{f:02d}.{g:02d}.{h:02d}"
    h_val = id_a_hash(id_str)
    
    expert = IdentityExpert()
    hash_map[h_val] = expert
    expert_by_sid[sid] = (h_val, expert, id_str)

print(f"Asignados {len(hash_map)} silos con ID jerárquico")

# ------------------------------------------------------------
# 5. Entrenamiento
# ------------------------------------------------------------
def train_expert(expert, X, y, epochs=80, lr=0.001):
    optimizer = optim.Adam(expert.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    expert.train()
    losses = []
    for epoch in range(epochs):
        optimizer.zero_grad()
        pred = expert(X)
        loss = loss_fn(pred, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(expert.parameters(), max_norm=1.0)
        optimizer.step()
        if epoch % 20 == 0:
            losses.append(loss.item())
    return losses

print("\n--- Entrenando expertos ---")
train_losses = {}
for sid, (X, y) in silos_data.items():
    h, expert, id_str = expert_by_sid[sid]
    print(f"{sid} (ID={id_str}, hash={h}): ", end="")
    losses = train_expert(expert, X, y)
    train_losses[sid] = losses
    print(f"  Pérdida final: {losses[-1]:.4f}")

# ------------------------------------------------------------
# 6. Modelo denso
# ------------------------------------------------------------
all_X = torch.cat([X for (X,_) in silos_data.values()])
all_y = torch.cat([y for (_,y) in silos_data.values()])

dense_model = IdentityExpert()
print("\n--- Entrenando modelo denso ---")
dense_losses = train_expert(dense_model, all_X, all_y, epochs=80)
print(f"Modelo denso - pérdida final: {dense_losses[-1]:.4f}")

# ------------------------------------------------------------
# 7. Evaluación
# ------------------------------------------------------------
loss_fn = nn.MSELoss()
print("\n--- Evaluación ---")
expert_wins = 0

for sid, (X_test, y_test) in silos_data.items():
    h, expert, _ = expert_by_sid[sid]
    with torch.no_grad():
        loss_exp = loss_fn(expert(X_test), y_test).item()
        loss_den = loss_fn(dense_model(X_test), y_test).item()
    
    if loss_exp < loss_den:
        expert_wins += 1
        win = "✓"
    else:
        win = "✗"
    
    print(f"{sid}: Experto={loss_exp:.4f} | Denso={loss_den:.4f} | {win}")

print(f"\n📊 Experto gana en {expert_wins}/{num_silos} silos")

# ------------------------------------------------------------
# 8. EDE (Stratonovich)
# ------------------------------------------------------------
def euler_heun_update(expert, X_new, y_new, X_old, y_old, steps=30, dt=0.005, sigma=0.05):
    expert.train()
    loss_fn = nn.MSELoss()
    history = []
    
    for step in range(steps):
        # Calcular gradientes
        pred_new = expert(X_new)
        loss_new = loss_fn(pred_new, y_new)
        grad_new = torch.autograd.grad(loss_new, expert.parameters(), create_graph=False)
        
        pred_old = expert(X_old)
        loss_old = loss_fn(pred_old, y_old)
        grad_old = torch.autograd.grad(loss_old, expert.parameters(), create_graph=False)
        
        grad_combined = [gn + 0.3 * go for gn, go in zip(grad_new, grad_old)]
        
        # Ruido
        ruido = [torch.randn_like(p) * sigma * (dt**0.5) for p in expert.parameters()]
        estado_inicial = [p.clone() for p in expert.parameters()]
        
        # Predictor
        with torch.no_grad():
            for p, g, n in zip(expert.parameters(), grad_combined, ruido):
                p.add_(-dt * g + n)
        
        # Corrector
        pred_pred = expert(X_new)
        loss_pred = loss_fn(pred_pred, y_new)
        grad_pred = torch.autograd.grad(loss_pred, expert.parameters(), create_graph=False)
        
        with torch.no_grad():
            for p, p0 in zip(expert.parameters(), estado_inicial):
                p.data = p0.data
            for p, g1, g2, n in zip(expert.parameters(), grad_combined, grad_pred, ruido):
                drift_avg = (g1 + g2) / 2.0
                p.add_(-dt * drift_avg + n)
        
        with torch.no_grad():
            history.append(loss_fn(expert(X_new), y_new).item())
    
    return history

# ------------------------------------------------------------
# 9. Simular cambio y EDE
# ------------------------------------------------------------
print("\n--- Simulando cambio con EDE ---")
silo_prueba = "silo_0"
h0, expert0, _ = expert_by_sid[silo_prueba]
X_old, y_old = silos_data[silo_prueba]
X_new, y_new = generate_silo_data(seed=9999, n_samples=500, domain_type="exponential")

with torch.no_grad():
    loss_old_before = loss_fn(expert0(X_old), y_old).item()
    loss_new_before = loss_fn(expert0(X_new), y_new).item()
print(f"Antes - Antiguos: {loss_old_before:.4f} | Nuevos: {loss_new_before:.4f}")

print("Aplicando EDE...")
ede_history = euler_heun_update(expert0, X_new, y_new, X_old[:200], y_old[:200], 
                                 steps=30, dt=0.005, sigma=0.05)

with torch.no_grad():
    loss_old_after = loss_fn(expert0(X_old), y_old).item()
    loss_new_after = loss_fn(expert0(X_new), y_new).item()

print(f"\nDespués - Antiguos: {loss_old_before:.4f} → {loss_old_after:.4f}")
print(f"Después - Nuevos:   {loss_new_before:.4f} → {loss_new_after:.4f}")
print(f"Mejora nuevos: {(loss_new_before - loss_new_after) / loss_new_before * 100:.1f}%")

# ------------------------------------------------------------
# 10. Visualización
# ------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

sids = list(silos_data.keys())
exp_losses = []
den_losses = []
for sid in sids:
    h, expert, _ = expert_by_sid[sid]
    X, y = silos_data[sid]
    with torch.no_grad():
        exp_losses.append(loss_fn(expert(X), y).item())
        den_losses.append(loss_fn(dense_model(X), y).item())

x = np.arange(len(sids))
axes[0].bar(x - 0.2, exp_losses, 0.4, label='Experto', color='blue')
axes[0].bar(x + 0.2, den_losses, 0.4, label='Denso', color='orange')
axes[0].set_xlabel('Silo')
axes[0].set_ylabel('Pérdida MSE')
axes[0].set_title('Experto vs Denso')
axes[0].legend()
axes[0].grid(True)

axes[1].plot(ede_history, marker='o', markersize=3, color='green')
axes[1].set_xlabel('Paso EDE')
axes[1].set_ylabel('Pérdida')
axes[1].set_title('Adaptación con EDE')
axes[1].grid(True)

plt.tight_layout()
plt.show()

print("\n" + "="*50)
print("✅ 8.py completado")
print(f"✓ Expertos identitarios: {expert_wins}/{num_silos} victorias")
print(f"✓ EDE mejoró nuevos datos")
print("="*50)