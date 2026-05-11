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

def generate_silo_data(seed, n_samples=500, input_dim=32, output_dim=32, variation=0.1):
    torch.manual_seed(seed)
    X = torch.randn(n_samples, input_dim)
    torch.manual_seed(42)
    W_base = torch.randn(input_dim, output_dim) * 0.8
    torch.manual_seed(seed)
    W_variation = torch.randn(input_dim, output_dim) * variation
    W = W_base + W_variation
    y = torch.mm(X, W) + torch.randn(n_samples, output_dim) * 0.05
    y_mean = y.mean(dim=0, keepdim=True)
    y_std = y.std(dim=0, keepdim=True) + 1e-8
    y = (y - y_mean) / y_std
    X_mean = X.mean(dim=0, keepdim=True)
    X_std = X.std(dim=0, keepdim=True) + 1e-8
    X = (X - X_mean) / X_std
    return X, y

# ------------------------------------------------------------
# Crear silos
# ------------------------------------------------------------
num_silos = 100
silos_data = {}
print(f"Generando {num_silos} silos...")
for i in range(num_silos):
    sid = f"silo_{i}"
    variation = 0.05 + (i / num_silos) * 0.1
    X, y = generate_silo_data(seed=1000+i, n_samples=500, variation=variation)
    silos_data[sid] = (X, y)

# ------------------------------------------------------------
# Asignación de hash con ID JERÁRQUICO REAL
# ------------------------------------------------------------
hash_map = {}
expert_by_sid = {}
random.seed(42)

for idx, (sid, (X, y)) in enumerate(silos_data.items()):
    a = idx % 1000
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

print(f"Asignados {len(hash_map)} hashes únicos")

# ------------------------------------------------------------
# Entrenamiento
# ------------------------------------------------------------
def train_expert(expert, X, y, epochs=80, lr=0.001):
    optimizer = optim.Adam(expert.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    expert.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        pred = expert(X)
        loss = loss_fn(pred, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(expert.parameters(), max_norm=1.0)
        optimizer.step()
    return expert

print("\n--- Entrenando expertos ---")
for i in range(10):
    sid = f"silo_{i}"
    h, expert, id_str = expert_by_sid[sid]
    X, y = silos_data[sid]
    train_expert(expert, X, y)
    print(f"{sid} (hash={h}, ID={id_str}): entrenado")

# ------------------------------------------------------------
# Modelo denso
# ------------------------------------------------------------
all_X = torch.cat([silos_data[f"silo_{i}"][0] for i in range(10)])
all_y = torch.cat([silos_data[f"silo_{i}"][1] for i in range(10)])
dense_model = IdentityExpert()
train_expert(dense_model, all_X, all_y, epochs=80)
print("Modelo denso entrenado")

# ------------------------------------------------------------
# Evaluación
# ------------------------------------------------------------
loss_fn = nn.MSELoss()
print("\n--- Evaluación ---")
expert_wins = 0
for i in range(10):
    sid = f"silo_{i}"
    h, expert, _ = expert_by_sid[sid]
    X, y = silos_data[sid]
    with torch.no_grad():
        loss_exp = loss_fn(expert(X), y).item()
        loss_den = loss_fn(dense_model(X), y).item()
    if loss_exp < loss_den:
        expert_wins += 1
        print(f"{sid}: Experto={loss_exp:.4f} | Denso={loss_den:.4f} | ✓")
    else:
        print(f"{sid}: Experto={loss_exp:.4f} | Denso={loss_den:.4f} | ✗")

print(f"\n📊 Experto gana en {expert_wins}/10 silos")

# ------------------------------------------------------------
# NUEVO: EDE con GEODÉSICA NEGATIVA (Gradiente Natural Riemanniano)
# ------------------------------------------------------------
def euler_heun_geodesic_negative(expert, X_new, y_new, X_old, y_old,
                                   steps=50, dt=0.003, sigma=0.03,
                                   natural_gradient=True):
    """
    EDE con GEODÉSICA NEGATIVA usando gradiente natural (inversa de Fisher)
    """
    expert.train()
    loss_fn = nn.MSELoss()
    history_new = []
    history_old = []
    
    print("  Calculando tensor métrico (Fisher)...")
    fisher = []
    with torch.no_grad():
        for p in expert.parameters():
            fisher.append(torch.zeros_like(p))
    
    batch_size = min(32, len(X_old))
    indices = torch.randperm(len(X_old))[:batch_size]
    X_fisher = X_old[indices]
    y_fisher = y_old[indices]
    
    for _ in range(20):  # más muestras para mejor Fisher
        expert.zero_grad()
        pred = expert(X_fisher)
        loss = loss_fn(pred, y_fisher)
        loss.backward()
        for idx, p in enumerate(expert.parameters()):
            if p.grad is not None:
                fisher[idx] += p.grad.data ** 2 / 20.0
    
    # Pequeña regularización para invertibilidad
    for idx, f in enumerate(fisher):
        fisher[idx] = f + 0.01
    
    for step in range(steps):
        # Gradiente Euclidiano nuevo
        pred_new = expert(X_new)
        loss_new = loss_fn(pred_new, y_new)
        grad_euclidiano = torch.autograd.grad(loss_new, expert.parameters(), create_graph=False)
        
        # Gradiente natural (geodésica): F^{-1} * ∇L
        if natural_gradient:
            grad_geodesico = []
            for g, f in zip(grad_euclidiano, fisher):
                grad_geodesico.append(g / f)  # inversa diagonal aproximada
        else:
            grad_geodesico = grad_euclidiano
        
        # Gradiente EWC (memoria)
        grad_ewc = []
        for idx, (p, f) in enumerate(zip(expert.parameters(), fisher)):
            grad_ewc.append(25.0 * f * (p - p.data))
        
        # Gradiente antiguos (replay)
        pred_old = expert(X_old[:batch_size])
        loss_old = loss_fn(pred_old, y_old[:batch_size])
        grad_old = torch.autograd.grad(loss_old, expert.parameters(), create_graph=False)
        
        # Combinación: geodésica natural + EWC + replay
        grad_combined = []
        for gg, ge, go in zip(grad_geodesico, grad_ewc, grad_old):
            grad_combined.append(gg + 0.3 * go + ge)
        
        # Ruido Riemanniano (escalado por la métrica)
        ruido = []
        for p, f in zip(expert.parameters(), fisher):
            std = sigma * (dt**0.5) / torch.sqrt(f + 1e-8)
            ruido.append(torch.randn_like(p) * std)
        
        estado_inicial = [p.clone() for p in expert.parameters()]
        
        # Predictor con geodésica
        with torch.no_grad():
            for p, g, n in zip(expert.parameters(), grad_combined, ruido):
                p.add_(-dt * g + n)
        
        # Corrector
        pred_pred = expert(X_new)
        loss_pred = loss_fn(pred_pred, y_new)
        grad_pred = torch.autograd.grad(loss_pred, expert.parameters(), create_graph=False)
        
        if natural_gradient:
            grad_pred_geodesico = [g / f for g, f in zip(grad_pred, fisher)]
        else:
            grad_pred_geodesico = grad_pred
        
        with torch.no_grad():
            for p, p0 in zip(expert.parameters(), estado_inicial):
                p.data = p0.data
            for p, g1, g2, n in zip(expert.parameters(), grad_combined, grad_pred_geodesico, ruido):
                drift_avg = (g1 + g2) / 2.0
                p.add_(-dt * drift_avg + n)
        
        with torch.no_grad():
            history_new.append(loss_fn(expert(X_new), y_new).item())
            history_old.append(loss_fn(expert(X_old[:batch_size]), y_old[:batch_size]).item())
    
    return history_new, history_old

# ------------------------------------------------------------
# Prueba con GEODÉSICA NEGATIVA
# ------------------------------------------------------------
print("\n" + "="*60)
print("PRUEBA CON GEODÉSICA NEGATIVA (GRADIENTE NATURAL)")
print("="*60)

silo_prueba = "silo_0"
h0, expert0, _ = expert_by_sid[silo_prueba]
X_old, y_old = silos_data[silo_prueba]
X_new, y_new = generate_silo_data(seed=9999, n_samples=500, variation=0.3)

with torch.no_grad():
    loss_old_before = loss_fn(expert0(X_old), y_old).item()
    loss_new_before = loss_fn(expert0(X_new), y_new).item()
print(f"Antes - Antiguos: {loss_old_before:.4f} | Nuevos: {loss_new_before:.4f}")

print("\nAplicando EDE con GEODÉSICA NEGATIVA...")
history_new, history_old = euler_heun_geodesic_negative(
    expert0, X_new, y_new, X_old, y_old,
    steps=50, dt=0.003, sigma=0.03,
    natural_gradient=True
)

with torch.no_grad():
    loss_old_after = loss_fn(expert0(X_old), y_old).item()
    loss_new_after = loss_fn(expert0(X_new), y_new).item()

print(f"\n--- Resultados con Geodésica Negativa ---")
print(f"Antiguos: {loss_old_before:.4f} → {loss_old_after:.4f} (variación {(loss_old_after/loss_old_before - 1)*100:.1f}%)")
print(f"Nuevos:   {loss_new_before:.4f} → {loss_new_after:.4f} (mejora {(loss_new_before-loss_new_after)/loss_new_before*100:.1f}%)")

if loss_old_after < loss_old_before * 1.1:
    print("✓ Olvido controlado (<10%)")
elif loss_old_after < loss_old_before * 1.2:
    print("⚠️ Olvido moderado (10-20%)")
else:
    print("✗ Olvido significativo (>20%)")

if loss_new_after < loss_new_before:
    print(f"✓ Mejora real en nuevos datos: {(loss_new_before-loss_new_after)/loss_new_before*100:.1f}%")
else:
    print(f"✗ No hubo mejora (diferencia: {(loss_new_after-loss_new_before)/loss_new_before*100:.1f}%)")

# ------------------------------------------------------------
# Visualización
# ------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].plot(history_new, marker='o', markersize=3, color='green', label='Nuevos datos')
axes[0].plot(history_old, marker='s', markersize=3, color='blue', label='Antiguos (replay)')
axes[0].set_xlabel('Paso EDE')
axes[0].set_ylabel('Pérdida MSE')
axes[0].set_title('Evolución con GEODÉSICA NEGATIVA')
axes[0].legend()
axes[0].grid(True)

axes[1].bar(['Antiguos', 'Nuevos'], 
            [loss_old_before, loss_new_before], width=0.35, label='Antes', color='red', alpha=0.7)
axes[1].bar(['Antiguos', 'Nuevos'], 
            [loss_old_after, loss_new_after], width=0.35, label='Después', color='green', alpha=0.7)
axes[1].set_ylabel('Pérdida MSE')
axes[1].set_title('Efecto de la Geodésica Negativa')
axes[1].legend()
axes[1].grid(True)

plt.tight_layout()
plt.show()

print("\n" + "="*60)
print("RESUMEN - 11B.py (GEODÉSICA NEGATIVA)")
print("="*60)
print(f"✓ Gradiente natural (F^-1 * ∇L)")
print(f"✓ Movimiento siguiendo la curvatura del espacio de parámetros")
print(f"✓ Antiguos: {loss_old_before:.4f} → {loss_old_after:.4f}")
print(f"✓ Nuevos:   {loss_new_before:.4f} → {loss_new_after:.4f}")
print("="*60)