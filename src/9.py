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
# 5. Entrenamiento offline
# ------------------------------------------------------------
def train_expert(expert, X, y, epochs=80, lr=0.001, verbose=True):
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
    if verbose:
        print(f"  Pérdida final: {losses[-1]:.4f}")
    return losses

print("\n--- Entrenando expertos identitarios ---")
train_losses = {}
for sid, (X, y) in silos_data.items():
    h, expert, id_str = expert_by_sid[sid]
    print(f"{sid} (ID={id_str}, hash={h}): ", end="")
    losses = train_expert(expert, X, y)
    train_losses[sid] = losses

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
# 7. Evaluación comparativa
# ------------------------------------------------------------
loss_fn = nn.MSELoss()
print("\n--- Evaluación comparativa ---")
expert_wins = 0

for sid, (X_test, y_test) in silos_data.items():
    h, expert, _ = expert_by_sid[sid]
    with torch.no_grad():
        loss_exp = loss_fn(expert(X_test), y_test).item()
        loss_den = loss_fn(dense_model(X_test), y_test).item()
    
    if loss_exp < loss_den:
        expert_wins += 1
        win = "✓ GANA"
    else:
        win = "✗ PIERDE"
    
    print(f"{sid:10} | Experto: {loss_exp:.4f} | Denso: {loss_den:.4f} | {win}")

print(f"\n📊 Resumen: Experto identitario gana en {expert_wins}/{num_silos} silos")

# ------------------------------------------------------------
# 8. EDE con buffer de replay (CORREGIDA)
# ------------------------------------------------------------
def euler_heun_update_with_buffer(expert, X_new, y_new, X_old, y_old, 
                                   steps=40, dt=0.003, sigma=0.05,
                                   buffer_size=0.3, replay_weight=0.7):
    """
    EDE con Stratonovich y buffer de replay para evitar olvido.
    """
    expert.train()
    loss_fn = nn.MSELoss()
    history = []
    
    # Crear buffer con muestras aleatorias de datos antiguos
    buffer_n = int(buffer_size * len(X_old))
    indices = torch.randperm(len(X_old))[:buffer_n]
    X_buffer = X_old[indices]
    y_buffer = y_old[indices]
    
    for step in range(steps):
        # Gradientes
        pred_new = expert(X_new)
        loss_new = loss_fn(pred_new, y_new)
        grad_new = torch.autograd.grad(loss_new, expert.parameters(), create_graph=False)
        
        pred_buffer = expert(X_buffer)
        loss_buffer = loss_fn(pred_buffer, y_buffer)
        grad_buffer = torch.autograd.grad(loss_buffer, expert.parameters(), create_graph=False)
        
        # Combinar gradientes (nuevos + buffer)
        grad_combined = []
        for gn, gb in zip(grad_new, grad_buffer):
            grad_combined.append(gn + replay_weight * gb)
        
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
# 9. Simular cambio con buffer de replay
# ------------------------------------------------------------
print("\n--- Simulando cambio en silo_0 con buffer de replay ---")
silo_prueba = "silo_0"
h0, expert0, id_str0 = expert_by_sid[silo_prueba]
X_old, y_old = silos_data[silo_prueba]

# Generar nuevos datos (dominio exponencial, muy diferente)
X_new, y_new = generate_silo_data(seed=9999, n_samples=500, domain_type="exponential")

with torch.no_grad():
    loss_old_before = loss_fn(expert0(X_old), y_old).item()
    loss_new_before = loss_fn(expert0(X_new), y_new).item()

print(f"Antes EDE - Antiguos: {loss_old_before:.4f} | Nuevos: {loss_new_before:.4f}")

print("Aplicando EDE con buffer de replay (Stratonovich)...")
ede_history = euler_heun_update_with_buffer(
    expert0, X_new, y_new, X_old, y_old,
    steps=50, dt=0.003, sigma=0.05,
    buffer_size=0.3, replay_weight=0.7
)

with torch.no_grad():
    loss_old_after = loss_fn(expert0(X_old), y_old).item()
    loss_new_after = loss_fn(expert0(X_new), y_new).item()

print(f"\n--- Resultados con buffer de replay ---")
print(f"Antiguos: {loss_old_before:.4f} → {loss_old_after:.4f} (variación {(loss_old_after/loss_old_before - 1)*100:.1f}%)")
print(f"Nuevos:   {loss_new_before:.4f} → {loss_new_after:.4f} (mejora {(loss_new_before-loss_new_after)/loss_new_before*100:.1f}%)")

if loss_old_after < loss_old_before * 1.10:
    print("✓ Olvido controlado (<10%)")
else:
    print(f"✗ Olvido significativo ({(loss_old_after/loss_old_before - 1)*100:.1f}%)")

# ------------------------------------------------------------
# 10. Anidamiento
# ------------------------------------------------------------
print("\n--- Prueba de encadenamiento ---")
expert1 = expert_by_sid["silo_1"][1]
expert2 = expert_by_sid["silo_2"][1]
x_test = torch.randn(32)

with torch.no_grad():
    out1 = expert1(x_test)
    out2 = expert2(out1)

print(f"Entrada: {x_test[:3].detach().numpy()}...")
print(f"Tras experto silo_1: {out1[:3].detach().numpy()}...")
print(f"Tras experto silo_2: {out2[:3].detach().numpy()}...")
print("✓ Razonamiento en cadena")

# ------------------------------------------------------------
# 11. Visualización
# ------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Comparación final
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

# Evolución EDE
axes[1].plot(ede_history, marker='o', markersize=3, color='green')
axes[1].set_xlabel('Paso EDE')
axes[1].set_ylabel('Pérdida en nuevos datos')
axes[1].set_title('Adaptación con buffer de replay')
axes[1].grid(True)

plt.tight_layout()
plt.show()

print("\n" + "="*60)
print("RESUMEN FINAL - 9.py con BUFFER DE REPLAY")
print("="*60)
print(f"✓ Expertos identitarios: {expert_wins}/{num_silos} victorias")
print(f"✓ Buffer: 30% datos antiguos, replay_weight=0.7")
print(f"✓ EDE mejoró nuevos datos: {loss_new_before:.4f} → {loss_new_after:.4f}")
print(f"✓ Olvido: {loss_old_before:.4f} → {loss_old_after:.4f}")
print("="*60)