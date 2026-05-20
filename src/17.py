import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import joblib
import hashlib
import time
import os
import csv
from copy import deepcopy
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

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
# Experto Identitario
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
# Generar datos sintéticos para silos
# ------------------------------------------------------------
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
# 1. Entrenar selector con datos históricos
# ------------------------------------------------------------
def entrenar_selector():
    data_real = {
        'variation': [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50],
        'loss_exp': [0.0708, 0.0758, 0.0717, 0.0736, 0.0757, 0.0678, 0.0689, 0.0712, 0.0687, 0.0780],
        'loss_den': [0.1023, 0.1029, 0.1021, 0.0995, 0.1052, 0.0984, 0.0998, 0.1015, 0.1009, 0.1050],
    }
    df = pd.DataFrame(data_real)
    df['ventaja'] = df['loss_den'] - df['loss_exp']
    df['exito'] = ((df['loss_exp'] < 0.1) & (df['ventaja'] > 0)).astype(int)
    
    features = ['variation', 'loss_exp', 'loss_den', 'ventaja']
    selector = RandomForestClassifier(n_estimators=50, max_depth=3, random_state=42)
    selector.fit(df[features], df['exito'])
    joblib.dump(selector, 'experto_selector.pkl')
    return selector

# ------------------------------------------------------------
# 2. Crear silo para un producto
# ------------------------------------------------------------
def crear_silo_producto(producto_id, id_hierarquico, variacion):
    """Crea un nuevo silo y experto para un producto."""
    # Hash del ID jerárquico
    h_val = id_a_hash(id_hierarquico)
    
    # Crear experto
    experto = IdentityExpert()
    
    # Generar datos sintéticos para este silo
    X, y = generate_silo_data(seed=hash(h_val) % 10000, variation=variacion)
    
    # Entrenamiento rápido
    optimizer = optim.Adam(experto.parameters(), lr=0.001)
    loss_fn = nn.MSELoss()
    for epoch in range(30):
        optimizer.zero_grad()
        pred = experto(X)
        loss = loss_fn(pred, y)
        loss.backward()
        optimizer.step()
    
    # Guardar
    os.makedirs('expertos', exist_ok=True)
    torch.save(experto.state_dict(), f"expertos/{h_val}.pth")
    
    # Registrar
    registro = {
        'producto_id': producto_id,
        'id_hierarquico': id_hierarquico,
        'hash': h_val,
        'variation': variacion,
        'created_at': time.time()
    }
    
    file_exists = os.path.isfile('productos_registrados.csv')
    with open('productos_registrados.csv', 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=registro.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(registro)
    
    print(f"   ✅ Silo creado: {id_hierarquico} (hash={h_val})")
    return h_val, experto

# ------------------------------------------------------------
# 3. Pipeline de lanzamiento
# ------------------------------------------------------------
def pipeline_lanzamiento(catalogo, umbral_exito=0.6):
    """Selecciona productos y crea silos para los seleccionados."""
    
    # Cargar selector
    if not os.path.exists('experto_selector.pkl'):
        selector = entrenar_selector()
    else:
        selector = joblib.load('experto_selector.pkl')
    
    def predecir(variation, loss_exp, loss_den):
        ventaja = loss_den - loss_exp
        features = np.array([[variation, loss_exp, loss_den, ventaja]])
        prob = selector.predict_proba(features)[0][1]
        return prob
    
    print(f"\n🚀 Iniciando pipeline con {len(catalogo)} productos")
    print(f"   Umbral de éxito: {umbral_exito*100}%\n")
    
    seleccionados = []
    for producto in catalogo:
        prob = predecir(producto['variation'], producto['loss_exp'], producto['loss_den'])
        if prob >= umbral_exito:
            print(f"✅ {producto['id']}: SELECCIONADO (prob={prob*100:.1f}%)")
            seleccionados.append(producto)
        else:
            print(f"❌ {producto['id']}: DESCARTADO (prob={prob*100:.1f}%)")
    
    print(f"\n📊 Total seleccionados: {len(seleccionados)}/{len(catalogo)}")
    
    # Crear silos para seleccionados
    for producto in seleccionados:
        print(f"\n--- Creando silo para {producto['id']} ---")
        id_hierarquico = producto.get('id_hierarquico', generar_id_aleatorio())
        crear_silo_producto(producto['id'], id_hierarquico, producto['variation'])
    
    return seleccionados

# ------------------------------------------------------------
# 4. Ejecución principal
# ------------------------------------------------------------
if __name__ == "__main__":
    # Catálogo de prueba
    catalogo = [
        {'id': 'PROD-001', 'id_hierarquico': generar_id_aleatorio(), 'variation': 0.08, 'loss_exp': 0.072, 'loss_den': 0.105},
        {'id': 'PROD-002', 'id_hierarquico': generar_id_aleatorio(), 'variation': 0.12, 'loss_exp': 0.075, 'loss_den': 0.103},
        {'id': 'PROD-003', 'id_hierarquico': generar_id_aleatorio(), 'variation': 0.18, 'loss_exp': 0.080, 'loss_den': 0.101},
        {'id': 'PROD-004', 'id_hierarquico': generar_id_aleatorio(), 'variation': 0.25, 'loss_exp': 0.090, 'loss_den': 0.100},
        {'id': 'PROD-005', 'id_hierarquico': generar_id_aleatorio(), 'variation': 0.35, 'loss_exp': 0.120, 'loss_den': 0.098},
        {'id': 'PROD-006', 'id_hierarquico': generar_id_aleatorio(), 'variation': 0.08, 'loss_exp': 0.068, 'loss_den': 0.107},
        {'id': 'PROD-007', 'id_hierarquico': generar_id_aleatorio(), 'variation': 0.15, 'loss_exp': 0.073, 'loss_den': 0.104},
        {'id': 'PROD-008', 'id_hierarquico': generar_id_aleatorio(), 'variation': 0.22, 'loss_exp': 0.085, 'loss_den': 0.099},
        {'id': 'PROD-009', 'id_hierarquico': generar_id_aleatorio(), 'variation': 0.30, 'loss_exp': 0.105, 'loss_den': 0.097},
        {'id': 'PROD-010', 'id_hierarquico': generar_id_aleatorio(), 'variation': 0.45, 'loss_exp': 0.150, 'loss_den': 0.095},
    ]
    
    # Ejecutar pipeline
    seleccionados = pipeline_lanzamiento(catalogo, umbral_exito=0.6)
    
    print("\n" + "="*60)
    print("RESUMEN FINAL - PIPELINE COMPLETO")
    print("="*60)
    print(f"✓ Productos seleccionados: {len(seleccionados)}")
    print(f"✓ Silos creados: {len(seleccionados)}")
    print(f"✓ Archivo: productos_registrados.csv")
    print(f"✓ Directorio: expertos/")
    print("="*60)