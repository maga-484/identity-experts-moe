import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib
import random

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
# 1. Generar datos balanceados (incluyendo fracasos)
# ------------------------------------------------------------
# Datos de éxitos (de 13.py)
exitos = [
    (0.05, 0.0708, 0.1023, "000.00.00.00.00.00.00.00"),
    (0.10, 0.0758, 0.1029, "001.07.13.17.19.23.29.31"),
    (0.15, 0.0717, 0.1021, "002.14.26.34.38.46.58.62"),
    (0.20, 0.0736, 0.0995, "003.21.39.51.57.69.87.93"),
    (0.25, 0.0757, 0.1052, "004.28.52.68.76.92.16.24"),
    (0.30, 0.0678, 0.0984, "005.35.65.85.95.15.45.55"),
    (0.35, 0.0689, 0.0998, "006.42.78.02.14.38.74.86"),
    (0.40, 0.0712, 0.1015, "007.49.91.19.33.61.03.17"),
    (0.45, 0.0687, 0.1009, "008.56.04.36.52.84.32.48"),
    (0.50, 0.0780, 0.1050, "009.63.17.53.71.07.61.79"),
]

# Datos de fracasos (simulados con pérdida alta)
fracasos = [
    (0.08, 0.250, 0.150, "100.01.02.03.04.05.06.07"),
    (0.12, 0.300, 0.160, "100.01.02.03.04.05.06.08"),
    (0.18, 0.280, 0.155, "100.01.02.03.04.05.06.09"),
    (0.22, 0.350, 0.170, "100.01.02.03.04.05.06.10"),
    (0.28, 0.400, 0.180, "100.01.02.03.04.05.06.11"),
    (0.32, 0.320, 0.165, "100.01.02.03.04.05.06.12"),
    (0.38, 0.380, 0.175, "100.01.02.03.04.05.06.13"),
    (0.42, 0.450, 0.190, "100.01.02.03.04.05.06.14"),
    (0.48, 0.420, 0.185, "100.01.02.03.04.05.06.15"),
    (0.52, 0.500, 0.200, "100.01.02.03.04.05.06.16"),
]

data_rows = []
for v, le, ld, hid in exitos:
    data_rows.append({'variation': v, 'loss_exp': le, 'loss_den': ld, 'id_hierarquico': hid, 'exito': 1})
for v, le, ld, hid in fracasos:
    data_rows.append({'variation': v, 'loss_exp': le, 'loss_den': ld, 'id_hierarquico': hid, 'exito': 0})

df = pd.DataFrame(data_rows)
df['ventaja'] = df['loss_den'] - df['loss_exp']

print("📊 Dataset balanceado (éxitos + fracasos):")
print(df[['id_hierarquico', 'variation', 'loss_exp', 'loss_den', 'ventaja', 'exito']])
print(f"\nDistribución: {df['exito'].sum()} éxitos, {len(df)-df['exito'].sum()} fracasos")

# ------------------------------------------------------------
# 2. Calcular hash para cada silo
# ------------------------------------------------------------
df['hash'] = df['id_hierarquico'].apply(id_a_hash)
print(f"\nHashes generados (primeros 5):")
for _, row in df.head().iterrows():
    print(f"  {row['id_hierarquico']} -> {row['hash']}")

# ------------------------------------------------------------
# 3. Entrenar selector
# ------------------------------------------------------------
features = ['variation', 'loss_exp', 'loss_den', 'ventaja']
X = df[features]
y = df['exito']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

selector = RandomForestClassifier(n_estimators=50, max_depth=3, random_state=42)
selector.fit(X_train, y_train)

y_pred = selector.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"\n🎯 Precisión del selector: {accuracy*100:.1f}%")
print("\n📋 Reporte de clasificación:")
print(classification_report(y_test, y_pred))

# ------------------------------------------------------------
# 4. Guardar modelo
# ------------------------------------------------------------
joblib.dump(selector, 'experto_selector.pkl')
print("\n✅ Experto Selector guardado como 'experto_selector.pkl'")

# ------------------------------------------------------------
# 5. Función para predecir nuevos productos (CORREGIDA)
# ------------------------------------------------------------
def predecir_producto(variation, loss_exp_estimado, loss_den_estimado):
    """Predice probabilidad de éxito para un nuevo producto."""
    ventaja = loss_den_estimado - loss_exp_estimado
    features = np.array([[variation, loss_exp_estimado, loss_den_estimado, ventaja]])
    
    probas = selector.predict_proba(features)
    
    # Si solo hay una clase (todos éxitos o todos fracasos)
    if probas.shape[1] == 1:
        # Si la única clase es éxito (1), devolver 1.0; si es fracaso (0), devolver 0.0
        clases = selector.classes_
        return 1.0 if clases[0] == 1 else 0.0
    else:
        return probas[0][1]  # Probabilidad de éxito (clase 1)

# Probar con un producto nuevo
nuevo_id = generar_id_aleatorio()
prob = predecir_producto(0.12, 0.072, 0.103)
print(f"\n🔮 Nuevo producto ID: {nuevo_id}")
print(f"   Probabilidad de éxito: {prob*100:.1f}%")

# Probar con un producto que debería fracasar
prob_fail = predecir_producto(0.30, 0.250, 0.150)
print(f"🔮 Producto con alta pérdida: prob={prob_fail*100:.1f}%")

# ------------------------------------------------------------
# 6. Simular selección de catálogo
# ------------------------------------------------------------
catalogo_prueba = [
    {'id': generar_id_aleatorio(), 'variation': 0.08, 'loss_exp': 0.072, 'loss_den': 0.105},
    {'id': generar_id_aleatorio(), 'variation': 0.12, 'loss_exp': 0.075, 'loss_den': 0.103},
    {'id': generar_id_aleatorio(), 'variation': 0.18, 'loss_exp': 0.080, 'loss_den': 0.101},
    {'id': generar_id_aleatorio(), 'variation': 0.25, 'loss_exp': 0.090, 'loss_den': 0.100},
    {'id': generar_id_aleatorio(), 'variation': 0.35, 'loss_exp': 0.120, 'loss_den': 0.098},
]

print("\n" + "="*60)
print("SELECCIÓN DE PRODUCTOS DEL CATÁLOGO")
print("="*60)

seleccionados = []
for producto in catalogo_prueba:
    prob = predecir_producto(producto['variation'], producto['loss_exp'], producto['loss_den'])
    if prob > 0.6:
        seleccionados.append(producto)
        print(f"✅ {producto['id']}: SELECCIONADO (prob={prob*100:.1f}%)")
    else:
        print(f"❌ {producto['id']}: DESCARTADO (prob={prob*100:.1f}%)")

print(f"\n📊 Total seleccionados: {len(seleccionados)}/{len(catalogo_prueba)}")
print("="*60)