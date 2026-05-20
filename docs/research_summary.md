
markdown
# Identity Experts MoE

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org)

**Enrutamiento hash O(1) para Mixture of Experts escalable con ID jerárquico `a.b.c.d.e.f.g.h`**

---

## 🚀 ¿Qué resuelve este proyecto?

En sistemas con **millones de entidades identificables** (productos, experimentos, usuarios, envíos, transacciones):

| Problema | Consecuencia |
|----------|--------------|
| Un modelo denso único para todo | Caro de entrenar, lento de actualizar |
| Los cambios en una entidad | Requieren reentrenar el modelo completo |
| La especialización "emerge" débilmente | O no emerge, o colapsa |

**Este proyecto implementa una alternativa radical:**  
*Un experto tiny por ID, enrutado mediante tabla hash O(1) con ID jerárquico `a.b.c.d.e.f.g.h`.*

---

## 🗺️ El origen: ID jerárquico

La clave de este sistema es el **identificador jerárquico** de 8 niveles:
a.b.c.d.e.f.g.h

text

| Componente | Rango | Opciones | Ejemplo |
|------------|-------|----------|---------|
| `a` | 000-999 | 1000 | provincia, categoría principal |
| `b` a `h` | 00-99 | 100 c/u | subcategorías, municipios |

**Espacio total:** `1000 × 100^7 = 10^17` (100 mil billones)

**Colisiones:** Prácticamente imposibles. Con 100 silos usas el 0.0000000000001% del espacio.

---

## 📊 Resultados clave (ejecución real)

| Métrica | Resultado |
|---------|-----------|
| **Expertos vs modelo denso** | **10/10 victorias** |
| **Ventaja promedio** | **~29%** (0.0722 vs 0.1018 MSE) |
| **Enrutamiento** | O(1) mediante hash jerárquico (espacio 10^17) |
| **Escalabilidad** | Probado con 100 silos (millones potenciales) |
| **Parámetros por experto** | ~50k (tiny vs ~100M en MoE estándar) |

### Especialización: 10/10 expertos vencen al denso

| Silo | Experto | Denso | Ventaja |
|------|---------|-------|---------|
| silo_0 | 0.0708 | 0.1023 | **+31%** |
| silo_1 | 0.0758 | 0.1029 | **+26%** |
| silo_2 | 0.0717 | 0.1021 | **+30%** |
| silo_3 | 0.0736 | 0.0995 | **+26%** |
| silo_4 | 0.0757 | 0.1052 | **+28%** |
| silo_5 | 0.0678 | 0.0984 | **+31%** |
| silo_6 | 0.0689 | 0.0998 | **+31%** |
| silo_7 | 0.0712 | 0.1015 | **+30%** |
| silo_8 | 0.0687 | 0.1009 | **+32%** |
| silo_9 | 0.0780 | 0.1050 | **+26%** |

**Promedio:** Experto **0.0722** | Denso **0.1018** → **29.1% mejora**

---

## 🏗️ Arquitectura
Entrada (ID jerárquico + datos)
│
▼
┌─────────────────────────┐
│ Hash Jerárquico │ ← a.b.c.d.e.f.g.h → hash único
│ Espacio: 10^17 │
└─────────────────────────┘
│
▼
┌─────────────────────────┐
│ Tabla Hash (dict) │ ← lookup O(1) determinista
└─────────────────────────┘
│
▼
┌─────────────────────────┐
│ Experto tiny │ ← MLP: 32 → 128 → 128 → 32
│ por ID │ (~50k parámetros)
└─────────────────────────┘
│
▼
Predicción específica del ID

text

**Principio:** Cada ID tiene su propio experto. Cada experto se entrena SOLO con sus datos. No hay "fuga" entre IDs.

---

## 🧪 Aplicaciones directas

| Industria | ID única | Qué predice el experto | Por qué importa |
|-----------|----------|------------------------|-----------------|
| **E-commerce** | Producto ID | Demanda, precio óptimo, fraude | Catalogar millones sin reentrenar |
| **Logística** | Tracking ID | Tiempo de entrega, rutas óptimas | Millones de envíos en paralelo |
| **I+D** | DOI / Run ID | Impacto, anomalías | Datos sensibles que no se mezclan |
| **Salud** | Historia clínica | Riesgo, tratamientos | GDPR/HIPAA: cada paciente ve solo sus datos |
| **Catastro** | parcela ID | Valoración, cambios | Estructura jerárquica natural |

---

## 🔬 Comparación con enfoques existentes

| Aspecto | MoE estándar | Este enfoque |
|---------|--------------|--------------|
| **Número de expertos** | ~100-1000 | **Millones** |
| **Parámetros por experto** | ~100M | **~50k** |
| **Enrutamiento** | Aprendido (colapsa) | **Hash determinista O(1)** |
| **Actualización por ID** | Reentrenar todo | **Nuevo experto (versionado)** |
| **Privacidad** | Datos centralizados | **Por diseño** (cada ID ve solo sus datos) |
| **Explicabilidad** | "Se activó el experto 7" | "Se activó el experto del ID X" |
| **Costo de inferencia** | k expertos (2-8) | **1 experto** |

---

## ⚠️ Hallazgo crítico (actualización continua)

| Método | Olvido antiguos | Mejora nuevos |
|--------|-----------------|---------------|
| EDE estándar | +74.8% | -9.4% |
| Con buffer de replay | +74.8% | -9.4% |
| Con EWC | +34.3% | -4.4% |
| Cambio numérico gradual | **+15.4%** | **-4.7%** |

**Conclusión:** No existe un punto de compromiso entre dominios diferentes.

> **Cambio de dominio = nuevo experto (versionado)**

No intentes actualizar un experto existente cuando el dominio cambia significativamente. Crea un nuevo silo (como control de versiones).

---


identity-experts-moe/
│
├── README.md
├── LICENSE
├── requirements.txt
│
├── src/
│   ├── final.py           # Pipeline completo
│   ├── baseline.py        # Versión base
│   └── selector.py        # Entrenamiento del selector
│
├── results/
│   └── figure_*.png
│
└── docs/
    └── research_summary.md
	


📚 Citación
bibtex
@software{identity_experts_moe,
  author = {Magali Ofelia Gafe},
  title = {Identity Experts MoE: Hash-Based Routing for Specialized Silos},
  year = {2026},
  url = {https://github.com/maga-484/identity-experts-moe}
}
📄 Licencia
MIT License - ver archivo LICENSE

🙋‍♀️ Contacto
GitHub: maga-484

Proyecto: identity-experts-moe

"Millones de IDs, un experto por ID, enrutamiento O(1)."
```
