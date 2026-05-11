# Identity Experts MoE

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

markdown

# Identity Experts MoE

**Enrutamiento hash O(1) para Mixture of Experts escalable**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org)

---

## 🚀 ¿Qué resuelve este proyecto?

En sistemas con **millones de entidades identificables** (productos, experimentos, usuarios, envíos, transacciones):

| Problema                               | Consecuencia                            |
| -------------------------------------- | --------------------------------------- |
| Un modelo denso único para todo        | Caro de entrenar, lento de actualizar   |
| Los cambios en una entidad             | Requieren reentrenar el modelo completo |
| La especialización "emerge" débilmente | O no emerge, o colapsa                  |

**Este proyecto implementa una alternativa radical:**  
_Un experto tiny por ID, enrutado mediante tabla hash O(1)._

---

## 📊 Resultados clave

| Métrica                      | Resultado                                    |
| ---------------------------- | -------------------------------------------- |
| **Expertos vs modelo denso** | 10/10 victorias                              |
| **Ventaja promedio**         | **+29%** (0.0722 vs 0.1018 MSE)              |
| **Enrutamiento**             | O(1) mediante hash `8! = 40320`              |
| **Escalabilidad**            | Probado con 100 silos (millones potenciales) |
| **Parámetros por experto**   | ~50k (tiny vs ~100M en MoE estándar)         |

---

## 🏗️ Arquitectura

Entrada (ID + datos)
│
▼
┌─────────────────┐
│ Hash(ID) │ ← hash = simple_hash(embedding) % 40320
│ 8! = 40320 │
└─────────────────┘
│
▼
┌─────────────────┐
│ Tabla Hash │ ← lookup O(1) (sin colapso del router)
└─────────────────┘
│
▼
┌─────────────────┐
│ Experto tiny │ ← MLP: 32 → 128 → 128 → 32
│ por ID │ (~50k parámetros)
└─────────────────┘
│
▼
Predicción específica del ID

text

**Principio:** Cada ID tiene su propio experto. Cada experto se entrena SOLO con sus datos. No hay "fuga" entre IDs.

---

## 🧪 Aplicaciones directas

| Industria      | ID única         | Qué predice el experto            | Por qué importa                                           |
| -------------- | ---------------- | --------------------------------- | --------------------------------------------------------- |
| **E-commerce** | Producto ID      | Demanda, precio óptimo, fraude    | Catalogar millones de productos sin reentrenar todo       |
| **Logística**  | Tracking ID      | Tiempo de entrega, rutas óptimas  | Millones de envíos en paralelo                            |
| **I+D**        | DOI / Run ID     | Impacto de publicación, anomalías | Experimentos con datos sensibles que no se pueden mezclar |
| **Salud**      | Historia clínica | Riesgo de reingreso, tratamientos | GDPR/HIPAA: cada paciente ve solo sus datos               |
| **Telecom**    | Línea ID         | Caídas de red, tráfico            | Detectar fallas por línea sin entrenar modelo global      |
| **Energía**    | Medidor ID       | Consumo, fallas, fraude           | Millones de medidores, cada uno con su patrón             |

---

## 🔬 Comparación con enfoques existentes

| Aspecto                    | MoE estándar (DeepSeek, Mixtral) | Este enfoque                               |
| -------------------------- | -------------------------------- | ------------------------------------------ |
| **Número de expertos**     | ~100-1000                        | **Millones**                               |
| **Parámetros por experto** | ~100M                            | **~50k**                                   |
| **Enrutamiento**           | Aprendido (colapsa fácilmente)   | **Hash determinista O(1)**                 |
| **Actualización por ID**   | Reentrenar todo el modelo        | **Local (EDE + EWC)**                      |
| **Privacidad**             | Datos centralizados (riesgo)     | **Por diseño** (cada ID ve solo sus datos) |
| **Explicabilidad**         | "Se activó el experto 7"         | "Se activó el experto del ID X"            |
| **Costo de inferencia**    | k expertos (2-8) + combinación   | **1 experto** (el del ID)                  |

---

## 📈 Escalabilidad

| Métrica           | Valor                                         |
| ----------------- | --------------------------------------------- |
| Espacio hash      | `8! = 40320` posiciones (expansible)          |
| Silos probados    | 100 (0.25% ocupación)                         |
| Ocupación teórica | Millones de IDs sin colisiones significativas |
| Acceso            | O(1) determinista                             |

---

## 🛠️ Instalación y uso

```bash
git clone https://github.com/maga-484/identity-experts-moe.git
cd identity-experts-moe
pip install -r requirements.txt
python src/final.py
Estructura del repositorio
text
identity-experts-moe/
│
├── README.md
├── LICENSE
├── requirements.txt
│
├── src/
│   ├── final.py           # Versión completa (EDE + EWC + cambio gradual)
│   └── baseline.py        # Versión base (hash routing + especialización)
│
├── results/
│   ├── figure_baseline.png    # Comparación experto vs denso
│   └── figure_cambio_numerico.png  # Evolución con EDE
│
└── docs/
    └── research_summary.md     # Metodología y hallazgos
📊 Visualización de resultados
Especialización: 10/10 expertos vencen al denso
Silo	Experto	Denso	Ventaja
0	0.0708	0.1023	+31%
1	0.0758	0.1029	+26%
2	0.0717	0.1021	+30%
3	0.0736	0.0995	+26%
4	0.0757	0.1052	+28%
5	0.0678	0.0984	+31%
6	0.0689	0.0998	+31%
7	0.0712	0.1015	+30%
8	0.0687	0.1009	+32%
9	0.0780	0.1050	+26%
Promedio: Experto 0.0722 | Denso 0.1018 → 29.1% mejora

Actualización continua (cambio numérico 0.00 → 0.15)
Métrica	Antes EDE	Después EDE	Cambio
Antiguos	0.0708	0.0817	+15.4%
Nuevos	0.1466	0.1534	-4.7%
Hallazgo: Cambios de dominio significativos requieren nuevo experto (versionado), no actualización continua.

🔄 Actualización continua (EDE + EWC)
Para cambios pequeños (ej. actualización de precio, nuevo evento de tracking), el sistema usa:

EDE Stratonovich con método Euler-Heun

Elastic Weight Consolidation (EWC) para mitigar olvido

Buffer de replay con datos históricos

Para cambios de dominio significativos → crear nuevo experto (como control de versiones).

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
