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
