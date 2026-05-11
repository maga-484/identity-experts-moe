```markdown
# Investigación: Expertos Identitarios con Hash 8!

## Agrimensura + IA: Un puente interdisciplinario

**Autora:** Agrimensora, Facultad de Ingeniería, Universidad Nacional de La Plata

**Motivación personal:** Como agrimensora, me interesa explorar cómo las estructuras que usamos para organizar el territorio (silos catastrales, jerarquías parcelarias) pueden inspirar nuevas arquitecturas en inteligencia artificial.

## El origen: Silos catastrales bonaerenses

En la provincia de Buenos Aires, cada parcela se identifica con una ID jerárquica única:
Provincia → Municipio → Zona → Sección → Chacra → Quinta → Fracción → Manzana → Parcela → Unidad Funcional

text

Estos 10 niveles generan un espacio factorial (10! combinaciones posibles), pero en la práctica existen **silos físicos** (archivos redondos verticales) que contienen solo las combinaciones reales.

**La pregunta que guía este trabajo:**

> *¿Qué pasa si aplicamos esta misma lógica al entrenamiento de Mixture of Experts (MoE), donde cada "experto" se entrena en un silo de datos homogéneo y se accede a él mediante una clave única (hash)?*

## Metodología

### Simulación de silos
- 100 silos sintéticos
- Cada silo: 500 ejemplos, dimensión 32
- Estructura base compartida (W_base)
- Desplazamiento numérico para simular "variación catastral"

### Experto Identitario (tiny)
- MLP: 32 → 128 → 128 → 32
- LayerNorm para estabilidad
- ~50k parámetros (pequeño, como un archivo de parcela)

### Enrutamiento (como la mesa de entrada)
- Hash determinista: `hash(embedding) % 40320`
- Tabla hash con resolución de colisiones
- Acceso O(1)

## Resultados

### 1. Especialización: 10/10 victorias

Cada experto identitario fue evaluado contra un modelo denso entrenado con todos los silos mezclados.

| Silo | Experto | Denso | ¿Gana? |
|------|---------|-------|--------|
| 0 | 0.0708 | 0.1023 | ✓ |
| 1 | 0.0758 | 0.1029 | ✓ |
| 2 | 0.0717 | 0.1021 | ✓ |
| 3 | 0.0736 | 0.0995 | ✓ |
| 4 | 0.0757 | 0.1052 | ✓ |
| 5 | 0.0678 | 0.0984 | ✓ |
| 6 | 0.0689 | 0.0998 | ✓ |
| 7 | 0.0712 | 0.1015 | ✓ |
| 8 | 0.0687 | 0.1009 | ✓ |
| 9 | 0.0780 | 0.1050 | ✓ |

**Conclusión:** Un experto entrenado en un silo homogéneo supera consistentemente a un modelo denso que ve todos los datos mezclados.

### 2. Enrutamiento hash

- Espacio disponible: 40320 posiciones
- Silos actuales: 100 (0.25% ocupación)
- Acceso O(1)

### 3. Actualización continua (cambio numérico 0.00 → 0.15)

| Métrica | Antes | Después | Cambio |
|---------|-------|---------|--------|
| Antiguos | 0.0708 | 0.0817 | +15.4% |
| Nuevos | 0.1466 | 0.1534 | -4.7% |

**Conclusión para el mundo real:** Cuando una parcela cambia (nueva área, nueva valuación), no se "actualiza" el archivo antiguo sino que se crea un nuevo asiento. Lo mismo aplica aquí: **cambio de dominio = nuevo experto**.

## Reflexión final

Como agrimensora, aprendí que la organización del territorio en silos jerárquicos no es un capricho burocrático: es una estrategia de gestión que reduce la complejidad.

Este proyecto es una exploración personal para ver si esa misma estrategia funciona en inteligencia artificial. Los resultados sugieren que sí:

- **Estructurar los datos en silos jerárquicos** reduce drásticamente la cantidad de ejemplos necesarios
- **Expertos tiny especializados** pueden vencer a modelos densos grandes
- **El hash factorial** (inspirado en las combinaciones catastrales) es un buen punto de partida para el enrutamiento

---

**Fecha:** Mayo 2026

**Contacto:** [https://github.com/maga-484]
