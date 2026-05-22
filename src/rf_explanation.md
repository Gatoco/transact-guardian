"""
Random Forest - Explicación simple

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
IMAGINA ESTO:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

En vez de preguntar a UNA persona, preguntas a 100 personas
y te quedas con la respuesta de la mayoría.

Cada persona (árbol) puede equivocarse, pero juntas (bosque)
llegan a la respuesta correcta.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
QUÉ ES UN ÁRBOL DE DECISIÓN:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Ejemplo simple: ¿Es fraude o no?

        ¿Amount > 500?
           /        \
         Sí          No
         │            │
    ¿V1 > 0?      ¿V3 < -2?
      /    \        /    \
    Sí      No    Sí      No
     │       │     │       │
   FRAUDE  LEGIT  FRAUDE  LEGIT

El árbol hace preguntas en cascada hasta decidir.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
POR QUÉ "RANDOM" FOREST:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. **Random data**: Cada árbol ve solo una muestra aleatoria
   de los datos (bootstrap sampling)

2. **Random features**: En cada split, solo ve unas pocas
   features al azar (ej: de 33 features, usa solo 10)

Esto hace que los árboles sean DIFERENTES entre sí.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CÓMO FUNCIONA LA VOTACIÓN:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Árbol 1: Fraude (80% seguro)
Árbol 2: Legit (60% seguro)
Árbol 3: Fraude (70% seguro)
...
Árbol 100: Fraude (75% seguro)

Resultado: 65/100 dicen Fraude → PREDICE FRAUDE

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PARÁMETROS PRINCIPALES:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

n_estimators: Cuántos árboles (más = mejor pero más lento)
  - 100 = rápido y bueno
  - 500 = mejor, pero lento

max_depth: Qué tan profundo es cada árbol
  - Profundo = más complejo = puede overfit
  - Shallow = más simple = puede underfit

min_samples_split: Mínimo samples para hacer un split
  - Alto = menos overfitting
  - Bajo = más detalle pero puede overfit

class_weight: Para manejar imbalance
  - 'balanced' = lo mismo que en Logistic Regression

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
LOGISTIC REGRESSION VS RANDOM FOREST:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Logistic Regression:
  ✓ Rápido
  ✓ Interpretable (coeficientes)
  ✓ Bueno para linearly separable
  ✗ No captura no-linearidades

Random Forest:
  ✓ Captura relaciones complejas (no-lineales)
  ✓ Robusto a outliers
  ✓ Feature importance integrada
  ✓ No necesita feature scaling
  ✗ Más lento (100+ árboles)
  ✗ Menos interpretable

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PARA FRAUD DETECTION:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Random Forest suele funcionar MEJOR que Logistic Regression
porque:
1. Las relaciones fraude/no-fraude no son lineales
2. Hay interacciones entre features
3. Random Forest es más robusto al imbalance

Pero Logistic Regression es buen baseline para comparar.
"""