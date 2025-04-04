
# Resumo: Treinamento de Redes Neurais com TensorFlow

## **Contexto da Semana**
- Foco: Treinar uma rede neural (semana 2 do curso).
- Semana anterior: Inferência em redes neurais.
- Exemplo prático: Reconhecimento de dígitos manuscritos (0 ou 1).

---

## **Arquitetura da Rede Neural**
- **Entrada**: Imagem (representada por `X`).
- **Camadas Ocultas**:
  - 1ª camada: 25 unidades com ativação **sigmoid**.
  - 2ª camada: 15 unidades com ativação **sigmoid**.
- **Saída**: 1 unidade (classificação binária).

---

## **Passos para Treinar o Modelo (TensorFlow)**
1. **Definição do Modelo**:
   ```python
   model = Sequential([
       Dense(25, activation='sigmoid', input_shape=(dimensão_entrada,)),
       Dense(15, activation='sigmoid'),
       Dense(1, activation='sigmoid')
   ])
   ```
   - Sequência de camadas definidas.

2. **Compilação do Modelo**:
   ```python
   model.compile(loss='binary_crossentropy')
   ```
   - Especificação da **função de perda** (`binary_crossentropy` para classificação binária).

3. **Treinamento**:
   ```python
   model.fit(X, Y, epochs=número_de_épocas)
   ```
   - Ajuste dos parâmetros usando gradiente descendente.
   - **Épocas**: Número de iterações do algoritmo de treinamento.

Aqui está o resumo em Markdown para a transcrição fornecida:


# Detalhes do Treinamento de Redes Neurais com TensorFlow
## **Comparação com Regressão Logística**
Os três passos para treinar uma rede neural são **análogos ao treinamento de regressão logística**:
1. **Especificar a saída**:
   - Regressão Logística: $f_{\mathbf{w},b}(x) = g(\mathbf{w} \cdot \mathbf{x} + b)$, onde $g$ é a sigmoide.
   - Rede Neural: Definição da arquitetura (camadas, unidades, funções de ativação) no código (`Sequential` e `Dense`).

2. **Definir a função de perda e custo**:
   - **Perda** (exemplo único): 
     - Regressão Logística: $\mathcal{L}(f(x), y) = -y \log(f(x)) - (1-y) \log(1-f(x))$.
     - Rede Neural: Mesma função (chamada de **`binary_crossentropy`** no TensorFlow).
   - **Custo**: Média da perda sobre todo o conjunto de treinamento.

3. **Minimizar o custo**:
   - **Gradiente Descendente**: Atualização iterativa de $\mathbf{w}$ e $b$ para minimizar $J(\mathbf{w}, b)$.
   - No TensorFlow: Automatizado via `model.fit()`.

---

## **Funções de Perda no TensorFlow**
- **Classificação Binária**:
  ```python
  model.compile(loss='binary_crossentropy')
  ```
  - Usada quando $y \in \{0, 1\}$.

- **Regressão**:
  ```python
  model.compile(loss='mean_squared_error')
  ```
  - **Perda**: $ \mathcal{L}(f(x), y) = \frac{1}{2}(f(x) - y)^2$ .
  - **Custo**: Média do erro quadrático.

---

## **Papel do TensorFlow no Treinamento**
- **Backpropagation**:
  - Algoritmo usado para calcular gradientes (derivadas parciais do custo em relação aos parâmetros).
  - Implementado automaticamente no `model.fit()`.

- **Gradiente Descendente**:
  - Atualização automática de pesos ($\mathbf{w}$) e vieses ($b$) em todas as camadas.
  - **Épocas (`epochs`)**: Número de iterações do algoritmo.


# Ativações em Redes Neurais: Alternativas à Sigmoide

## **Motivação para Novas Funções de Ativação**
- **Limitação da Sigmoide**:
  - Saída restrita a valores entre **0 e 1** (ex.: "consciência" de um produto não pode ser modelada como valor contínuo não negativo).
  - Problemas de **vanishing gradient** em redes profundas.

---

## **Função ReLU (Rectified Linear Unit)**
- **Definição**:
  
  $$g(z) = \max(0, z)$$
  
  - Saída é **0** para  $z < 0$ e **z** para  $z \geq 0 $.
- **Vantagens**:
  - Permite valores **não negativos ilimitados** (ex.: modelar "consciência" como 0, 100, 1000).
  - Computacionalmente eficiente (evita cálculos exponenciais da sigmoide).
  - Mitiga vanishing gradient em redes profundas.

- **Exemplo de Uso**:
  - Camadas ocultas para problemas não lineares (ex.: previsão de demanda, reconhecimento de imagem).

---

## **Outras Funções de Ativação Comuns**
1. **Sigmoide**:
   
   $$g(z) = \frac{1}{1 + e^{-z}}$$   
   - Uso comum em **classificação binária** (camada de saída).

2. **Linear**:
   
   $$g(z) = z$$
   

   - Equivale a "nenhuma ativação" (saída = entrada ponderada + viés).
   - Usada em **problemas de regressão** (ex.: prever preços).

3. **Softmax** (abordada posteriormente):
   - Ideal para **classificação multiclasse** (camada de saída).

---

## **Quando Usar Cada Ativação?**
- **Camadas Ocultas**:
  - **ReLU** é padrão para a maioria dos casos (performance e simplicidade).
  - Alternativas: Leaky ReLU, Parametric ReLU (para evitar "neurônios mortos").
  
- **Camada de Saída**:
  - **Sigmoide**: Classificação binária (probabilidade).
  - **Linear**: Regressão (valores contínuos).
  - **Softmax**: Classificação multiclasse.

---

# Escolha de Funções de Ativação em Redes Neurais

## **Camada de Saída: Diretrizes por Tipo de Problema**
| Tipo de Problema              | Função de Ativação | Exemplo de Uso                  |
|-------------------------------|--------------------|----------------------------------|
| **Classificação Binária**     | Sigmoide           | Prever probabilidade  $y = 1$ |
| **Regressão (y positivo/negativo)** | Linear       | Variação de preço de ações       |
| **Regressão (y não negativo)**| ReLU               | Preço de imóveis (≥ 0)           |

**Exemplo TensorFlow**:
```python
model = Sequential([
    Dense(25, activation='relu', input_shape=(input_dim,)),  # Camada oculta
    Dense(15, activation='relu'),                            # Camada oculta
    Dense(1, activation='sigmoid')  # Saída: classificação binária
])
# Ou: activation='linear' para regressão, 'relu' para y ≥ 0
```

---

## **Camadas Ocultas: Por Que ReLU?**
- **Vantagens**:
  - **Eficiência Computacional**: Cálculo simples $(\max(0, z))$ vs. exponencial (sigmoide).
  - **Evita Vanishing Gradient**: Gradientes não "desaparecem" em regiões planas (apenas para $z < 0$).
  - **Não-linearidade**: Permite modelar relações complexas (essencial para redes profundas).

- **Comparação com Sigmoide**:
  | Característica       | ReLU                          | Sigmoide                       |
  |----------------------|-------------------------------|--------------------------------|
  | Faixa de Saída       | $[0, +\infty)$                | $(0, 1)$                       |
  | Regiões Planas       | Apenas $z < 0$                | $z \to \pm\infty$              |
  | Velocidade de Treino | Mais rápido                   | Mais lento (gradientes pequenos) |
---

## **Outras Funções (Opcionais)**
- **LeakyReLU**: Versão modificada do ReLU que evita "neurônios mortos" ($g(z) = \max(\alpha z, z)$, com $\alpha \approx 0{.}01$).
- **tanh**: Similar à sigmoide, mas com saída entre $[-1, 1]$. Menos comum em camadas ocultas.
- **Swish**: $g(z) = z \cdot \sigma(z)$. Performance superior em alguns cenários (pesquisa recente).

---

## **Por Que Não Usar Apenas Ativações Lineares?**
- **Redes Profundas Colapsam**: Combinações lineares de camadas equivalem a uma única transformação linear.
- **Sem Não-linearidade**: Incapaz de aprender padrões complexos (ex.: XOR, imagens, séries temporais).


# Por Que Redes Neurais Precisam de Funções de Ativação Não Lineares?

## **O Problema das Ativações Lineares em Todas as Camadas**
Se todas as camadas usarem **ativações lineares** $(g(z) = z )$, a rede neural **colapsa em um modelo linear**, perdendo sua capacidade de aprender padrões complexos. Exemplo:

### **Cenário Simplificado**  
- **Arquitetura**: 1 camada oculta + 1 camada de saída (ambas lineares).
- **Cálculo**:
  ```
  a1 = w1 * x + b1        (Camada oculta linear)
  a2 = w2 * a1 + b2       (Camada de saída linear)
  → a2 = (w2 * w1) * x + (w2 * b1 + b2)
  ```
  - Equivalente a **regressão linear** (\( y = Wx + b \)), mesmo com múltiplas camadas!

### **Generalização para Redes Profundas**  
- **Qualquer número de camadas lineares** equivale a **uma única transformação linear**:
  ```
  a_final = W_n * (... * (W_2 * (W_1 * x + b_1) + b_2) ...) + b_n
  → Reduzível a \( y = W_{total} \cdot x + b_{total} \)
  ```
- **Conclusão**: Sem não-linearidades, redes profundas são inúteis.

---

## **Impacto das Funções de Ativação Não Lineares**
### **Exemplo com ReLU**  
- **Arquitetura**:  
  ```python
  model = Sequential([
      Dense(10, activation='relu'),  # Camada oculta não linear
      Dense(1, activation='linear')  # Camada de saída linear
  ])
  ```
- **Resultado**:  
  A rede pode modelar **relações não lineares** (ex.: curvas, interações complexas).

### **Comparação: Com vs. Sem Ativações Não Lineares**
|                          | **Com ReLU**                     | **Sem Ativações Não Lineares**       |
|--------------------------|----------------------------------|--------------------------------------|
| **Capacidade de Modelagem** | Aproxima funções complexas      | Equivalente a regressão linear       |
| **Aplicações**           | Classificação, Visão Computacional | Problemas triviais (ex.: tendências) |
| **Eficiência**           | Alta (aprende padrões hierárquicos) | Baixa (limitada a relações lineares) |

---

## **Casos Especiais e Recomendações**
1. **Camada de Saída Linear**:
   - **Uso Aceitável**: Problemas de regressão onde \( y \in \mathbb{R} \).
   - **Exemplo**:  
     ```python
     Dense(1, activation='linear')  # Prever preços, temperaturas
     ```

2. **Camadas Ocultas**:
   - **Nunca use ativações lineares**: Opte por **ReLU** (padrão) ou alternativas (LeakyReLU, tanh).
   - **Exemplo de Boa Prática**:  
     ```python
     Dense(64, activation='relu')  # Camada oculta com ReLU
     ```

---

## **Conclusão**  
- **Funções não lineares** (ex.: ReLU, sigmoide) são **essenciais** para que redes neurais aprendam representações hierárquicas e complexas.
- **Sem elas**: Redes profundas tornam-se equivalentes a modelos lineares simples, desperdiçando capacidade computacional.
- **Regra de Ouro**:  
  - **Ocultas**: ReLU (ou variações).  
  - **Saída**: Escolha conforme o problema (sigmoide, linear, softmax).

# Classificação Multiclasse: Introdução e Contexto

## **Definição**
- **Problema**: Classificação com **mais de duas categorias** possíveis para a saída $y$.
- **Exemplos**:
  - Reconhecimento de dígitos manuscritos (0 a 9).
  - Diagnóstico médico entre múltiplas doenças.
  - Inspeção de defeitos em produtos (ex.: arranhões, descoloração, chips).

---

## **Diferença para Classificação Binária**

| **Classificação Binária**   | **Classificação Multiclasse**             |
|-----------------------------|-------------------------------------------|
| $y \in \{0, 1\}$            | $y \in \{1, 2, ..., C\}$ ($C \geq 3$)     |
| Exemplo: "0" vs "1"         | Exemplo: "0", "1", ..., "9"               |
| Fronteira de decisão binária| Múltiplas fronteiras (ex: 4 classes)      |

**Exemplo Visual**:
- Dados com 4 classes:  
  ![Fronteiras de Decisão](./images/multiclass.png)
  - Cada classe (círculos, triângulos, quadrados, cruzes) é separada por fronteiras não lineares.

---

## **Softmax Regression: Generalização da Logística**

- **Objetivo**: Prever a **probabilidade** de $y$ pertencer a cada classe $c$.
- **Funcionamento**:
  - Calcula um "score":  
    $$
    z_c = \mathbf{w}_c \cdot \mathbf{x} + b_c
    $$
  - Converte scores em probabilidades com a função **softmax**:
    $$
    P(y = c \mid \mathbf{x}) = \frac{e^{z_c}}{\sum_{k=1}^C e^{z_k}}
    $$
- **Vantagem**: Permite modelar **relações complexas** entre múltiplas categorias.

---

## **Integração com Redes Neurais**
- **Camada de Saída**:
  - **Ativação Softmax**: Substitui a sigmoide para gerar probabilidades normalizadas.
  - Exemplo em TensorFlow:
    ```python
    model = Sequential([
        Dense(25, activation='relu'),
        Dense(15, activation='relu'),
        Dense(10, activation='softmax')  # 10 classes (ex.: dígitos 0-9)
    ])
    ```
- **Função de Perda**: `sparse_categorical_crossentropy` (para rótulos inteiros) ou `categorical_crossentropy` (one-hot encoded).


# Softmax Regression: Generalização para Classificação Multiclasse

---

## **Visão Geral**
- **Objetivo**: Estender a **regressão logística** (binária) para problemas com **C classes** (\( C \geq 2 \)).
- **Aplicações**: Reconhecimento de dígitos (0-9), diagnóstico médico múltiplo, classificação de defeitos industriais.

---
## **Mecanismo do Algoritmo**

### **Passo 1: Cálculo dos "Scores" ($z_j$)**
Para cada classe $j \in \{1, 2, ..., C\}$:
$$
z_j = \mathbf{w}_j \cdot \mathbf{x} + b_j
$$
- $\mathbf{w}_j$: Vetor de pesos para a classe $j$.
- $b_j$: Viés para a classe $j$.

### **Passo 2: Função Softmax**
Converte scores em probabilidades normalizadas:
$$
a_j = \frac{e^{z_j}}{\sum_{k=1}^C e^{z_k}}
$$
- **Interpretação**: $a_j$ é a probabilidade estimada de $y = j$ dado $\mathbf{x}$.
- **Propriedade**: $\sum_{j=1}^C a_j = 1$.

**Exemplo**:
- Se $a_1 = 0.3$, $a_2 = 0.2$, $a_3 = 0.15$, então $a_4 = 0.35$ (soma = 1).

---

## **Função de Perda e Custo**

- **Perda (Exemplo Único)**:
  $$
  \mathcal{L}(y, \mathbf{a}) = -\log(a_y)
  $$
  - $y$: Classe verdadeira (ex.: $y = 2$).
  - Penaliza **baixa confiança** na classe correta (ex.: se $a_2 = 0.1$, $\mathcal{L} = -\log(0.1) \approx 2.3$).

- **Custo**:
  $$
  J(\mathbf{W}, \mathbf{b}) = \frac{1}{m} \sum_{i=1}^m \mathcal{L}(y^{(i)}, \mathbf{a}^{(i)})
  $$
  - Média das perdas sobre todo o conjunto de treinamento.

---

## **Relação com a Regressão Logística**

- **Caso Especial ($C = 2$)**:
  - Softmax equivale à regressão logística.
  - Probabilidade da classe 1:


# Implementação de Redes Neurais com Saída Softmax para Classificação Multiclasse

---

## **Arquitetura da Rede Neural**
- **Camadas Ocultas**:  
  ```python
  model = Sequential([
      Dense(25, activation='relu'),  # 1ª camada oculta
      Dense(15, activation='relu'),  # 2ª camada oculta
      Dense(10)  # Camada de saída (sem ativação explícita!)
  ])
  ```
  - **Observação**: A camada de saída **não usa `activation='softmax'` diretamente** (motivo explicado abaixo).

---

## **Função de Perda: `SparseCategoricalCrossentropy`**
- **Uso**:
  ```python
  model.compile(
      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
      optimizer='adam'
  )
  ```
- **Por Que `from_logits=True`?**  
  - **Estabilidade Numérica**: O TensorFlow calcula o softmax **internamente** de forma otimizada, evitando erros de precisão.
  - **Evita Duplicação**: Combinar `softmax` na camada de saída + `SparseCategoricalCrossentropy` sem `from_logits` pode levar a cálculos redundantes e instabilidade.

---

## **Treinamento e Predição**
1. **Treinar o Modelo**:
   ```python
   model.fit(X_train, y_train, epochs=100)
   ```
2. **Obter Probabilidades**:
   ```python
   logits = model.predict(X_new)  # Saídas "brutas" (antes do softmax)
   probabilidades = tf.nn.softmax(logits).numpy()  # Aplica softmax
   ```

---

## **Exemplo Completo (Código Recomendado)**
```python
import tensorflow as tf

# Definir arquitetura
model = tf.keras.Sequential([
    tf.keras.layers.Dense(25, activation='relu', input_shape=(784,)),
    tf.keras.layers.Dense(15, activation='relu'),
    tf.keras.layers.Dense(10)  # Sem ativação!
])

# Compilar modelo
model.compile(
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer='adam',
    metrics=['accuracy']
)

# Treinar
model.fit(X_train, y_train, epochs=10)

# Predição (exemplo)
logits = model.predict(X_test)
probabilidades = tf.nn.softmax(logits)
classes_preditas = tf.argmax(probabilidades, axis=1)
```

---

## **Por Que Não Usar `activation='softmax'` Diretamente?**
- **Problema**:  
  Aplicar softmax na camada de saída + `SparseCategoricalCrossentropy` sem `from_logits=True` pode causar **underflow/overflow** numérico (valores extremos em `e^z`).
- **Solução**:  
  `from_logits=True` faz o TensorFlow calcular o softmax de forma **numericamente estável**, usando truques como subtração do valor máximo (`logsumexp`).

---

## **Resumo de Boas Práticas**
1. **Camada de Saída**:  
   - Sem ativação (`Dense(10)`).
2. **Função de Perda**:  
   - `SparseCategoricalCrossentropy(from_logits=True)`.
3. **Pós-Processamento**:  
   - Aplicar `tf.nn.softmax` apenas durante inferência para obter probabilidades.

---

## **Comparação: Codificação de Rótulos**
| **Tipo de Rótulo**       | **Função de Perda**                     | **Exemplo**                  |
|--------------------------|-----------------------------------------|------------------------------|
| **Inteiros (0, 1, 2...)** | `SparseCategoricalCrossentropy`         | `y = [2, 0, 4]`              |
| **One-Hot Encoded**      | `CategoricalCrossentropy`               | `y = [[0,0,1], [1,0,0], ...]` |

# Implementação Estável de Softmax no TensorFlow: Evitando Erros Numéricos

---

## **O Problema: Instabilidade Numérica no Cálculo do Softmax**

- **Exponenciais Extremos**: Valores altos/baixos de $z_j$ causam *overflow* ($e^{z_j} \to \infty$) ou *underflow* ($e^{z_j} \to 0$).

- **Exemplo**:
  ```python
  # Cálculo ingênuo (problemático)
  import numpy as np

  z = [1000, 2000, 3000]
  e_z = [np.exp(i) for i in z]  # Resulta em overflow (NaN)
  ```

---

## **Solução: Trabalhar com Logits (Forma Estável)**
### **Passo a Passo**
1. **Camada de Saída Linear**:
   ```python
   model = Sequential([
       Dense(25, activation='relu'),
       Dense(15, activation='relu'),
       Dense(10)  # Sem ativação! Saídas são logits (z_j)
   ])
   ```
2. **Função de Perda com `from_logits=True`**:
   ```python
   model.compile(
       loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
       optimizer='adam'
   )
   ```
3. **Inferência com Softmax**:
   ```python
   logits = model.predict(X_new)
   probabilidades = tf.nn.softmax(logits).numpy()  # Aplica softmax após predição
   ```

---

## **Comparação: Implementação Tradicional vs. Recomendada**
| **Aspecto**               | **Implementação Tradicional**                     | **Implementação Recomendada**                  |
|---------------------------|--------------------------------------------------|-----------------------------------------------|
| **Camada de Saída**        | `Dense(10, activation='softmax')`                | `Dense(10)` (saída linear)                    |
| **Cálculo de Perda**       | `loss='sparse_categorical_crossentropy'`         | `SparseCategoricalCrossentropy(from_logits=True)` |
| **Estabilidade Numérica** | Risco de overflow/underflow (especialmente em C grande) | Cálculo otimizado internamente pelo TensorFlow |
| **Legibilidade**           | Mais intuitiva                                   | Menos intuitiva (requer conhecimento de logits) |

---

## **Por Que Funciona?**
- **Truque do Log-Sum-Exp**:
  ```python
  logits = z_j - max(z)  # Subtrai o máximo para evitar overflow
  softmax = e^(logits) / sum(e^(logits))
  ```
  - O TensorFlow aplica esta otimização automaticamente quando `from_logits=True`.

---

## **Exemplo Completo (MNIST)**
```python
# Definir modelo
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10)  # Logits
])

# Compilar com cálculo estável
model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)

# Treinar
model.fit(train_images, train_labels, epochs=5)

# Predição
logits = model.predict(test_images)
probabilidades = tf.nn.softmax(logits)
```

---

## **Conclusão**
- **Evite `activation='softmax'` na Camada de Saída**: Use saída linear + `from_logits=True`.
- **Benefícios**:
  - **Estabilidade**: Previne NaN durante o treinamento.
  - **Eficiência**: Cálculo vetorizado otimizado pelo TensorFlow.
- **Trade-off**: Legibilidade ligeiramente reduzida, mas ganhos práticos significativos.

👉 **Dica Prática**: Sempre use `from_logits=True` com `SparseCategoricalCrossentropy` para problemas multiclasse!


# Classificação Multi-Label: Detecção Simultânea de Múltiplos Objetos

## **Definição**
- **Problema**: Classificar uma única entrada (ex.: imagem) em **múltiplas categorias independentes**.
- **Exemplo Prático**:  
  - **Entrada**: Imagem de trânsito.
  - **Saída**: Vetor binário indicando presença de `[carro, ônibus, pedestre]`.
  ```python
  y = [1, 0, 1]  # Carro presente, ônibus ausente, pedestre presente
  ```

---

## **Arquitetura da Rede Neural**
### **Abordagem Recomendada: Rede Única com Múltiplas Saídas**
```python
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(input_dim,)),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(3, activation='sigmoid')  # 3 labels: carro, ônibus, pedestre
])
```

- **Camada de Saída**:  
  - **Neurônios**: Um por label (ex.: 3 para carro, ônibus, pedestre).  
  - **Ativação**: `sigmoid` (calcula probabilidade independente para cada label).

### **Função de Perda**
```python
model.compile(
    loss='binary_crossentropy',  # Perda para cada label individual
    optimizer='adam',
    metrics=['accuracy']
)
```
- **Motivação**: Cada saída é um problema de **classificação binária separado**.

---

## **Comparação: Multi-Label vs. Multi-Classe**
| **Característica**       | **Multi-Label**                          | **Multi-Classe**                     |
|--------------------------|------------------------------------------|--------------------------------------|
| **Número de Labels**     | Múltiplos por exemplo (ex.: [1, 0, 1])  | Único por exemplo (ex.: classe 3)    |
| **Exclusividade**        | Labels independentes                    | Labels mutuamente exclusivos         |
| **Camada de Saída**      | Sigmoid (por neurônio)                  | Softmax (probabilidades normalizadas)|
| **Aplicações**           | Detecção de objetos, tags               | Reconhecimento de dígitos, diagnósticos |

---

## **Exemplo de Uso no TensorFlow**
```python
# Dados de exemplo (imagens e labels multi-hot encoded)
X_train = ...  # Imagens de trânsito
y_train = np.array([
    [1, 0, 1],  # Carro e pedestre
    [0, 1, 0],  # Ônibus
    [1, 1, 1]   # Carro, ônibus, pedestre
])

# Treinamento
model.fit(X_train, y_train, epochs=10)

# Predição
imagem_nova = ...  # Nova imagem
predicao = model.predict(imagem_nova)
# Saída: [0.98, 0.05, 0.89] → [carro, não ônibus, pedestre]
```

---

## **Por Que Não Usar Softmax?**
- **Independência dos Labels**: Softmax força a soma 1, implicando dependência entre labels (não desejado em multi-label).
- **Sigmoid**: Permite que cada label seja tratado como um problema binário separado.

---

## **Casos de Uso Comuns**
- **Tags em Redes Sociais**: Uma foto pode ter tags como `#praia`, `#família`, `#pôr-do-sol`.
- **Diagnóstico Médico**: Paciente pode ter múltiplas condições (ex.: diabetes, hipertensão).
- **Autonomia Veicular**: Detecção simultânea de carros, pedestres, semáforos.

---

# Otimização em Aprendizado de Máquina: Do Gradiente Descendente ao Adam

---
## **Desafios do Gradiente Descendente Clássico**

- **Taxa de Aprendizado Fixa** ($\alpha$):
  - **Problema**: Mesmo $\alpha$ para todos os parâmetros.
  - **Consequências**:
    - Passos lentos em regiões planas (ex.: vales rasos).
    - Oscilações em terrenos íngremes (ex.: saltos entre encostas).

- **Exemplo Visual**:
  - **Formato de Tijolo vs. Elipse**: Caminho de otimização em formato elíptico amplifica oscilações.

---
## **Algoritmo Adam: Mecanismo e Vantagens**

### **Componentes-Chave**

1. **Momentum**:
   - Acelera atualizações na mesma direção (ex.: "inércia" em descidas consistentes).
   - Equação:  
     $$ v_j = \beta_1 v_j + (1 - \beta_1) \frac{\partial J}{\partial w_j} $$

2. **Adaptação por Parâmetro**:
   - Calcula "velocidades" ($v_j$) e "escalas" ($s_j$) individuais.
   - Ajusta $\alpha$ dinamicamente:  
     $$ \alpha_j = \frac{\alpha}{\sqrt{s_j} + \epsilon} $$

3. **Supressão de Oscilações**:
   - Reduz $\alpha$ para parâmetros oscilantes (ex.: $w_2$ na direção íngreme).

### **Implementação no TensorFlow**
```python
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='binary_crossentropy'
)
```

---

## **Benchmark: Gradiente Descendente vs. Adam**

| **Critério**              | **Gradiente Descendente**         | **Adam**                               |
|---------------------------|-----------------------------------|----------------------------------------|
| **Taxa de Convergência**  | Lenta (1000 épocas)               | Rápida (100 épocas)                    |
| **Sensibilidade a** $ \alpha $ | Alta (requer ajuste fino)       | Baixa (robusto a $ \alpha $)           |
| **Uso de Memória**        | Baixo                             | Moderado (armazena $ v_j, s_j $)       |

---

## **Casos de Uso Práticos**
- **Treinamento de CNNs**: Adam acelera convergência em redes profundas (ex.: ResNet).
- **NLP com Transformers**: Momentum ajuda em padrões sequenciais longos.
- **Projetos com Restrições Computacionais**: Adam reduz tempo de treinamento em 90%.

---

## **Por Que Adam é Revolucionário?**
- **Unifica Conceitos**:
  - Combina momentum (como em **RMSProp**) e adaptação por parâmetro (como em **AdaGrad**).
- **Democratização de ML**:
  - Permite treinar redes complexas sem ajuste hiperparamétrico intensivo.
- **Exemplo Real**:
  - Treinamento de GPT-3: Adam foi crucial para estabilizar otimização em 175B parâmetros.

# Redes Neurais Convolucionais (CNNs): Além das Camadas Densas

---

## **Camadas Convolucionais: Conceito Básico**
- **Diferente das Camadas Densas**:  
  - Neurônios **não conectam-se a todas as ativações anteriores**.  
  - Cada neurônio analisa **apenas uma região local** da entrada (ex.: janela em imagem ou série temporal).  
- **Exemplo com Sinal ECG (1D)**:  
  - **Entrada**: 100 pontos temporais ($X_1, X_2, ..., X_{100}$).  
  - **1ª Camada Convolucional**:  
    - Neurônio 1: Janela de $X_1$ a $X_{20}$.  
    - Neurônio 2: Janela de $X_{11}$ a $X_{30}$.  
    - [...]  
    - Neurônio 9: Janela de $X_{81}$ a $X_{100}$.  
  - **2ª Camada Convolucional**:  
    - Cada neurônio analisa subjanelas da camada anterior (ex.: 5 ativações).

---

## **Vantagens das CNNs**
1. **Eficiência Computacional**:  
   - Menos parâmetros (conexões locais vs. densas).  
2. **Redução de Overfitting**:  
   - Foco em padrões locais (ex.: bordas em imagens, picos em ECG).  
3. **Invariância a Transladações**:  
   - Detecta características independentemente da posição (útil em imagens/séries temporais).  

---

## **Arquitetura de Exemplo para Classificação de ECG**
1. **Camada de Entrada**:  
   - 100 valores temporais (sinal ECG).  
2. **Camada Convolucional 1D**:  
   - 9 neurônios, janela de 20 pontos.  
3. **Camada Convolucional 1D**:  
   - 3 neurônios, janela de 5 ativações.  
4. **Camada de Saída (Densa)**:  
   - 1 neurônio com ativação sigmoide (diagnóstico binário).  

---

## **Implementação em TensorFlow/Keras**
```python
model = tf.keras.Sequential([
    # Camada convolucional 1D para séries temporais
    tf.keras.layers.Conv1D(
        filters=9, 
        kernel_size=20, 
        activation='relu', 
        input_shape=(100, 1)  # 100 timesteps, 1 feature
    ),
    # Segunda camada convolucional
    tf.keras.layers.Conv1D(
        filters=3, 
        kernel_size=5, 
        activation='relu'
    ),
    # Camada de saída
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy')
```

---

## **Aplicações e Contexto**
- **Processamento de Imagens (2D)**:  
  - Detectores de bordas, reconhecimento de objetos.  
- **Séries Temporais (1D)**:  
  - Análise de ECG, previsão de demanda, dados de sensores.  
- **Pesquisa Moderna**:  
  - **Transformers**, **LSTMs**, e modelos com **mecanismos de atenção** usam camadas especializadas para contextos específicos.  

---

## **Por Que Usar CNNs?**
| **Critério**          | **Camadas Densas**               | **Camadas Convolucionais**       |
|-----------------------|-----------------------------------|-----------------------------------|
| **Conexões**          | Todas para todas                 | Locais/regionais                 |
| **Parâmetros**        | Alto risco de overfitting        | Reduzidos (menos conexões)       |
| **Uso de Dados**      | Requer grandes datasets          | Eficiente com dados limitados    |
| **Velocidade**        | Mais lento (operações densas)    | Mais rápido (operações locais)   |

