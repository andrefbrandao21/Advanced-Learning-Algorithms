
# Redes Neurais: Conceitos Fundamentais e Aplicações

## 1. Intuição sobre Redes Neurais
### 1.1 Origem e Princípios Biológicos
- **Inspiração Neural**:  
  Modelagem inicial baseada em neurônios biológicos (dendritos, axônios, sinapses).  
  Objetivo original: replicar processos cognitivos humanos (aprendizado e tomada de decisão).

- **Evolução Histórica**:  
  - **1950s**: Primeiros modelos matemáticos de neurônios artificiais.  
  - **1980-90s**: Aplicações práticas em reconhecimento de padrões (dígitos manuscritos, processamento de cheques).  
  - **Pós-2005**: Revolução do *deep learning* com foco em escalabilidade e otimização computacional.

### 1.2 Neurônio Artificial vs. Biológico
| **Neurônio Biológico**       | **Neurônio Artificial**         |
|------------------------------|----------------------------------|
| Dendritos (entradas)          | Vetor de características (`x`)  |
| Núcleo (processamento)        | Função de ativação (`g(z)`)     |
| Axônio (saída)                | Ativação (`a = g(w·x + b)`)     |

- **Limitações da Analogia**:  
  Modelos atuais priorizam eficiência computacional sobre fidelidade biológica.

---

## 2. Arquitetura e Funcionamento
### 2.1 Componentes Essenciais
- **Camadas**:  
  - **Entrada (`Input Layer`)**: Dados brutos (ex.: vetores, imagens).  
  - **Oculta (`Hidden Layer`)**: Transformações não-lineares para *feature learning*.  
  - **Saída (`Output Layer`)**: Resultado final (classificação, regressão).

- **Exemplo Prático (Demanda de Camisetas)**:  
  ```python
  # Estrutura simplificada
  input_layer = [preço, custo_envio, marketing, qualidade]
  hidden_layer = [acessibilidade, consciência_marca, qualidade_percebida]
  output_layer = probabilidade_top_seller
  ```

### 2.2 Deep Learning e MLPs
- **Multilayer Perceptron (MLP)**:  
  ```mermaid
  graph LR
    A[Input] --> B[Hidden 1]
    B --> C[Hidden 2]
    C --> D[Output]
  ```
- **Decisões de Projeto**:  
  - Número de camadas: Trade-off entre complexidade e custo computacional.  
  - Neurônios por camada: Redução progressiva (ex.: 64 → 25 → 15 → 1).

---

## 3. Aplicações em Visão Computacional
### 3.1 Reconhecimento Facial
- **Fluxo de Processamento**:  
  1. **Camadas Iniciais**: Detecção de bordas e texturas.  
  2. **Camadas Intermediárias**: Identificação de componentes (olhos, nariz).  
  3. **Camadas Finais**: Reconhecimento de padrões complexos (rostos completos).

- **Adaptabilidade**:  
  - **Exemplo com Carros**:  
    ```mermaid
    graph LR
      E[Bordas] --> F[Peças: portas, rodas]
      F --> G[Forma do carro]
    ```

### 3.2 Vantagens Competitivas
- **Escalabilidade**: Desempenho linear com aumento de dados/hardware (GPUs).  
- **Versatilidade**: Aplicável em saúde (diagnóstico por imagem), automação industrial, e-commerce.

---

## 4. Forward Propagation: Algoritmo de Inferência
### 4.1 Passo a Passo (Exemplo: Dígitos Manuscritos)
1. **Camada 1**:  
   - Entrada: Imagem 8x8 pixels (64 valores).  
   - 25 neurônios → Vetor de ativações `a¹ = [0.2, 0.7, ..., 0.4]`.

2. **Camada 2**:  
   - Entrada: `a¹`.  
   - 15 neurônios → Vetor `a² = [0.5, 0.1, ..., 0.9]`.

3. **Camada 3 (Saída)**:  
   - Entrada: `a²`.  
   - 1 neurônio → Probabilidade `a³ = 0.84` (dígito "1").

### 4.2 Implementação e Otimização
```python
# Pseudocódigo para Forward Propagation
def forward_propagation(x, weights, biases):
    a = x
    for i in range(len(weights)):
        z = np.dot(weights[i], a) + biases[i]
        a = sigmoid(z)
    return a
```

---




# Implementação de Redes Neurais com TensorFlow

## 1. **Inferência em TensorFlow**
### 1.1 Estrutura Básica de uma Rede Neural
- **Exemplo (Torrefação de Café)**:  
  ```python
  # Entrada: temperatura (200°C) e duração (17min)
  x = np.array([[200, 17]])  # Matriz 1x2 (formato TensorFlow)
  
  # Camada 1: 3 neurônios com ativação sigmoide
  layer_1 = tf.keras.layers.Dense(units=3, activation='sigmoid')
  a1 = layer_1(x)  # Saída: matriz 1x3 (ex: [0.2, 0.7, 0.3])
  
  # Camada 2 (Saída): 1 neurônio com sigmoide
  layer_2 = tf.keras.layers.Dense(units=1, activation='sigmoid')
  a2 = layer_2(a1)  # Saída: matriz 1x1 (ex: 0.8)
  
  # Threshold para classificação binária
  y_pred = 1 if a2 >= 0.5 else 0
  ```

### 1.2 Fluxo de Dados e Tensores
- **Tensor vs. NumPy Array**:  
  | **TensorFlow (Tensor)**         | **NumPy (Array)**              |
  |---------------------------------|--------------------------------|
  | `tf.Tensor([[0.8]], shape=(1,1))` | `np.array([[0.8]])`            |
  | Otimizado para GPU/TPU          | Estrutura padrão para cálculos |
  | Conversão: `a2.numpy()`         | Conversão: `tf.constant(x)`    |

---

## 2. **Modelagem com `Sequential API`**
### 2.1 Construção Simplificada da Rede
- **Exemplo (Classificação de Dígitos)**:  
  ```python
  model = tf.keras.Sequential([
      tf.keras.layers.Dense(25, activation='sigmoid', input_shape=(64,)),  # Camada 1: 25 neurônios
      tf.keras.layers.Dense(15, activation='sigmoid'),                     # Camada 2: 15 neurônios
      tf.keras.layers.Dense(1, activation='sigmoid')                       # Camada 3 (Saída)
  ])
  
  # Compilação do modelo (detalhes de treino serão abordados posteriormente)
  model.compile(optimizer='sgd', loss='binary_crossentropy') 
  
  # Treinamento
  model.fit(X_train, y_train, epochs=100)
  
  # Inferência
  X_new = np.array([[0.1, 0.2, ..., 0.9]])  # Exemplo de entrada
  y_pred = model.predict(X_new)              # Saída: probabilidade
  ```

### 2.2 Vantagens do `Sequential`
- **Automatização**: Conecta camadas sequencialmente sem manipulação manual de tensores.  
- **Eficiência**: Gerencia automaticamente formas (*shapes*) e inicialização de parâmetros.  
- **Flexibilidade**: Suporta redes profundas com poucas linhas de código.

---

## 3. **Representação de Dados**
### 3.1 NumPy vs. TensorFlow
| **Característica**       | **NumPy**                      | **TensorFlow**               |
|--------------------------|--------------------------------|-------------------------------|
| Estrutura                | 1D (vetores) ou 2D (matrizes) | Tensores (n-dimensionais)     |
| Formato Padrão           | `np.array([200, 17])` (1D)    | `tf.constant([[200, 17]])` (2D) |
| Uso em Redes Neurais     | Requer conversão para tensor  | Nativo (operações otimizadas) |

### 3.2 Boas Práticas
- **Entrada sempre em 2D**:  
  ```python 
  x = np.array([[200, 17]])  # Formato correto (1 exemplo, 2 features)
  ```
- **Saídas como Tensores**:  
  - Camadas ocultas: `a1` é um tensor 1x3.  
  - Saída final: `a2` é um tensor 1x1 (classificação binária).

---

## 4. **Casos Práticos**
### 4.1 Torrefação de Café
- **Objetivo**: Prever se temperatura/duração produzem café de qualidade.  
- **Dataset**:  
  - Features: `[[temp1, dur1], [temp2, dur2], ...]` (matriz `n x 2`).  
  - Labels: `[0, 1, 1, 0]` (array 1D).  

### 4.2 Reconhecimento de Dígitos (MNIST)
- **Entrada**: Imagens 8x8 (64 pixels → vetor 1x64).  
- **Arquitetura**:  
  - 3 camadas (25 → 15 → 1 neurônios).  
  - Ativação sigmoide para todas as camadas.  

---




# Implementação Manual de Forward Propagation em Python

## 1. **Implementação de uma Camada (Dense Layer)**
### 1.1 Função para uma Camada Neural
```python
import numpy as np

def dense(a_prev, W, b, activation='sigmoid'):
    """
    Calcula a saída de uma camada neural densa (totalmente conectada).
    
    Args:
        a_prev (np.array): Ativações da camada anterior (shape: (n_prev, 1)).
        W (np.array): Matriz de pesos (shape: (n_prev, n_units)).
        b (np.array): Vetor de biases (shape: (n_units, 1)).
        activation (str): Função de ativação ('sigmoid', 'relu', etc.).
        
    Returns:
        a (np.array): Ativações da camada atual.
    """
    # Cálculo do produto escalar (z = W · a_prev + b)
    z = np.dot(W.T, a_prev) + b  # W.T para alinhar dimensões
    
    # Aplicação da função de ativação
    if activation == 'sigmoid':
        a = 1 / (1 + np.exp(-z))
    elif activation == 'relu':
        a = np.maximum(0, z)
    else:
        raise ValueError("Função de ativação não suportada.")
    
    return a
```

### 1.2 Exemplo de Uso para uma Camada
```python
# Parâmetros da camada (3 neurônios)
W = np.array([[0.2, -0.5, 0.1],   # Pesos (2 features de entrada → 3 neurônios)
              [0.3, 0.4, -0.7]])
b = np.array([[-1.0], [0.5], [0.2]])  # Biases (3 neurônios)

# Entrada (exemplo)
a_prev = np.array([[200], [17]])  # Formato (2, 1)

# Forward propagation na camada
a_layer = dense(a_prev, W, b)
print("Ativações da camada:\n", a_layer)
```

---

## 2. **Rede Neural com Múltiplas Camadas**
### 2.1 Implementação Completa
```python
def forward_propagation(X, params):
    """
    Executa forward propagation em uma rede neural com arquitetura definida.
    
    Args:
        X (np.array): Dados de entrada (shape: (n_features, n_examples)).
        params (dict): Dicionário com pesos e biases de cada camada.
        
    Returns:
        a_final (np.array): Saída da rede.
    """
    a = X
    num_layers = len(params) // 2  # Número de camadas (W1, b1, W2, b2...)
    
    for l in range(1, num_layers + 1):
        W = params[f'W{l}']
        b = params[f'b{l}']
        a = dense(a, W, b, activation='sigmoid')  # Ativação sigmoide para todas as camadas
    
    return a
```

### 2.2 Exemplo com Duas Camadas
```python
# Parâmetros da rede (2 camadas)
params = {
    'W1': np.array([[0.2, -0.5, 0.1], [0.3, 0.4, -0.7]]),  # Camada 1: 3 neurônios
    'b1': np.array([[-1.0], [0.5], [0.2]]),
    'W2': np.array([[0.1, -0.3], [0.4, 0.2], [-0.5, 0.6]]),  # Camada 2: 2 neurônios
    'b2': np.array([[0.1], [-0.2]])
}

# Entrada (2 features, 1 exemplo)
X = np.array([[200], [17]])

# Forward propagation
a_final = forward_propagation(X, params)
print("Saída final da rede:\n", a_final)
```

---

## 3. **Intuição por Trás de Frameworks como TensorFlow/PyTorch**
### 3.1 Comparação com Implementação Manual
| **Operação**               | **Implementação Manual**                 | **TensorFlow/PyTorch**                   |
|----------------------------|------------------------------------------|------------------------------------------|
| **Cálculo de Ativações**    | Loops explícitos ou operações matriciais | Operações vetorizadas e otimizadas (GPU) |
| **Gestão de Parâmetros**    | Dicionários ou variáveis separadas       | Tensores com autograd (backprop automático) |
| **Funções de Ativação**     | Implementadas manualmente                | Funções pré-construídas (`tf.nn.sigmoid`) |

### 3.2 Vantagens dos Frameworks
- **Vectorização**: Cálculos em lote (batch) otimizados para velocidade.
- **Autograd**: Cálculo automático de gradientes para backpropagation.
- **Abstração**: APIs de alto nível (ex.: `Sequential`) simplificam a construção.

---

## 4. **Exemplo Completo (Coffee Roasting)**
```python
# Parâmetros do exemplo (2 camadas)
params_coffee = {
    'W1': np.array([[ 0.2, -3.7,  1.1 ], 
                   [-0.5,  4.0, -2.2 ]]),
    'b1': np.array([[-1.0], [0.5], [0.2]]),
    'W2': np.array([[-0.7, 1.2, -0.5]]),
    'b2': np.array([[0.3]])
}

# Entrada: temperatura=200°C, duração=17min
X_coffee = np.array([[200], [17]])

# Forward propagation
a1 = dense(X_coffee, params_coffee['W1'], params_coffee['b1'])
a2 = dense(a1, params_coffee['W2'], params_coffee['b2'])

print(f"Probabilidade de café bom: {a2[0][0]:.2f}")  # Exemplo: 0.83
```

---

## 5. **Conclusão**
Implementar manualmente o forward propagation:
- **Prós**: Entendimento profundo do funcionamento interno de redes neurais.
- **Contras**: Inviável para redes complexas (grande esforço de codificação e otimização).

Frameworks como **TensorFlow** e **PyTorch** abstraem esses detalhes, permitindo focar no design da arquitetura e no treinamento. No entanto, compreender a implementação manual é crucial para:
- Debuggar modelos.
- Customizar camadas ou funções de perda.
- Entender limitações e trade-offs de performance.
