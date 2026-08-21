# Proposta de Pesquisa Científica

## 1. Identificação do Projeto

* **Título Proposto (PT):** *Inspeção Visual de Defeitos Industriais com Pipeline Paralelo de Inferência em Dispositivos de Borda Restritos*
* **Título Proposto (EN):** *Visual Industrial Defect Inspection using Parallel Inference Pipelines on Resource-Constrained Edge Devices*
* **Eixos Temáticos:** *Computer Vision* • *Embedded AI* • *Redes Neurais* • *Computação Paralela*
* **Hardware/Ferramentas:** Raspberry Pi 5 (ARM Cortex-A76), Pi Camera, TensorFlow Lite, OpenCV.

---

## 2. Contextualização e Formulação do Problema (*Problem Statement*)

### 2.1. O Cenário Formal
A Detecção de Anomalias Visuais (VAD) para controle de qualidade na Indústria 4.0 exige o processamento de imagens de alta resolução em tempo real. Soluções no estado da arte, como PatchCore e EfficientAD, operam sob o paradigma de aprendizado não supervisionado, sendo treinadas unicamente com imagens normais (sem defeitos). Durante a inferência, elas identificam anomalias baseadas em desvios estatísticos de características (*features*) extraídas. 

Contudo, a avaliação desses modelos na literatura científica quase invariavelmente pressupõe a disponibilidade de GPUs de alto desempenho (e.g., NVIDIA RTX series) ou servidores robustos, ignorando os gargalos práticos do chão de fábrica, como latência de rede, custo de hardware e limitações de energia.

### 2.2. A Lacuna Científica e Tecnológica (*The Scientific and Technological Gap*)
Ao tentar implantar modelos complexos de VAD em hardwares restritos, como Single Board Computers (SBCs) baseados em arquitetura ARM, o processo sequencial de (1) captura, (2) pré-processamento, (3) inferência e (4) pós-processamento torna-se um gargalo inaceitável. O processador frequentemente fica ocioso aguardando o fim do processamento de tensores, reduzindo o *throughput* (frames por segundo - FPS) para níveis incompatíveis com a velocidade de uma esteira industrial. Falta na literatura um estudo empírico robusto sobre o balanceamento de carga e o uso de paralelismo de pipeline intra-dispositivo para inferência de VAD em hardware não-GPU de baixo custo.

**A Pergunta de Pesquisa:** *Como arquitetar um pipeline de processamento multi-thread assíncrono que viabilize a execução em tempo real de modelos avançados de detecção visual de anomalias (como EfficientAD quantizado) em dispositivos de borda com processadores ARM de baixo custo, garantindo throughput compatível com exigências industriais?*

---

## 3. Hipótese Científica e Objetivos

### 3.1. Hipótese
> $H_1$: *A implementação de um pipeline de processamento desacoplado (onde captura, pré-processamento SIMD, inferência quantizada INT8 e geração de mapa de calor operam de forma concorrente em threads dedicadas) em um SBC multicore (como o Raspberry Pi 5) resulta em um aumento de throughput (FPS) superior a 300% em relação à execução sequencial síncrona, mantendo a Área Sob a Curva ROC (AUROC) degradada em no máximo 2% frente aos baselines FP32 executados em hardware especializado.*

### 3.2. Objetivos
* **Objetivo Geral:** Desenvolver, quantizar e implantar um sistema de inspeção visual de defeitos utilizando modelos de última geração em um dispositivo de borda restrito (Raspberry Pi 5), empregando técnicas avançadas de computação paralela para maximizar o throughput.
* **Objetivos Específicos:**
  1. Treinar um modelo de Anomaly Detection (EfficientAD ou versão leve do PatchCore) no dataset MVTec AD e quantizá-lo para INT8 via TFLite.
  2. Implementar um sistema produtor-consumidor multi-thread em Python/C++ para paralelizar os estágios de captura, pré-processamento e inferência.
  3. Realizar benchmarking rigoroso de latência, RAM e AUROC, comparando as versões originais (FP32) com as quantizadas em ambiente de borda.
  4. Validar o sistema de forma prática utilizando peças reais capturadas em tempo real com a Pi Camera.

---

## 4. Metodologia Detalhada por Etapas de Pesquisa

```mermaid
flowchart TD
    subgraph Pipeline Concorrente (Raspberry Pi 5)
        direction LR
        A[Thread 1: Captura \n Pi Camera] -- Frame N --> B((Buffer \n Circular 1))
        
        B -- Frame N-1 --> C[Thread 2: Pré-processamento \n Resize, SIMD NEON Normalize]
        
        C -- Tensor N-2 --> D((Buffer \n Circular 2))
        
        D -- Tensor N-3 --> E[Thread 3: Inferência \n TFLite INT8]
        
        E -- Features N-4 --> F((Buffer \n Circular 3))
        
        F -- Features N-5 --> G[Thread 4: Pós-processamento \n Heatmap & Alerta]
    end
```

### Etapa 1 — Treinamento e Compressão do Modelo (Offline)
- O treinamento do modelo (ex: EfficientAD) será realizado em ambiente não restrito (Desktop/GPU), utilizando o dataset **MVTec AD** (15 categorias, >5.000 imagens).
- Aplicação de **Post-Training Quantization (PTQ)** para conversão dos pesos de ponto flutuante de 32 bits (FP32) para inteiros de 8 bits (INT8).
- Caso a degradação de AUROC seja significativa, será aplicado **Quantization-Aware Training (QAT)**. O modelo final será exportado no formato TFLite.

### Etapa 2 — Desenvolvimento do Pipeline Paralelo (Edge)
- A arquitetura de software será baseada no padrão Produtor-Consumidor usando *queues* assíncronas.
- O pipeline será dividido em quatro estágios distintos, fixando a afinidade de threads (*thread affinity*) aos 4 núcleos físicos do Cortex-A76 do Raspberry Pi 5.
- O gargalo provável (a inferência TFLite) ditará o throughput máximo. O objetivo é garantir que as demais threads nunca deixem a thread de inferência faminta (*starvation*).

### Etapa 3 — Protocolo Experimental e Benchmarks
O sistema será avaliado em dois cenários:
- **Baseline Sequencial:** O RPi processa um frame inteiro (da captura ao alerta) de forma síncrona, usando 1 thread.
- **Pipeline Paralelo:** A execução concorrente proposta.
Serão coletadas métricas de **Latência End-to-End**, **Throughput (FPS)**, **Uso de Memória RAM**, e **Temperatura do SoC** (para avaliar *thermal throttling*).

### Etapa 4 — Validação em Ambiente Controlado
- O sistema será testado na detecção de defeitos de amostras reais e não catalogadas (ex: parafusos enferrujados, tecidos rasgados, peças plásticas com ranhuras) sob um ambiente controlado de iluminação.

---

## 5. Cronograma Executivo de Desenvolvimento (Sprints)

```
[Semana 1-2] Revisão bibliográfica profunda (PatchCore, EfficientAD, TFLite) + Setup do RPi 5
     │
[Semana 3-4] Treinamento do modelo na GPU + Exportação e Quantização (INT8)
     │
[Semana 5-6] Desenvolvimento do pipeline sequencial no RPi 5 (Baseline)
     │
[Semana 7-8] Implementação do paralelismo (Threads, Queues, Buffer Circular)
     │
[Semana 9]   Testes de Benchmarking (Latência, FPS, RAM) + Validação com câmera
     │
[Semana 10]  Redação Final do Relatório Científico e Apresentação
```

---

## 6. Justificativa Estratégica e Impacto da Pesquisa

A ausência de testes em hardwares restritos na literatura acadêmica reflete os incentivos da pesquisa científica, e não a falta de viabilidade comercial. Em conferências de visão computacional, como aquela em que o PatchCore foi apresentado, o objetivo principal dos pesquisadores é alcançar o estado da arte em precisão. Para garantir a publicação de seus artigos, eles rodam os modelos em ambientes controlados com GPUs de alto desempenho, pois a otimização de infraestrutura não é o foco do estudo. A academia prova que o algoritmo funciona matematicamente, mas deixa para a indústria a tarefa de adaptar o modelo para o ambiente real de produção.

Existe um valor de mercado muito alto em trazer esses modelos de detecção de anomalias para dispositivos de borda. A primeira justificativa técnica para isso é a **dependência de rede e a latência**. Em uma linha de produção, dezenas de câmeras capturam imagens de alta resolução em frações de segundo. Transmitir todo esse volume contínuo de dados para servidores centrais consome uma largura de banda massiva e introduz um tempo de trânsito imprevisível. Quando uma peça defeituosa passa pela esteira, o braço robótico ou o pistão pneumático tem um tempo exato para ejetá-la. Se houver oscilação na rede, a resposta atrasa e o defeito passa. Processar a imagem localmente garante um tempo de resposta determinístico.

Outro fator determinante para o mercado é o **custo de escalabilidade**. Colocar um computador industrial com placa de vídeo dedicada em cada ponto de inspeção visual encarece drasticamente o projeto. Isso restringe o uso de inteligência artificial apenas aos produtos de altíssimo valor agregado. Um dispositivo como o Raspberry Pi 5 custa uma fração desse valor e consome muito menos energia. Se um sistema embarcado for capaz de manter uma taxa de precisão aceitável através do paralelismo do pipeline de captura e inferência, a tecnologia de controle de qualidade se torna viável para linhas de produção menores e fábricas de médio porte.

Há também a questão da **privacidade e segurança da informação industrial**. Muitas linhas de montagem lidam com processos proprietários, matrizes exclusivas ou designs confidenciais. Enviar imagens em tempo real para processamento externo cria um risco de segurança que muitas empresas preferem evitar. Manter o dado no dispositivo que o gerou resolve essa restrição de governança.

**Impacto na Formação:** Investir os quatro meses da sua disciplina nesse desenvolvimento é uma decisão bastante estratégica para a sua área de Ciência de Dados e Inteligência Artificial. O mercado de trabalho carece de profissionais que consigam ultrapassar a fase de treinamento de modelos. Desenvolver a capacidade técnica de quantizar redes neurais, otimizar tensores, reduzir o consumo de memória RAM e sincronizar o processamento de imagens utilizando múltiplas threads demonstra competências avançadas de engenharia de software aplicada. O projeto testa os limites físicos do hardware e entrega uma resposta real sobre o quão longe a computação de borda consegue chegar na detecção de defeitos não supervisionada.
