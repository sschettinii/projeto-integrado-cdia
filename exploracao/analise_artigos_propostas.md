# Análise de Artigos e Propostas de Pesquisa — Projeto Integrado em Ciência de Dados e IA

---

## Mapeamento dos 15 Artigos Analisados

| # | Artigo | Tema Central | Eixo |
|---|--------|-------------|------|
| 1 | Leo & Kalita (2024) — *Neurocomputing* | Incremental/Continual Learning em DL | Data Streams |
| 2 | Salehi et al. (2021) — *arXiv* | Anomaly, Novelty, Open-Set, OOD Detection | Novelty/OOD |
| 3 | Faria et al. (2016) — *AI Review* | Novelty Detection em Data Streams | Data Streams |
| 4 | SLR (2024) — *ACM Comp. Surveys* | Novelty Detection em Data Streams (SLR) | Data Streams |
| 5 | Cacciarelli & Kulahci (2024) — *Machine Learning* | Active Learning para Data Streams | Active Learning |
| 6 | Krawczyk et al. (2017) — *Information Fusion* | Ensemble Learning para Data Streams | Data Streams |
| 7 | Ren et al. (2021) — *ACM Comp. Surveys* | Deep Active Learning | Active Learning |
| 8 | Li et al. (2024) — *IEEE TNNLS* | Deep Active Learning: avanços e fronteiras | Active Learning |
| 9 | Settles (2009) — *UW-Madison* | Active Learning Literature Survey | Active Learning |
| 10 | Review (2023) — *Micromachines* | AI em Sistemas Embarcados | Embedded AI |
| 11 | Cordova-Cardenas et al. (2025) — *Electronics* | Edge AI: Deploy de NNs em Embarcados | Embedded AI |
| 12 | Dini et al. (2024) — *Electronics* | AI/ML em IIoT Embarcado | Embedded AI |
| 13 | Survey (2024) — *Information Fusion* | Multimodal Hybrid DL para Computer Vision | Computer Vision |
| 14 | Carrilho et al. (2024) — *Electronics* | Detecção de Defeitos com CV | Computer Vision |
| 15 | Nafea et al. (2024) — *BJML* | ML/DL Supervisionado em CV | Computer Vision |

---

# PARTE A — Pesquisas Aplicadas a um Domínio Específico

## A1 — Classificação de Doenças em Folhas de Plantas: Treinamento Eficiente com Active Learning e Deploy no Edge

**Domínio:** Agricultura de Precisão
**Eixos:** Computer Vision + Active Learning + Embedded AI
**Usa RPi 5 + Câmera:** ✅ Sim

### Descrição detalhada

A ideia central é investigar como **active learning pode reduzir a quantidade de rótulos necessários** para treinar um classificador de doenças em folhas de plantas, e se o modelo resultante pode ser comprimido e executado em tempo real no Raspberry Pi 5 com a Pi Camera Module 3.

O projeto tem três etapas bem definidas:

1. **Treinamento com Active Learning (no PC):** Partindo de uma rede leve (MobileNetV3-Small ou EfficientNet-Lite0), vocês treinam inicialmente com apenas 10-15% das imagens rotuladas do PlantVillage. A cada ciclo de AL, o modelo seleciona as imagens sobre as quais tem mais incerteza (uncertainty sampling via entropia do softmax), essas imagens são "rotuladas pelo oráculo" (na prática, vocês revelam o rótulo real do dataset), e o modelo é re-treinado. Vocês repetem isso por 5-8 ciclos até atingir convergência. O resultado é uma **curva de aprendizado** mostrando acurácia vs. % de rótulos utilizados, comparando AL com seleção aleatória.

2. **Compressão e Deploy (RPi 5):** O modelo treinado é quantizado (FP32 → INT8 via TFLite) e deployado no RPi 5. Vocês medem FPS, latência por inferência, uso de RAM e acurácia pós-quantização.

3. **Demo com a câmera:** Vocês coletam fotos de folhas reais (saudáveis e com sintomas visíveis) no campus ou em jardins, e demonstram o sistema classificando em tempo real. Isso **não exige acesso a ambientes restritos** — folhas com doenças comuns (manchas, murchamento) são facilmente encontráveis ao ar livre, e o PlantVillage cobre doenças de tomate, batata, milho e outras plantas comuns.

A contribuição do artigo é dupla: (1) demonstrar que AL reduz significativamente o custo de rotulagem em agricultura de precisão, e (2) validar que o modelo resultante é viável em hardware embarcado de baixo custo.

### Motivação (lacunas nos artigos)

- Li et al. (Art. 8) e Settles (Art. 9) mostram que AL reduz custo de rotulagem, mas **nenhum avalia em pipeline completo de edge deploy**
- Cordova-Cardenas et al. (Art. 11) propõem framework de deploy em edge mas **não integram AL no processo de treinamento**
- Nafea et al. (Art. 15) revisam CV supervisionado mas não abordam cenários com rótulos escassos

### Datasets

- [PlantVillage](https://github.com/spMohanty/PlantVillage-Dataset): 54.306 imagens, 38 classes (14 espécies, saudável + doenças). Totalmente público, sem restrição de uso
- [Plant Pathology 2021 (Kaggle)](https://www.kaggle.com/c/plant-pathology-2021-fgvc8): 18k imagens de folhas de maçã. Complementar

### Principais dificuldades e riscos

| Dificuldade | Severidade | Mitigação |
|-------------|-----------|-----------|
| PlantVillage tem imagens "limpas" (fundo uniforme) que diferem de fotos reais da câmera | ⚠️ Média | Usar data augmentation agressivo no treinamento. Na demo, fotografar folhas contra fundo claro |
| Ciclo de AL pode não mostrar ganho expressivo se o dataset for "fácil demais" | ⚠️ Média | Usar subconjuntos difíceis (apenas classes visualmente semelhantes). Testar também com Plant Pathology 2021 que é mais desafiador |
| Quantização INT8 pode degradar acurácia significativamente em classes raras | ⚠️ Baixa | Usar quantization-aware training (QAT) ao invés de post-training quantization. EfficientNet-Lite é robusto a quantização |
| Encontrar folhas doentes reais para a demo | ⚠️ Baixa | PlantVillage cobre doenças muito comuns (manchas, ferrugem). Mesmo no campus, é possível encontrar. Caso contrário, a demo com imagens do dataset no monitor já valida o deploy |

---

## A2 — Detecção e Contagem de Veículos em Tempo Real com Otimização de Deploy no Raspberry Pi 5

**Domínio:** Mobilidade Urbana / Smart Cities
**Eixos:** Computer Vision + Embedded AI
**Usa RPi 5 + Câmera:** ✅ Sim

### Descrição detalhada

O objetivo é construir um **sistema de contagem e classificação de veículos** (carro, moto, ônibus, caminhão, bicicleta) operando em tempo real no RPi 5, posicionando a Pi Camera em uma janela ou passarela com vista para uma rua. O projeto investiga qual combinação de modelo + compressão oferece o melhor trade-off entre acurácia de detecção e velocidade de inferência neste hardware.

O projeto tem três frentes:

1. **Seleção e fine-tuning de modelo (no PC):** Vocês partem do YOLOv8n pré-treinado no COCO (que já detecta veículos) e fazem fine-tuning opcional no UA-DETRAC (dataset específico de trânsito) para melhorar detecção de motos e ônibus. Testam também SSD-MobileNetV2 como alternativa mais leve.

2. **Benchmark de compressão no RPi 5:** Cada modelo é exportado em 3 formatos (FP32, FP16, INT8) e testado no RPi 5. Vocês medem mAP@0.5, FPS, latência por frame e uso de memória. Isso gera uma **tabela de benchmark inédita para o RPi 5**.

3. **Sistema completo com tracking e contagem:** O melhor modelo é integrado com ByteTrack (tracker leve) e uma linha virtual de contagem. A Pi Camera filma uma rua e o sistema conta veículos passando. Um dashboard web simples (Flask) exibe contagem em tempo real.

A contribuição é prática: provar que um sistema de ~R$500 pode substituir equipamentos comerciais de ~R$10k+ para contagem de tráfego, e documentar os trade-offs de performance no RPi 5.

### Motivação

- Review Micromachines (Art. 10) destaca RPi como plataforma viável mas **faltam benchmarks práticos e atualizados** (RPi 5 é de 2023)
- Cordova-Cardenas et al. (Art. 11) propõem framework de deploy mas **não testam com detecção de objetos em tempo real**
- A survey de Information Fusion (Art. 13) sobre deep learning multimodal não avalia restrições de hardware de baixo custo

### Datasets

- [COCO 2017](https://cocodataset.org/): modelo base pré-treinado. Contém classes de veículos
- [UA-DETRAC](https://detrac-db.rit.albany.edu/): 140k frames de vídeo de trânsito com anotações de veículos. Público
- Captura própria: basta uma janela com vista para rua movimentada

### Principais dificuldades e riscos

| Dificuldade | Severidade | Mitigação |
|-------------|-----------|-----------|
| FPS pode ser baixo demais para contagem confiável (abaixo de 10 FPS) | ⚠️ Média | YOLOv8n INT8 atinge ~15-20 FPS no RPi 5 segundo benchmarks da comunidade. Se insuficiente, usar resolução 320×320 ao invés de 640×640 |
| Câmera posicionada em ângulo desfavorável prejudica detecção | ⚠️ Média | Escolher local com vista elevada (2º andar). Testar ângulos antes de coletar dados |
| Condições de iluminação variáveis (noite, contraluz) | ⚠️ Média | Pi Camera Module 3 tem HDR. Focar a avaliação em horário diurno. Mencionar limitação noturna como trabalho futuro |
| Contagem pode ter erros por oclusão entre veículos | ⚠️ Baixa | ByteTrack lida razoavelmente com oclusão curta. Erros de contagem são esperados e devem ser documentados como margem |

---

## A3 — Detecção de Intrusão em Redes de Computadores com Active Learning em Data Streams

**Domínio:** Cibersegurança
**Eixos:** Data Streams + Active Learning + Novelty Detection
**Usa RPi 5 + Câmera:** ❌ Não

### Descrição detalhada

Este projeto trata o tráfego de rede como um **data stream** onde novos tipos de ataques (novelties) surgem ao longo do tempo, e a rotulagem de fluxos como "ataque" ou "normal" é cara (exige analista de segurança). A pergunta central é: **como usar active learning para selecionar os fluxos de rede mais informativos para rotulagem, de forma a maximizar a detecção de ataques (incluindo novos) com o menor custo possível?**

O projeto funciona inteiramente no PC/notebook com datasets públicos:

1. **Simulação de stream:** O dataset CIC-IDS2017 (ou UNSW-NB15) é tratado como stream temporal — os dados chegam em ordem cronológica e o classificador é atualizado incrementalmente. Classes de ataque que aparecem apenas em dias posteriores simulam **novelty**.

2. **Active Learning com budget:** A cada batch de N fluxos, o classificador pode pedir rótulo para apenas K deles (K << N). Vocês testam diferentes estratégias de seleção: aleatória, uncertainty sampling, query-by-committee, e uma proposta híbrida que prioriza amostras que o detector de novelty considera "desconhecidas".

3. **Avaliação:** Métricas de detecção (F1, precision, recall por classe de ataque), custo total de queries, e análise temporal de quando cada tipo de ataque é detectado.

A contribuição é combinar, em um problema concreto, três técnicas usualmente tratadas de forma isolada: classificação em streams, active learning com budget, e detecção de classes novas.

### Motivação

- Faria et al. (Art. 3) mencionam detecção de intrusão como aplicação natural de novelty detection em streams
- Cacciarelli & Kulahci (Art. 5) destacam que **AL em streams é pouco testado em aplicações de segurança**
- SLR (Art. 4) identifica que métodos de novelty detection não são avaliados com restrição de rotulagem
- Salehi et al. (Art. 2) argumentam que anomaly detection e novelty detection precisam de avaliação unificada

### Datasets

- [CIC-IDS2017](https://www.unb.ca/cic/datasets/ids-2017.html): ~2.8M fluxos, 14 tipos de ataque ao longo de 5 dias. Amplamente usado, público
- [UNSW-NB15](https://research.unsw.edu.au/projects/unsw-nb15-dataset): 2.5M registros, 9 tipos de ataque. Complementar
- [NSL-KDD](https://www.unb.ca/cic/datasets/nsl.html): versão limpa do KDD Cup 99. Útil para validação cruzada

### Principais dificuldades e riscos

| Dificuldade | Severidade | Mitigação |
|-------------|-----------|-----------|
| CIC-IDS2017 tem classes desbalanceadas (ataques raros) | ⚠️ Média | Usar métricas por classe (macro-F1), não acurácia global. SMOTE ou oversampling para classes raras no buffer de treino |
| Simular um stream a partir de dataset estático pode não capturar concept drift real | ⚠️ Média | Usar a ordem temporal real do dataset (dia 1, dia 2...). Inserir drift sintético adicional se necessário |
| Implementar o loop de AL + novelty + classificação incremental do zero é complexo | ⚠️ Alta | Usar o framework [River](https://riverml.xyz/) que já tem classificadores incrementais e drift detectors. Foquem na estratégia de query, não na infraestrutura |
| Definir o que conta como "novelty" vs. "ruído" em features de rede | ⚠️ Média | Usar a divisão temporal do dataset: ataques que aparecem apenas nos dias 3-5 são definidos como novelties |

---

## A4 — Classificação de Imagens Médicas com Active Learning: Reduzindo Custo de Anotação por Especialistas

**Domínio:** Saúde / Diagnóstico Assistido
**Eixos:** Computer Vision + Active Learning + Deep Learning
**Usa RPi 5 + Câmera:** ❌ Não

### Descrição detalhada

Em problemas médicos, obter rótulos é extraordinariamente caro — cada imagem precisa ser analisada por um especialista. Este projeto investiga **quão poucos rótulos são realmente necessários** para treinar um classificador de lesões dermatológicas com acurácia clínica aceitável, usando active learning para selecionar as imagens mais informativas.

O projeto roda inteiramente no PC (idealmente com GPU, mas funciona com CPU):

1. **Setup experimental:** Vocês usam o dataset HAM10000 (10.015 imagens dermatológicas, 7 classes de lesão). Simulam o cenário onde inicialmente apenas 5% das imagens têm rótulo. O oráculo é o próprio dataset (vocês revelam o rótulo real quando uma imagem é "consultada").

2. **Ciclo de Active Learning:** A cada rodada, o modelo (MobileNetV3 ou ResNet-18 com fine-tune do ImageNet) indica as K imagens mais incertas. Vocês testam 4 estratégias: random baseline, uncertainty sampling (entropia), query-by-committee (ensemble de 3 modelos), e BADGE (batch-mode diversity). Cada estratégia é avaliada em curvas de aprendizado: acurácia vs. % de rótulos usados.

3. **Análise crítica:** Além das curvas, vocês analisam **quais classes se beneficiam mais de AL** (classes raras vs. comuns?), e se AL introduz bias (ex: foco excessivo em classes difíceis mas raras).

A contribuição é um estudo experimental rigoroso sobre eficiência de rotulagem em classificação médica, com análise por classe e por estratégia.

### Motivação

- Li et al. (Art. 8) identificam que **DAL em imagens médicas é um dos campos mais promissores, mas carece de estudos comparativos de estratégias**
- Settles (Art. 9) formaliza que AL pode reduzir em 50-90% a necessidade de labels — mas isso precisa ser validado por domínio
- Ren et al. (Art. 7) catalogam DAL mas a maioria dos experimentos usa CIFAR/MNIST — **faltam avaliações em datasets médicos reais**

### Datasets

- [HAM10000](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T): 10.015 imagens de lesões de pele, 7 classes. Público e bem documentado
- [ISIC 2019](https://challenge.isic-archive.com/landing/2019/): 25k imagens dermatológicas. Extensão do HAM10000
- [Chest X-Ray Pneumonia (Kaggle)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia): 5.8k radiografias. Para validação cruzada em outro domínio médico

### Principais dificuldades e riscos

| Dificuldade | Severidade | Mitigação |
|-------------|-----------|-----------|
| HAM10000 é muito desbalanceado (dermatofibroma tem ~115 imagens vs. nevos com ~6.700) | ⚠️ Alta | Usar macro-F1 e balanced accuracy. Estratificar o pool de AL. Reportar resultados por classe |
| Sem GPU, o treinamento de múltiplos ciclos de AL com ResNet-18 pode ser lento | ⚠️ Média | Usar MobileNetV3-Small (treinamento ~3x mais rápido). Google Colab como alternativa (T4 gratuita). Reduzir resolução para 128×128 |
| AL pode "convergir" rápido e as diferenças entre estratégias serem estatisticamente insignificantes | ⚠️ Média | Repetir cada experimento 5x com seeds diferentes. Usar teste estatístico (Wilcoxon). Analisar em regime de poucos rótulos (5-15%) onde as diferenças são maiores |
| Revisores podem questionar relevância clínica sem validação com médicos reais | ⚠️ Baixa | Enquadrar como estudo de eficiência de rotulagem, não como ferramenta clínica. Comparar com baselines publicados no ISIC Challenge |

---

# PARTE B — Pesquisas Não-Aplicadas (Contribuição Metodológica)


## B1 — Active Learning Combinado com Detecção de Novelty em Data Streams

**Eixos:** Active Learning + Data Streams + Novelty Detection
**Usa RPi 5 + Câmera:** ❌ Não

### Descrição detalhada

Na literatura de data streams, dois problemas são estudados de forma isolada: (1) **active learning** — escolher quais instâncias rotular quando há restrição de budget, e (2) **novelty detection** — identificar quando uma instância pertence a uma classe nunca vista. Este projeto propõe e avalia uma **estratégia unificada** que faz as duas coisas simultaneamente.

A intuição é simples: em um stream onde novas classes podem surgir, o classificador precisa decidir não só "sobre qual instância perguntar o rótulo?" mas também "esta instância pertence a algo que eu nunca vi?". A estratégia proposta funciona assim:

1. **Classificador base incremental** (Hoeffding Tree ou Adaptive Random Forest via River).
2. **Módulo de novelty:** calcula a distância da instância aos centroides dos micro-clusters conhecidos. Se acima de um threshold, marca como possível novelty.
3. **Estratégia de query unificada:** instâncias marcadas como novelty recebem **prioridade máxima** para query ao oráculo. Entre instâncias in-distribution, aplica uncertainty sampling padrão.
4. **Avaliação:** Comparar com (a) AL puro (sem consciência de novelty), (b) novelty detection puro (sem budget), (c) baseline aleatório. Métricas: F1 para classes conhecidas, F1 para classes novas, custo total de queries, latência de detecção de novas classes.

O experimento é rodado em streams sintéticos (onde vocês controlam quando novas classes aparecem) e em streams reais (Forest Covertype, KDD).

### Motivação

- Cacciarelli & Kulahci (Art. 5): AL em streams **ignora completamente o surgimento de novas classes**
- Faria et al. (Art. 3) e SLR (Art. 4): novelty detection em streams é tratada **sem restrição de budget de rotulagem**
- **Nenhum dos 15 artigos analisados combina esses dois problemas.** É a lacuna mais clara e explícita do levantamento

### Datasets

- **Sintéticos (MOA):** Rotating Hyperplane, RandomRBF com inserção controlada de classes novas em timestamps definidos
- **Reais:** [Forest Covertype](https://archive.ics.uci.edu/ml/datasets/covertype) (581k instâncias, 7 classes), [Electricity](https://www.openml.org/d/151), [KDD Cup 99](http://kdd.ics.uci.edu/databases/kddcup99/)

### Principais dificuldades e riscos

| Dificuldade | Severidade | Mitigação |
|-------------|-----------|-----------|
| Definir o threshold de novelty de forma não-supervisionada é difícil | ⚠️ Alta | Usar threshold adaptativo baseado no desvio padrão das distâncias recentes. Testar sensibilidade a diferentes valores |
| O método pode ser simplesmente "AL que trata outliers como novelty" — contribuição incremental | ⚠️ Média | Formalizar a estratégia de priorização. Mostrar experimentalmente que novelty-aware AL detecta novas classes X% mais rápido e com Y% menos queries que AL puro |
| Datasets reais podem não ter "novelty" natural clara (todas as classes aparecem desde o início) | ⚠️ Média | Nos datasets reais, retirar 2-3 classes do treinamento e introduzi-las apenas na metade do stream, simulando novelty |
| Implementar o pipeline integrado do zero | ⚠️ Média | Usar River como base. Micro-clusters via CluStream (já implementado em River). Foquem na estratégia de query, não na infra |

---

## B2 — Deep Active Learning sob Concept Drift: Como Estratégias de Query se Degradam em Ambientes Não-Estacionários

**Eixos:** Active Learning + Data Streams + Deep Learning
**Usa RPi 5 + Câmera:** ❌ Não

### Descrição detalhada

Todas as estratégias de Deep Active Learning (DAL) catalogadas nos surveys (uncertainty, QBC, core-set, BADGE, learning loss) são avaliadas em cenários **estacionários**: a distribuição dos dados de treino e teste é a mesma. Este projeto investiga o que acontece quando **a distribuição muda ao longo do tempo (concept drift)** — algo inevitável em cenários reais.

O projeto tem duas fases:

1. **Fase diagnóstica — "O que quebra?":** Vocês implementam 3-4 estratégias de DAL populares (uncertainty sampling, QBC com ensemble, BADGE, random) usando uma MLP ou CNN leve. Treinam em streams com drift sintético (rotating hyperplane, SEA com drift controlado, CIFAR-10 com permutação de classes ao longo do tempo). Medem como a acurácia de cada estratégia se degrada com diferentes tipos e intensidades de drift.

2. **Fase propositiva — "Como consertar?":** Vocês propõem uma adaptação simples: acoplar um **detector de drift** (ADWIN ou Page-Hinkley) ao pipeline de DAL. Quando drift é detectado, o sistema (a) aumenta temporariamente o budget de queries, e/ou (b) invalida o pool de amostras pré-drift. Avaliam se essa intervenção melhora a performance.

A contribuição é dupla: (1) evidência experimental de que DAL padrão falha sob drift (ninguém documentou isso sistematicamente), e (2) uma proposta simples de adaptação drift-aware.

### Motivação

- Li et al. (Art. 8) e Ren et al. (Art. 7): catalogam ~200 papers de DAL — **zero tratam concept drift**
- Krawczyk et al. (Art. 6): ensembles adaptam-se a drift, mas **sem integração com active learning**
- Cacciarelli & Kulahci (Art. 5): mencionam drift como "desafio aberto" para AL em streams

### Datasets

- **Streams tabulares com drift:** Rotating Hyperplane (drift gradual), SEA Concepts (drift abrupto), Agrawal (drift incremental) — todos via MOA generators
- **Streams visuais com drift sintético:** CIFAR-10 onde as 10 classes são divididas em 3 "tarefas" temporais (classes 0-3, depois 4-6, depois 7-9)
- **Stream real:** [Electricity](https://www.openml.org/d/151) (drift natural por mudanças sazonais)

### Principais dificuldades e riscos

| Dificuldade | Severidade | Mitigação |
|-------------|-----------|-----------|
| Drift sintético pode ser "artificial demais" e não representar problemas reais | ⚠️ Média | Incluir pelo menos 1 dataset real (Electricity). Testar diferentes intensidades de drift sintético |
| CNN sobre CIFAR-10 como stream é computacionalmente caro para muitos ciclos de AL | ⚠️ Alta | Usar MLP sobre features extraídas (penúltima camada de ResNet-18 pré-treinado). Ou usar Google Colab. Ou focar experimentos em streams tabulares e fazer CIFAR como experimento complementar |
| A proposta de "aumentar budget quando drift detectado" pode ser vista como trivial | ⚠️ Média | Testar também invalidação de pool, reset do modelo, e combinações. Formalizar como algoritmo com pseudocódigo. Comparar com "re-treinar do zero" como baseline |
| Muitos hiperparâmetros: tipo de drift × estratégia de AL × detector de drift × modelo base | ⚠️ Média | Fixar modelo base e detector de drift, variar apenas tipo de drift × estratégia de AL. Ter design experimental claro desde o início |

---

## B3 — Benchmark de Técnicas de Compressão de Modelos de Classificação Visual no Raspberry Pi 5

**Eixos:** Embedded AI + Computer Vision
**Usa RPi 5 + Câmera:** ✅ Sim (para demo, mas o core do experimento é benchmark com datasets)

### Descrição detalhada

O Raspberry Pi 5 (lançado em 2023) é significativamente mais poderoso que seus predecessores, mas **não existe benchmark publicado** que avalie sistematicamente técnicas de compressão de redes neurais de classificação de imagens neste hardware. Este projeto preenche essa lacuna.

O projeto é um estudo experimental rigoroso:

1. **Modelos base (3):** MobileNetV3-Small, EfficientNet-Lite0, ShuffleNetV2-x1.0 — todos pré-treinados no ImageNet.

2. **Técnicas de compressão (4):**
   - **(a) Quantização:** FP32 → FP16 → INT8 → INT4 (post-training e quantization-aware)
   - **(b) Pruning:** structured (filtros inteiros) e unstructured (pesos individuais), com taxas de 20%, 40%, 60%
   - **(c) Knowledge Distillation:** EfficientNet-B4 como teacher → cada modelo como student
   - **(d) Combinações:** pruning + quantização, distillation + quantização

3. **Métricas no RPi 5:** Top-1 accuracy, latência por inferência (ms), throughput (FPS), RAM de pico, consumo energético estimado (via `vcgencmd measure_temp` + modelagem térmica como proxy)

4. **Análise:** Fronteira de Pareto (quais variantes são dominadas vs. ótimas), recomendações por cenário de uso (priorizar acurácia vs. priorizar velocidade)

O resultado é um **guia prático com tabelas e gráficos** que qualquer pesquisador ou desenvolvedor pode consultar ao deployar modelos no RPi 5. A demo com a câmera serve para ilustrar o sistema funcionando, mas não é o core do experimento.

### Motivação

- Review Micromachines (Art. 10): lista técnicas de compressão mas **nenhum benchmark é feito no RPi 5**
- Cordova-Cardenas et al. (Art. 11): framework de deploy focado em MCUs, não em SBCs como RPi
- Dini et al. (Art. 12): destaca necessidade de avaliar AI em hardware atualizado — RPi 5 é o SBC mais popular do mercado

### Datasets

- [ImageNette](https://github.com/fastai/imagenette): subset de 10 classes do ImageNet, fácil de baixar (1.6GB). Benchmark padrão para compressão
- [CIFAR-100](https://www.cs.toronto.edu/~kriz/cifar.html): 60k imagens, 100 classes. Complementar
- [Food-101](https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/): 101k imagens, 101 classes. Para testar generalização

### Principais dificuldades e riscos

| Dificuldade | Severidade | Mitigação |
|-------------|-----------|-----------|
| O espaço de combinações (3 modelos × 4 técnicas × múltiplas configurações) é enorme | ⚠️ Alta | Definir design experimental fixo na semana 1. Priorizar quantização (maior impacto prático). Pruning e KD como extensões |
| Medição de consumo energético no RPi 5 não é trivial (não tem sensor nativo) | ⚠️ Média | Usar temperatura como proxy (correlação com consumo). Ou usar multímetro USB (~R$30). Ou omitir energia e focar em latência/RAM |
| Contribuição pode ser vista como "apenas tabelas" sem avanço metodológico | ⚠️ Média | Adicionar análise de Pareto, recomendações concretas, e demo com câmera. Posicionar como "resource paper" ou "benchmarking study" |
| Resultados podem variar entre execuções no RPi (throttling térmico) | ⚠️ Baixa | Fazer warm-up de 50 inferências antes de medir. Rodar cada benchmark 3x e reportar média + desvio. Usar dissipador de calor |

---

## B4 — Estudo Comparativo de Algoritmos de Novelty Detection sob Diferentes Tipos de Concept Drift em Data Streams

**Eixos:** Data Streams + Novelty Detection
**Usa RPi 5 + Câmera:** ❌ Não

### Descrição detalhada

Algoritmos de novelty detection em data streams (MINAS, ECSMiner, CLAM, etc.) são tipicamente avaliados em 1-2 cenários experimentais, com tipos específicos de drift. Mas **como cada algoritmo se comporta diante de cada tipo de concept drift?** Este projeto faz uma análise comparativa cruzada e sistemática.

O projeto é essencialmente um estudo experimental controlado:

1. **Algoritmos avaliados (4-5):** MINAS (Faria et al.), ECSMiner (Masud et al.), CLAM, e pelo menos 1 método recente do SLR de 2024 (Art. 4). Se possível, implementar uma variante simplificada de autoencoder incremental.

2. **Tipos de drift gerados (4):**
   - **Abrupto:** em um timestamp definido, a distribuição muda instantaneamente
   - **Gradual:** a distribuição muda progressivamente ao longo de uma janela
   - **Incremental:** mudanças pequenas e contínuas sem ponto de transição claro
   - **Recorrente:** conceitos antigos reaparecem após um período de mudança

3. **Métricas padronizadas:** F-measure para classes conhecidas, F-measure para classes novas, latência de detecção (quantos timesteps até detectar a novelty), taxa de falso positivo (noise classificado como novelty)

4. **Resultado:** Matriz de recomendação: *"Para drift tipo X, use método Y"*

### Motivação

- SLR (Art. 4): critica que a maioria dos trabalhos é avaliada em cenários restritos
- Faria et al. (Art. 3): propõem métricas mas a comunidade não convergiu para protocolo padrão
- Salehi et al. (Art. 2): fragmentação entre anomaly/novelty/OOD impede comparações

### Datasets

- **Sintéticos (controle total):** MOA generators — Rotating Hyperplane (gradual), RandomRBF (abrupto), Agrawal (incremental). Para recorrente: alternar entre 2 configurações do mesmo generator
- **Reais:** [Forest Covertype](https://archive.ics.uci.edu/ml/datasets/covertype), [Keystroke Dynamics](https://www.cs.cmu.edu/~keystroke/), [KDD Cup 99](http://kdd.ics.uci.edu/databases/kddcup99/)

### Principais dificuldades e riscos

| Dificuldade | Severidade | Mitigação |
|-------------|-----------|-----------|
| Implementar MINAS, ECSMiner etc. do zero é trabalhoso | ⚠️ Alta | Usar implementações existentes: [scikit-multiflow](https://scikit-multiflow.readthedocs.io/), [MOA](https://moa.cms.waikato.ac.nz/). Se não houver implementação pronta, selecionar apenas métodos com código disponível |
| Comparação pode ser injusta se hiperparâmetros não forem bem calibrados | ⚠️ Média | Usar configurações padrão dos papers originais. Reportar hiperparâmetros usados. Fazer análise de sensibilidade para os 2 hiperparâmetros mais importantes de cada método |
| Contribuição pode ser vista como "survey experimental" sem novidade metodológica | ⚠️ Média | Adicionar recomendações concretas (a matriz drift × método). Propor um protocolo de avaliação padronizado que outros podem reusar |
| Gerar drift recorrente de forma controlada exige cuidado no design experimental | ⚠️ Baixa | Usar alternância periódica entre 2 distribuições no MOA. Documentar claramente a configuração |

---
---

# Quadro Comparativo Geral

| # | Proposta | RPi 5 | Tipo | Dificuldade | Originalidade | Viabilidade |
|---|---------|:-----:|------|:-----------:|:-------------:|:-----------:|
| A1 | Doenças em Plantas + AL + Edge | ✅ | Aplicada | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| A2 | Contagem de Veículos RPi 5 | ✅ | Aplicada | ⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| A3 | Intrusão em Redes + AL + Novelty | ❌ | Aplicada | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| A4 | Imagens Médicas + AL | ❌ | Aplicada | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| B1 | AL + Novelty em Streams | ❌ | Metodológica | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| B2 | DAL sob Concept Drift | ❌ | Metodológica | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| B3 | Benchmark Compressão RPi 5 | ✅ | Metodológica | ⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| B4 | Comparativo Novelty × Drift | ❌ | Metodológica | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |

---

# Recomendação Final

> ### Aplicada — Top 2
> | Rank | Proposta | Por que |
> |------|---------|--------|
> | 🥇 | **A1** — Doenças em Plantas + AL + Edge | Viabilidade altíssima (dataset pronto, folhas acessíveis, pipeline claro). Contribuição dupla: AL + Edge. Demo com câmera realista |
> | 🥈 | **A3** — Intrusão em Redes + AL + Novelty | Lacuna mais clara entre as aplicadas. Sem dependência de hardware. Datasets robustos e bem documentados |
>
> ### Não-Aplicada — Top 2
> | Rank | Proposta | Por que |
> |------|---------|--------|
> | 🥇 | **B1** — AL + Novelty em Data Streams | Lacuna mais explícita dos 15 artigos. River facilita prototipação. Resultado experimental comparável e publicável |
> | 🥈 | **B2** — DAL sob Concept Drift | Dois campos maduros que ninguém cruzou. Alto potencial de impacto. Requer mais cuidado no design experimental |
