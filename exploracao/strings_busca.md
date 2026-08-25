# Strings de busca

---

## Instruções para busca
**Google Scholar:** Apenas copiar a string e buscar  
**Scopus:** TITLE-ABS-KEY({string})  
**IEEE Xplore:** Pode usar Field Tags. Pesquisar  
**Web of Science:** TS=({string})  

---

## Strings para reconhecimento

survey "data streams" "concept drift"  
review "novelty detection" "data streams"  
survey "online learning" "concept drift" "catastrophic forgetting"  
"catastrophic forgetting" "concept drift"  
"catastrophic forgetting" "data streams"  
"novelty detection" "catastrophic forgetting"  
"label noise" "data streams"  
"label noise" "concept drift"  
"noisy labels" "online learning"  
"data streams" "novelty detection"  
"data streams" "concept drift"  
"online learning" "concept drift"  
"novelty detection" "concept drift"  
"class-incremental learning" "concept drift"  
"recurring concepts" "data streams"  
"open-set recognition" "data streams"  
"stability-plasticity" "concept drift" 

---

## Strings para busca de artigos


#### Geral
("data stream*" OR "streaming data")
AND
("novelty detection" OR "anomaly detection" OR "outlier detection" OR "unknown class*" OR "unseen class*")
AND
("online learning" OR "incremental learning" OR "sequential learning" OR "adaptive learning")
AND
("concept drift" OR "concept evolution" OR "concept change" OR "distribution shift" OR "non-stationary")

#### Data streams
"data stream*" OR "streaming data" OR "stream mining" OR "stream processing"

#### Novelty detection
"novelty detection" OR "anomaly detection" OR "outlier detection" OR "unknown class*" OR "unseen class*" OR "novel class*"

#### Online Learning
"online learning" OR "incremental learning" OR "sequential learning" OR "adaptive learning" OR "lifelong learning"

#### Concept Drift
"concept drift" OR "concept evolution" OR "concept change" OR "distribution shift" OR "non-stationary environment*"

#### Data Streams + Novelty Detection
("data stream*" OR "streaming data" OR "stream mining")
AND
("novelty detection" OR "anomaly detection" OR "outlier detection" OR "unknown class*" OR "unseen class*")

#### Data Streams + Concept Drift
("data stream*" OR "streaming data" OR "stream mining")
AND
("concept drift" OR "concept evolution" OR "concept change" OR "distribution shift")

#### Data Streams + Online Learning
("data stream*" OR "streaming data" OR "stream mining")
AND
("online learning" OR "incremental learning" OR "sequential learning")

#### Novelty Detection + Concept Drift
("novelty detection" OR "unknown class*" OR "unseen class*" OR "novel class*")
AND
("concept drift" OR "concept evolution" OR "non-stationary environment*")

#### Online Learning + Concept Drift
("online learning" OR "incremental learning" OR "adaptive learning")
AND
("concept drift" OR "concept evolution" OR "distribution shift")

#### Novelty Detection + Online Learning
("novelty detection" OR "unknown class*" OR "unseen class*")
AND
("online learning" OR "incremental learning" OR "sequential learning")

#### Data Streams + Novelty Detection + Concept Drift
("data stream*" OR "streaming data")
AND
("novelty detection" OR "unknown class*" OR "unseen class*" OR "novel class*")
AND
("concept drift" OR "concept evolution" OR "non-stationary environment*")

#### Data Streams + Online Learning + Concept Drift
("data stream*" OR "streaming data")
AND
("online learning" OR "incremental learning" OR "adaptive learning")
AND
("concept drift" OR "concept evolution" OR "distribution shift")

#### Novelty Detection + Online learning + Concept Drift
("novelty detection" OR "unknown class*" OR "unseen class*")
AND
("online learning" OR "incremental learning" OR "sequential learning")
AND
("concept drift" OR "concept evolution" OR "non-stationary environment*")
