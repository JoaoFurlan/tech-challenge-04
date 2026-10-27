# Validação de Dados e Bibliotecas de Qualidade — resumo do módulo (Aulas 1 a 8)

> Fontes: `Aula 1 - Fundamentos e Importância da Validação de Dados em ML.pdf` …
> `Aula 8 - Validação Contínua com MLOps.pdf`

---

## 0. Fio condutor do módulo

```
Aula 1  POR QUÊ        → Data-Centric AI, 6 dimensões, Data Cascades, custo, viés
Aula 2  TEORIA         → framework ICRA, ISO 25012/5259-4, MCAR/MAR/MNAR, imputação
Aula 3  SUJEIRA        → duplicatas, LSH/MinHash, data leakage, Matrix Completion
Aula 4  DRIFT          → covariate/prior/concept, KS/PSI/χ²/JS, ADWIN, Evidently+MLflow
Aula 5  CONTRATO       → schema-as-code (Pandera, Pydantic, JSON Schema), Data Contracts, TFDV
Aula 6  ANOMALIAS      → Z-score, IQR, LOF, Isolation Forest, OCSVM, Autoencoders, PyOD
Aula 7  FERRAMENTAS    → GX, Deequ, Pandera, TFDV — comparação e integração
Aula 8  OPERAÇÃO       → CI/CD/CT, MLflow, Databricks, Unity Catalog, lineage, compliance
```

---

## Aula 1 — Fundamentos e Importância da Validação de Dados em ML

### Data-Centric AI vs. Model-Centric AI

| | Model-Centric | **Data-Centric** (Andrew Ng, 2021) |
|---|---|---|
| Pergunta | "qual é o melhor **modelo** para estes dados?" | "quais são os melhores **dados** para este problema?" |
| O que itera | arquitetura, loss, regularização | qualidade, rótulos, anotação |
| O que fica fixo | os dados | o modelo |

**O estudo de referência de Ng** (detecção de defeitos em manufatura, 500 imagens):

| Abordagem | Ganho médio de acurácia |
|---|---|
| Model-centric (testar arquiteturas) | **+0,4 p.p.** |
| Data-centric (corrigir rótulos, remover ambíguos, padronizar anotações) | **+3,2 p.p.** |

**Oito vezes mais ganho com o mesmo esforço.**

> **Ponto-chave:** em datasets de tamanho moderado (< 100.000 exemplos), a qualidade dos
> dados tem impacto sistematicamente maior na performance final que a escolha do algoritmo.
> *Modelos sofisticados treinados em dados ruins produzem resultados ruins — com alta confiança.*

### As seis dimensões de qualidade (Zhou et al., 2024)

| Dimensão | Definição | Risco em ML |
|---|---|---|
| **Completude** | todos os dados necessários estão presentes | o modelo aprende padrões a partir de informação parcial |
| **Acurácia** | os dados representam corretamente a realidade | o modelo **reproduz** erros de medição e rótulo, não os corrige |
| **Consistência** | mesmas regras e convenções em todo o dataset | unidades conflitantes e encodings divergentes confundem o aprendizado |
| **Atualidade** | quão recentes são os dados | discrepância perigosa entre treino e inferência |
| **Unicidade** | ausência de duplicatas | viés por sobre-representação + **data leakage** treino/teste |
| **Validade** | valores dentro dos domínios esperados | erros de coleta, corrupção ou anomalias genuínas |

> **Nenhuma dimensão pode ser avaliada isoladamente:** um dataset com alta completude mas
> baixa acurácia nos rótulos é **potencialmente mais prejudicial** do que um dataset
> incompleto com rótulos corretos.

### Data Cascades (Sambasivan et al., Google Research, CHI 2021)

Problemas de qualidade que **se amplificam ao longo do pipeline**, causando falhas sistêmicas
difíceis de diagnosticar e ainda mais difíceis de corrigir.

- **92% dos profissionais entrevistados** (53 profissionais de ML em 5 países) já
  experimentaram pelo menos um.
- Raramente visíveis no início — manifestam-se **semanas ou meses depois**, muitas vezes só
  quando o modelo entra em produção.
- Custo médio de identificação e correção: **dezenas de milhares de dólares** em retrabalho.

**Três fontes principais:** práticas inadequadas de coleta (instrumentos mal calibrados,
protocolos inconsistentes); incentivos organizacionais que priorizam **quantidade sobre
qualidade**; inexistência de processos formais de validação nos pipelines.

> Conclusão dos autores: qualidade de dados é um problema **sócio-técnico** — exige mudança
> de processo e cultura organizacional, não apenas ferramentas melhores.

### Custo econômico

Gartner (2020): má qualidade de dados custa às organizações, em média, **pelo menos
US$ 12,9 milhões por ano**. Em ML especificamente: custo contínuo de infraestrutura (modelos
retreinados com dados ruins exigem monitoramento mais frequente), **risco regulatório** em
saúde/crédito/contratação, e custo de reputação.

### Representatividade e viés algorítmico

Dimensão frequentemente negligenciada. Um dataset pode ter completude perfeita, zero
duplicatas e alta acurácia individual — e ainda assim ser **inadequado para treinar um modelo
justo** se não refletir a diversidade da população atendida.

**Gender Shades (Buolamwini & Gebru, 2018):**

| Grupo | Taxa de erro em reconhecimento facial comercial |
|---|---|
| Homens de pele clara | **0,8%** |
| Mulheres de pele escura | **34,7%** |

> A causa raiz **não era o algoritmo** — era o dataset, composto majoritariamente por rostos
> de homens de pele clara. *O modelo aprendeu perfeitamente o que os dados ensinaram.*

### Trustworthy AI (NIST AI RMF 1.0, 2023)

Um sistema confiável deve ser válido e verificável, confiável, seguro, explicável,
interpretável, respeitoso à privacidade, justo e gerenciável. A conexão com dados é direta:

- dados incompletos → **não verificável** (limites de operação desconhecidos)
- rótulos ruidosos → **não confiável** (performance real pior que a estimada)
- dados não-representativos → **não justo**, independentemente do algoritmo

### Casos emblemáticos

| Caso | O que aconteceu | Custo |
|---|---|---|
| **Unity Technologies (2022)** | dados incorretos no pipeline de ML de anúncios; o sistema aprendeu padrões equivocados e **prejudicava** as campanhas dos clientes | **~US$ 110 milhões** |
| **Amazon recrutamento (2014–2018)** | treinado com currículos de contratações passadas (maioria homens); aprendeu que termos associados ao gênero feminino eram preditivos de não-contratação | projeto cancelado, decisões reais já influenciadas |
| **IBM Watson Oncology** | treinado com **casos hipotéticos** gerados por oncologistas do MSK, não com casos reais; recomendou tratamentos perigosos em hospitais parceiros | falha de representatividade em sistema de alto risco |
| **Google Flu Trends (2012–13)** | errou por fator de 2× vs. CDC: temporalidade e consistência — mudanças no comportamento de busca e no próprio algoritmo do Google, sem retreino | referência clássica de drift não tratado |

### Quatro categorias de ferramentas (Zhou et al., 2024 — 17 ferramentas analisadas)

| Categoria | Quando atua no pipeline |
|---|---|
| **Profiling** (relatórios exploratórios automatizados) | fase exploratória |
| **Validação** (regras e restrições declarativas) | pipeline de ingestão |
| **Limpeza e anotação** | antes do treinamento |
| **Monitoramento** | contínuo em produção |

**Tendência:** LLMs para avaliação e melhoria automática de qualidade — detectar anomalias
semânticas, sugerir regras de validação e imputar valores com raciocínio contextual.

---

## Aula 2 — Dimensões de Qualidade de Dados

### Caso de abertura

Modelo de risco cardiovascular falha sistematicamente para mulheres acima de 60 anos.
Causa: 40% dos registros desse subgrupo tinham exames laboratoriais ausentes, e a "solução
técnica inofensiva" foi **preencher com a média global** — introduzindo viés silencioso e
devastador.

### Framework ICRA (Wang & Strong, 1996)

> *"Data quality is defined as **fitness for use** — a concept that cannot be assessed without
> considering the context in which data will be used."*

15 dimensões em 4 categorias (a partir de 179 atributos coletados junto a profissionais):

| Categoria | Dimensões | Relevância em ML |
|---|---|---|
| **Intrínseca** | acurácia, objetividade, credibilidade, reputação | qualidade factual dos dados de treino; risco de aprender padrões incorretos |
| **Contextual** | completude, temporalidade, volume, relevância, valor agregado | seleção de features; **data freshness** em produção |
| **Representacional** | interpretabilidade, consistência, concisão, integração | encoding, normalização, padronização de tipos |
| **Acessibilidade** | acessibilidade, segurança | **feature stores**, SLAs de pipelines, compliance LGPD/GDPR |

Exemplos concretos: `'30/02/1985'` é data inválida (acurácia); `'col_123'` vs.
`'idade_paciente_anos'` (interpretabilidade); `'2024-01-15'`, `'15/01/2024'` e
`'Jan 15, 2024'` no mesmo campo (consistência); CPF com e sem pontuação quebra joins
(facilidade de manipulação).

### Normas internacionais

- **ISO/IEC 25012:2008** (família SQuaRE) — 15 características em duas perspectivas:
  **inerentes ao dado** (acurácia, completude, consistência, credibilidade, atualidade) e
  **dependentes do sistema** (acessibilidade, conformidade, confidencialidade, eficiência,
  precisão, rastreabilidade, compreensibilidade, portabilidade, recuperabilidade,
  disponibilidade).
- **ISO/IEC 5259-4:2024** — framework específico para **qualidade de dados em IA**, com
  quatro particularidades não capturadas por normas gerais:
  1. **Representatividade** — adequação da distribuição de treino à população de interesse
  2. **Separação treino/teste** — dados de teste exigem qualidade especialmente alta
  3. **Linhagem de dados** — rastreabilidade de origem e transformações (auditoria/compliance)
  4. **Qualidade em aprendizado contínuo** (retreinamento)

> Estudos empíricos mostram ganhos de **33% a 88%** em classificação ao melhorar
> sistematicamente a qualidade dos dados (Zhou et al., 2024).

### Operacionalizando métricas

Transformar "este dataset tem problemas" em **"23,7% dos registros de colesterol estão
ausentes e 4,2% apresentam valores biologicamente impossíveis"**.

- **Completude** — medível por célula, coluna, linha e dataset; cada granularidade direciona
  uma estratégia diferente.
- **Acurácia** — a mais difícil de automatizar (requer domínio): combina regras de negócio
  (range/type checks, integridade referencial), comparação com **ground truth externo** e
  detecção estatística de anomalias.
- **Consistência** — três cenários: **intra-registro** (valores contraditórios no mesmo
  registro), **inter-registro** (violações de dependências funcionais) e **temporal**.
- **Temporalidade** — idade máxima dos registros, distribuição temporal treino vs. produção,
  taxa de atualização. **Primeiro sinal de alerta em sistemas sujeitos a data drift.**

### Mecanismos de dados faltantes (Rubin, 1976) — o coração da aula

Seja `Y` a matriz de dados (`Y_obs` observados, `Y_mis` ausentes) e `R` a matriz indicadora
(1 = observado, 0 = ausente). A pergunta é: **por que o dado está faltando?**

| Mecanismo | Formalização | Exemplo | Métodos válidos |
|---|---|---|---|
| **MCAR** (Missing Completely At Random) | `P(R \| Y_obs, Y_mis) = P(R)` | sensor que falha por interferência eletromagnética não relacionada aos valores | qualquer método; **complete case analysis é não-viesada**; imputação simples por média/mediana é defensável |
| **MAR** (Missing At Random) | `P(R \| Y_obs, Y_mis) = P(R \| Y_obs)` | pacientes idosos têm maior probabilidade de não completar o questionário — depende da **idade observada**, não da resposta | complete case é **viesada**; k-NN, MICE, imputação múltipla são não-viesados |
| **MNAR** (Missing Not At Random) | `P(R \| Y_obs, Y_mis) ≠ P(R \| Y_obs)` | pessoas com renda muito alta ou muito baixa não a reportam; sensor falha acima de um threshold | **qualquer imputação introduz viés** sem modelar explicitamente o mecanismo; modelos de seleção, pattern mixture, bayesiano |

**O perigo do MNAR em ML:** num dataset de crédito onde clientes de maior risco tendem a não
reportar a renda, um modelo com imputação simples aprenderá que **"renda não informada
indica baixo risco"** — o oposto da realidade.

> **Teste de Little (1988)** é o padrão para verificar MCAR empiricamente.
> Identificar MNAR **requer conhecimento de domínio** — testes estatísticos não distinguem
> MAR de MNAR apenas pelos dados.

### Métodos de imputação — do clássico ao deep learning (Fan, 2025)

| Família | Métodos | Características |
|---|---|---|
| **Univariada** | média (numérica simétrica), mediana (assimétrica/outliers), moda (categórica) | **distorce a distribuição, reduz artificialmente a variância, cria correlações espúrias** |
| **k-NN** | k registros mais similares no espaço observado | preserva correlações; válido sob MAR; **normalize antes** (escalas dominam a distância) |
| **MICE** (Multiple Imputation by Chained Equations) | múltiplos datasets completos amostrados da posterior, combinados pela regra de Rubin | **padrão-ouro clássico para MAR**; preserva a incerteza nos intervalos de confiança e p-valores |
| **Autoencoders / VAEs** | representação comprimida e reconstrução | VAEs acrescentam incerteza quantificada |
| **GANs — GAIN** (Yoon et al., 2018) | gerador produz valores plausíveis, discriminador tenta distingui-los | força imputações **estatisticamente indistinguíveis** dos reais |
| **Transformers tabulares** (SAINT) | cada coluna é um "token", atenção captura dependências | |
| **Modelos de difusão** | dados faltantes como deterioração; difusão reversa reconstrói | estado da arte |

> **Trade-off prático:** para datasets pequenos (< 10.000 registros), **MICE com Random Forest
> frequentemente supera deep learning**. A escolha deve considerar tamanho do dataset,
> mecanismo de ausência e recursos computacionais.

### Hands-on

`ydata-profiling` → `ProfileReport(df, explorative=True)` → `to_file()` gera relatório HTML
interativo com heatmap de valores faltantes, alertas automáticos (alta correlação, alta taxa
de nulos, constantes, outliers, distribuições enviesadas) e análise de duplicatas;
`get_description()` expõe tudo programaticamente para integrar a um pipeline automatizado.

### Fronteira

**LLMs para imputação tabular** (usando conhecimento de domínio e raciocínio contextual —
especialmente promissor para **MNAR**, cujo mecanismo pode ser descrito em linguagem natural)
e **imputação federada preservando privacidade** (Federated MICE, FedGAN) em saúde e finanças.

---

## Aula 3 — Detecção e Tratamento de Duplicatas e Valores Ausentes

### Ordem importa

> **Deduplicar ANTES de imputar** (imputar antes amplifica padrões espúrios e infla métricas)
> e **verificar leakage ANTES de qualquer split treino/teste.**

### Taxonomia de duplicatas (Elmagarmid et al., 2007 — *Entity Resolution / Record Linkage*)

| Tipo | Definição formal | Exemplo | Impacto em ML |
|---|---|---|---|
| **Exata** | `r_i = r_j` — todos os atributos idênticos | linha copiada duas vezes no CSV | data leakage direto se cruzar train/test |
| **Parcial** | subconjunto de atributos idênticos | mesma pessoa com campos diferentes preenchidos | viés de amostragem, over-representation |
| **Near-duplicate** | `sim(r_i, r_j) ≥ θ`, θ ∈ [0,8; 1,0) | artigos com parágrafos iguais; paciente com erro de digitação | inflação de métricas, **memorização** |

> Duplicatas **permanecem completamente invisíveis** até que seus efeitos sejam investigados
> — ao contrário de outliers (aparecem na EDA) e nulos (produzem erros visíveis).
> Common Crawl, C4 e The Pile têm **1% a 15% de near-duplicação** (Lee et al., 2022).

### Impacto por etapa do ciclo de vida

- **No treino:** exemplos duplicados recebem **peso implícito maior** na função de perda,
  distorcendo gradientes; em k-NN e kernel SVMs, inflam a densidade local criando viés de
  representação.
- **Cruzando treino↔teste (data leakage):** *"o modelo não está generalizando, está
  reconhecendo"* — as métricas deixam de ser preditores confiáveis da performance real.
- **Em LLMs:** Kandpal et al. (2022) mostraram que a probabilidade de **memorização é
  diretamente proporcional à frequência** de uma sequência no dataset — amplificada por
  near-duplicates.

### Tipos de data leakage e mitigação

| Tipo | Mecanismo | Detecção | Mitigação |
|---|---|---|---|
| Duplicata exata train↔test | registro idêntico em ambos | comparação de hashes entre splits | **deduplicar antes do split** |
| Near-duplicate train↔test | registro similar cruza os conjuntos | LSH / MinHash | **entity-level splitting** (`GroupShuffleSplit`) |
| Temporal (séries) | sobreposição de janelas | verificação de timestamps | **walk-forward validation** |
| Target leakage | feature calculada com informação do target | análise causal de features | feature engineering cauteloso |

**Albelali & Ahmed (2025)** — modelos LSTM avaliados com **k-fold aleatório** apresentaram
**RMSE até 40% melhor** que com walk-forward validation; janelas de entrada maiores agravam o
leakage. Afeta qualquer domínio com dependência: pacientes com múltiplas visitas, usuários com
múltiplas interações, moléculas similares.

### Detecção

**Exatas:** hash **SHA-256** sobre cada registro serializado — O(n), colisão praticamente
impossível. Muito mais rápido que comparação par-a-par.

**Near-duplicates — medidas de similaridade:**

| Medida | Fórmula | Tipo de dado | Propriedades |
|---|---|---|---|
| **Jaccard** | `J(A,B) = \|A∩B\| / \|A∪B\|` | conjuntos, textos (shingles) | simétrica; **base teórica do MinHash** |
| **Cosseno** | `(A·B) / (‖A‖·‖B‖)` | vetores numéricos, TF-IDF | invariante a magnitude; rápida |
| **Edit distance** | mín. de operações s→t | strings, IDs, nomes | simétrica, métrica |
| **Hamming** | `Σ[aᵢ≠bᵢ]` | strings de mesmo comprimento, bits | adequada a perceptual hashing |

**O problema da escala:** comparação par-a-par é **quadrática** — com n = 1 milhão,
~5×10¹¹ comparações. Inviável.

**Locality-Sensitive Hashing (Indyk & Motwani, 1998):** garante **matematicamente** que pares
similares têm alta probabilidade de cair no mesmo *bucket* e pares dissimilares, em buckets
diferentes. **MinHash** transforma documentos em assinaturas curtas que preservam a
similaridade de Jaccard; o LSH usa essas assinaturas para encontrar candidatos sem comparar
todos os pares.

**Clustering dos candidatos:** **Union-Find** (Disjoint Set Union) une clusters a cada par
identificado. Alternativas por densidade (**DBSCAN**, OPTICS) dispensam k e tratam ruído como
outlier. Para texto, **DBSCAN sobre embeddings de Sentence-BERT** detecta near-duplicates
**semânticos** (dois artigos que descrevem o mesmo fato com vocabulário diferente) —
impossível para Jaccard lexical (Phan et al., 2024).

### Matrix Completion Theory

Trata dados ausentes como o problema de preencher uma matriz incompleta, assumindo que a
matriz completa tem **rank baixo** (representável de forma compacta). Como minimizar o rank é
NP-difícil, usa-se a **norma nuclear** (soma dos valores singulares) como substituto convexo —
resolvido por **SVT** (Singular Value Thresholding) e **ALS** (Alternating Least Squares).

> O `IterativeImputer` do scikit-learn é **essencialmente ALS aplicado à imputação**: trata
> cada coluna com ausentes como função das outras e itera até convergir. Com forte correlação
> entre variáveis (colesterol, pressão, IMC), `KNNImputer` e `IterativeImputer` produzem
> imputações significativamente mais precisas que métodos univariados.

### GAIN em detalhe (Yoon et al., 2018)

Três componentes: o **Gerador** observa os dados existentes e o vetor de máscara para
preencher as lacunas; o **Discriminador** opera **valor por valor**, tentando discernir o que
foi observado do que foi imputado; o **Hint Vector** estabiliza o treinamento dando ao
discriminador informação parcial sobre a localização dos ausentes — forçando o gerador a
aprender a **verdadeira distribuição** para enganar um adversário cada vez mais informado.

⚠️ Modelos mal treinados podem introduzir **"alucinações estatísticas"** no dataset — validar
rigorosamente via RMSE e análise de distribuição.

### Hands-on

`missingno` (matrix, bar, matriz ordenada por uma coluna) para visualizar padrões de
missingness; comparação de `SimpleImputer` (média/mediana) × `KNNImputer` (k=3, **com
`StandardScaler` antes**) × `IterativeImputer` (MICE), medindo RMSE **apenas nas posições
mascaradas** — o `IterativeImputer` obteve o menor erro, confirmando a Matrix Completion Theory.

### Case: deduplicação em LLMs (Lee et al., 2022)

MinHash LSH aplicado a datasets massivos de pré-treinamento (T5). A deduplicação:
**reduz a memorização literal** (protege privacidade, previne plágio), **melhora a calibração**
(predições mais honestas) e **mantém o desempenho** — validando a técnica como etapa essencial
e economicamente viável.

### Tendências

**Deduplicação neural** (embeddings de foundation models em vez de shingles lexicais, para
detectar near-duplicates semânticos) e **Privacy-Preserving Record Linkage (PPRL)** — técnicas
como **Bloom Filter Encoding** permitem comparar registros de saúde entre hospitais para
identificar o mesmo paciente **sem trocar dados identificáveis** (Vatsalan et al., 2023).

---

## Aula 4 — Monitoramento de Data Drift e Mudanças de Distribuição

> Esta aula cobre o mesmo terreno do módulo de Data Drift — ver
> [`01-data-drift.md`](01-data-drift.md). Abaixo, o que é específico desta versão.

### Taxonomia (Moreno-Torres et al., 2012; Gama et al., 2014)

| Tipo | Formalização | Exemplo | Método de detecção |
|---|---|---|---|
| **Covariate Shift** (Data/Feature Drift) | `P_train(X) ≠ P_prod(X)`, mas `P(Y\|X)` igual | perfil demográfico muda após campanha de marketing | testes nas **features**: KS, PSI, JS, χ² — **sem precisar de labels** |
| **Prior Probability Shift** (Label Drift) | `P_train(Y) ≠ P_prod(Y)`, mas `P(X\|Y)` igual | taxa de fraude sobe de 0,1% para 1% | monitorar a distribuição do target — **requer labels de produção** |
| **Concept Drift** | `P(Y\|X)` muda | pandemia altera consumo; nova regulamentação muda critérios de crédito | monitorar **métricas do modelo** (acurácia, F1, AUC) no tempo |

> **O label delay** torna o monitoramento de prior shift **retrospectivo** (em crédito, a
> inadimplência só é conhecida meses depois), enquanto covariate shift é monitorável em
> tempo real. Daí a assimetria prática na escolha de métricas.

### Natural drift × Technical drift — a distinção que muda a resposta

| | **Natural drift** | **Technical drift** |
|---|---|---|
| Causas | sazonalidade e tendências, eventos macroeconômicos, mudanças regulatórias (LGPD/GDPR), evolução comportamental | alterações em pipelines de ETL, bugs em feature engineering, atualização de APIs de terceiros (escala de um score de bureau muda), problemas de encoding, mudanças de schema |
| **Mitigação** | **retreinar o modelo** | **corrigir o pipeline de dados** |
| Perigo | visível no negócio | **mais sutil — passa despercebido por longos períodos** |

### Comparação dos testes

| Teste | Tipo de feature | Vantagem | Limitação |
|---|---|---|---|
| **KS** | numérica contínua | não-paramétrico; sensível a forma **e** localização | **falsos positivos em amostras muito grandes** |
| **PSI** | numérica (discretizada) | interpretável; padrão da indústria financeira | dependente do número de bins (**use 10–20, cada um com ≥ 5% da referência**) |
| **χ²** | categórica | robusto para discretas, implementação simples | exige frequências esperadas ≥ 5 por categoria (agrupar raras ou usar Fisher) |
| **JS Divergence** | numérica e categórica | simétrica, finita, estável em todos os cenários | thresholds menos consolidados que PSI |

Faixas de PSI: `< 0,1` estável · `0,1–0,2` investigar e aumentar a frequência de monitoramento
· `≥ 0,2` **retreinar e investigar o pipeline de dados**.

Para KS, **complemente o p-value com a estatística D** — thresholds dependem do contexto
(D > 0,1 já preocupa em crédito; D > 0,2 em e-commerce).

JSD acima de **0,1** tipicamente indica drift significativo.

### Sliding windows e ADWIN

Referência fixa (treino ou período estável) comparada periodicamente com uma janela móvel.
**Trade-off:** janela pequena reage a flutuações pontuais (falsos positivos); janela grande
dilui mudanças locais e atrasa a detecção.

**ADWIN** (Bifet & Gavaldà, 2007) resolve ajustando o tamanho **automaticamente** — cresce
quando os dados são estáveis, encolhe ao detectar mudança, localizando-a com precisão.
Disponível na biblioteca **River** (`delta=0.002`, atributo `drift_detected`).

Na prática: job agendado (cron, Airflow, Databricks Job) compara a janela atual com a
referência, registra os resultados (MLflow, Grafana) e dispara alertas. **Monitorar a mesma
feature em múltiplas escalas** (hora, dia, semana, mês) reduz falsos positivos e negativos.

### Evidently + MLflow (o pipeline do hands-on)

**`DataDriftPreset`** escolhe o teste adequado automaticamente por feature:

| Caso | Teste escolhido |
|---|---|
| Numérica com > 1.000 observações | **KS** |
| Numérica com amostras menores | **Wasserstein** |
| Categórica com poucas categorias | **Chi-Square** |
| Alta cardinalidade | **JS Divergence** |

Evidently tem dois modos: **Reports** (análises descritivas com visualizações, para
stakeholders) e **Test Suites** (verificações passa/falha para **CI/CD**). Já cobre
monitoramento de sistemas baseados em LLMs (RAGs, chatbots).

**MLflow** registra cada execução de monitoramento como um *run* — `log_metric` de KS/PSI/JS
**por feature** formando séries temporais que revelam features em deterioração gradual e
correlacionam drift com performance; `search_runs` consulta o histórico auditável.

### Regulação e tendências

**SR 11-7 do Federal Reserve** exige que instituições financeiras monitorem continuamente a
estabilidade de modelos de risco — drift detection é **obrigação regulatória**, não apenas boa
prática. **Netflix** dispara retreinos automáticos ao ultrapassar thresholds em features de
comportamento. Tendência: **SHAP integrado ao monitoramento de drift**, identificando quais
features contribuem mais para a mudança — camada de **interpretabilidade** sobre o processo.

---

## Aula 5 — Validação de Schema e Contratos de Dados

### O desastre que a aula previne

Um campo é renomeado (`transaction_amount` → `txn_value`) sem aviso. O modelo passa a receber
`null` onde deveria haver valores reais. **As previsões colapsam silenciosamente**, alertas não
disparam, e fraudes passam despercebidas por horas.

> Objetivo: construir pipelines que **falham de forma ruidosa e controlada, nunca silenciosa e
> catastrófica.**

### Débito técnico de qualidade

Sem schema explícito: (1) um novo integrante precisa **ler o pipeline inteiro** para entender
o formato dos dados — não existe fonte de verdade; (2) quando um sistema upstream muda um tipo
de `string` para `int`, **não há mecanismo para detectar antes da falha em produção**.

### As três ferramentas Python

| Ferramenta | Valida | Quando usar |
|---|---|---|
| **Pandera** | DataFrames (pandas, Polars, Modin) | validação de DataFrames; **schema-as-code** versionável e testável em CI/CD |
| **Pydantic v2** | registros individuais | payloads de API de inferência, configs de experimento, entidades entre serviços |
| **JSON Schema** | documentos JSON | **contratos entre serviços** em linguagens diferentes (microsserviços, streaming) |

**Pandera** — classes `DataFrameModel` com type hints:
```python
class TransactionSchema(pa.DataFrameModel):
    transaction_id: Series[str]  = pa.Field(nullable=False, unique=True)
    amount:         Series[float] = pa.Field(gt=0, le=1_000_000)
    category:       Series[str]  = pa.Field(isin=["food","transport","retail","other"])
    is_fraud:       Series[bool] = pa.Field(nullable=True)
    class Config:
        strict = True
        coerce = True

@pa.check_types                       # valida entrada E saída da função
def preprocess(df: DataFrame[TransactionSchema]) -> DataFrame[TransactionSchema]: ...
```
`validate(lazy=True)` reporta **todas** as violações de uma vez em `SchemaErrors`.
Suporta **checks Hypothesis** (propriedades estatísticas) — ponte natural com monitoramento de
data drift.

**Pydantic v2** — core reescrito em Rust, **5× a 50× mais rápido**; espinha dorsal de FastAPI,
SQLModel e LangChain; +200 milhões de downloads/mês. O `@model_validator(mode="after")`
permite **regras cross-field** impossíveis de expressar campo a campo:
```python
@model_validator(mode="after")
def online_high_risk_requires_review(self):
    if self.is_online and self.high_risk_country and self.amount > 5000:
        raise ValueError("Requer revisão manual")
    return self
```

**JSON Schema** — padrão aberto, **neutro de linguagem**, com implementações em 30+ linguagens.
`$defs` + `$ref` permitem bibliotecas de tipos reutilizáveis; `allOf`/`anyOf`/`oneOf`
expressam polimorfismo. `additionalProperties: false` fecha o contrato.

### Data Contracts — o nível organizacional

> Um data contract **não é apenas uma definição técnica de campos e tipos: é um SLA entre
> equipes** que inclui qualidade mínima, frequência de atualização, proprietário e compromisso
> de compatibilidade retroativa.

A **Data Contract Specification** (YAML) codifica:
```yaml
dataContractSpecification: 0.9.3
id: urn:datacontract:finance:transactions:v1
info:    { title, version, owner, contact }
servers: { production: { type: kafka, topic: finance.transactions.v1 } }
terms:
  usage:        "Uso restrito a modelos de risco e detecção de fraude"
  limitations:  "Sem uso para marketing direto"
  noticePeriod: P3M          # 3 meses de aviso para breaking changes
models:
  Transaction:
    fields: { transaction_id: {...}, amount: {...}, category: {...}, timestamp: {...} }
quality:
  freshness:    "max 5 minutos de latência"
  completeness: ">99.9%"
  accuracy:     "Validado contra sistema transacional"
```

### Schema Evolution — compatibilidade (Avro / Confluent Schema Registry)

| Tipo | Regra central | Mudança permitida | Caso de uso ideal |
|---|---|---|---|
| **Backward** | novos schemas leem dados antigos | adicionar campo opcional | **consumidor atualiza primeiro** |
| **Forward** | schemas antigos leem dados novos | remover campo opcional | **produtor atualiza primeiro** |
| **Full** | ambas as direções | apenas campos opcionais | **deploys independentes** |
| **None** | sem garantias | qualquer mudança | ambiente de desenvolvimento |

Ferramentas: `fastavro` (serialização Avro), `confluent-kafka` (Schema Registry). Sem Kafka,
o Pandera exporta schemas em JSON versionáveis e validáveis em CI/CD.

### TFDV — training-serving skew

Computa estatísticas de forma **distribuída (Apache Beam)**, **infere o schema inicial** de
grandes datasets e detecta anomalias e **skew entre treino e serving** — o drift técnico.

```python
train_stats = tfdv.generate_statistics_from_dataframe(train_df)
schema = tfdv.infer_schema(statistics=train_stats)
tfdv.get_feature(schema, "amount").float_domain.min = 0.01          # refino manual
tfdv.get_feature(schema, "category").string_domain.value.append("crypto")
anomalies = tfdv.validate_statistics(statistics=serving_stats, schema=schema)
tfdv.display_anomalies(anomalies)
```

### Roteiro de adoção (dica prática da aula)

```
1. Pandera       → validação de DataFrames
2. Pydantic      → validação de entidades individuais
3. JSON Schema   → quando precisar de interoperabilidade entre linguagens
4. TFDV          → datasets de grande escala + monitorar training-serving skew
5. Data Contracts→ passo organizacional final: formalizar SLAs entre times
```

### Case: Uber

Trilhões de eventos/dia; precisava garantir que mudanças de schema não quebrassem **centenas
de consumidores downstream**. Solução: **Data Quality SLAs** + **Tiering de Dados (Tiers 0 a 2)**
classificando criticidade, com monitoramento e alerta automatizados.

---

## Aula 6 — Detecção de Outliers e Anomalias

### A pergunta certa

> A questão central **não é "existe um outlier?"**, mas **"o que este ponto representa no meu
> problema de negócio?"**

Um ponto extremo pode distorcer uma regressão, enviesar fronteiras de decisão e **contaminar
as estatísticas de normalização do pipeline**. Mas em fraude, monitoramento industrial e
cibersegurança, **os pontos anômalos carregam o maior valor informacional** — precisamos
detectá-los, não eliminá-los.

### Três conceitos que não são sinônimos

| Termo | Definição |
|---|---|
| **Outlier** (Hawkins, 1980) | observação que se desvia tanto das outras a ponto de **levantar suspeita de ter sido gerada por um mecanismo diferente** |
| **Anomalia** | padrão fora do comportamento normal que, em muitas aplicações, **é o que queremos detectar** (fraudes, falhas) |
| **Novelty** | observação nova que difere do treino mas **pode ser um padrão legítimo a incorporar** (um novo tipo de cliente) |

### Taxonomia de anomalias (Chandola, Banerjee & Kumar, 2009)

| Tipo | Definição | Exemplo | Algoritmos |
|---|---|---|---|
| **De ponto** | instância individual anômala em relação ao restante | salário de R$ 500.000/mês onde a média é R$ 8.000 | Z-score, IQR, Isolation Forest (**a maioria foi projetada para este tipo**) |
| **Contextual** | anômala **num contexto específico**, não globalmente | 35°C é normal em agosto em Recife e anômalo em julho em Curitiba; R$ 10.000 é normal para um cliente de alto valor e suspeito para quem nunca gastou mais de R$ 200 | exige **atributos contextuais** (tempo, localização, perfil) |
| **Coletiva** | um **conjunto** de instâncias relacionadas é anômalo, mesmo que cada uma não seja | transações de R$ 49,90 a cada 3 horas = *card testing* | LSTMs, autoencoders temporais |

> **Classificar o tipo é o passo mais crítico antes de escolher o algoritmo** — um método para
> anomalias de ponto aplicado a anomalias coletivas terá taxa de detecção muito baixa,
> independentemente da implementação.

### Métodos

**Z-score** — `Z = (x − μ)/σ`, outlier se `|Z| > 3` (regra dos três sigmas, ≈99,7%).
**Limitação crítica:** sensível aos próprios outliers — extremos inflam μ e σ e **mascaram
outros outliers**. Variante robusta com mediana e **MAD**: `M_i = 0,6745·(x_i − x̃)/MAD`.

**IQR (Tukey, 1977)** — `[Q1 − k·IQR, Q3 + k·IQR]`, com k = 1,5 (padrão) ou 3,0 (extremos).
Robusto a extremos por construção e não exige normalidade — a escolha para exploração inicial.
**Limitação: univariado** — aplicar dimensão a dimensão gera muitos falsos positivos.

**KNN** — distância ao k-ésimo vizinho como score. k pequeno = sensível a ruído local;
k grande = perde outliers em clusters moderados.

**LOF (Breunig et al., 2000)** — introduziu o conceito de **localidade**: compara a densidade
local de cada ponto com a de seus vizinhos, não com uma densidade global. `LOF ≈ 1` normal;
`LOF >> 1` outlier. **Grande vantagem: detecta outliers em clusters de densidades variadas** —
limitação crítica dos métodos estatísticos.

**Isolation Forest (Liu, Ting & Zhou, 2008)** — **isola** os outliers em vez de modelar o
normal: outliers precisam de poucos cortes aleatórios para serem separados. O *path length* é
inversamente proporcional à anormalidade; o score é a média normalizada nas árvores, mapeada
para [0,1]. Eficiente, escalável, robusto à dimensionalidade e a features não normalizadas.
**Limitações:** dificuldade com anomalias coletivas e dependência do hiperparâmetro de
contaminação.
⚠️ *O scikit-learn pode inverter a escala (anomalias com score negativo) — verifique a
convenção antes de definir limiares.*

**One-Class SVM (Schölkopf et al., 2001)** — encontra uma hipersuperfície (no espaço do kernel)
que envolve a maioria dos pontos de treino com máxima margem em relação à origem. `nu ∈ (0,1]`
controla a fração tratada como outlier; kernel RBF com `gamma` controlando a largura.
Eficaz em alta dimensão **com treino limpo**; custoso em datasets grandes — alternativa:
`SGDOneClassSVM`.

**Autoencoders** — a rede reconstrói bem os padrões normais vistos no treino e **erra nos
incomuns** que o espaço latente não representa. O **erro de reconstrução é o anomaly score**.
Threshold: sem rótulos, um percentil alto (90/95/99%) do erro no treino; com rótulos parciais,
otimizar por curva ROC ou Precision-Recall.

### Comparação

| Método | Interpretabilidade | Escalabilidade | Melhor caso de uso | Limitação principal |
|---|---|---|---|---|
| Z-score / IQR | **Alta** | Alta | análise exploratória, dados normais univariados | assume normalidade; sensível a extremos (Z-score) |
| LOF | Média | Média | clusters de densidades diferentes | lento em grandes datasets; sensível a k |
| Isolation Forest | Média | **Alta** | **tabulares em larga escala; uso geral** | dificuldade com anomalias em clusters; mascaramento |
| One-Class SVM | Baixa | Baixa | alta dimensionalidade, treino limpo | lento; sensível a normalização e hiperparâmetros |
| Autoencoder | Baixa | Alta (GPU) | dados complexos de alta dimensionalidade | requer muitos dados; threshold subjetivo; caixa-preta |

> **Não existe algoritmo universalmente superior** (Boukerche, Zheng & Alfandi, 2020) — a
> performance relativa varia substancialmente com as características dos dados. **Teste
> múltiplas abordagens.**

**PyOD** (Zhao et al., 2019, JMLR): **50+ algoritmos** numa API consistente compatível com
scikit-learn (`fit()`, `predict()`, `decision_function()`, `predict_proba()`) — permite
comparações justas e pipelines modulares.

### Remover, transformar ou manter? — o framework de decisão

| Decisão | Quando | Como |
|---|---|---|
| **Remover** | evidência sólida de **erro**: −999 °C é código de erro; 150 anos num cadastro é digitação. Ou quando o outlier está **fora do escopo de uso** do modelo (mansões > R$ 50 mi num modelo de imóveis comuns) | documentar a restrição |
| **Transformar** | o outlier contém **informação legítima**, mas sua magnitude distorce o aprendizado | log(x)/log(1+x) para skew positivo, raiz quadrada, Box-Cox, **winsorizing** (clipping nos percentis p5/p95) |
| **Manter** | **o outlier é o fenômeno de interesse** — em fraude, as fraudes *são* os outliers e removê-las destrói o objetivo. Também quando há incerteza e o custo de descartar uma observação legítima supera o de manter um possível erro | treinar com e sem, comparar métricas em teste independente, **documentar a decisão** |

### Pipeline em tempo real

Requisitos além do algoritmo: modelos persistidos e recarregados eficientemente; pré-processamento
com os **mesmos parâmetros do treino**; e tratamento de data drift. ⚠️ **Um detector treinado há
meses pode estar degradado se a distribuição dos dados normais mudou** — implemente retreino
periódico ou acionado por drift detection.

> A taxa de contaminação é uma **hipótese de sensibilidade** que, em produção, deve ser
> estimada e **monitorada ativamente**.

### Cases

**PayPal / Stripe** — milhões de transações por hora combinando supervisionado, não
supervisionado e semi-supervisionado. Desafio central: **equilíbrio entre taxa de detecção e
falsos positivos** — cada transação legítima bloqueada é impacto negativo na experiência.
**Siemens Senseye** — manutenção preditiva sobre sensores de vibração e temperatura; alerta
horas ou dias antes da falha catastrófica, permitindo intervenção planejada a uma fração do
custo emergencial. **DARPA Transparent Computing** — datasets de referência com rastros de
APTs; autoencoders LSTM e One-Class SVM modelam tráfego normal de rede.

---

## Aula 7 — Frameworks de Qualidade de Dados

### Validação declarativa vs. imperativa

| | **Imperativa** | **Declarativa** |
|---|---|---|
| Como | código procedural ("se nulo, lançar exceção") | declara-se **o quê**; o framework executa **o como** |
| Prós | flexível | regras são **artefatos de primeira classe**, versionáveis, reutilizáveis, legíveis por não-técnicos, documentáveis automaticamente |
| Contras | mistura validação e lógica de negócio; difícil documentar | menos flexível para casos exóticos |

> Schelter et al. (2018): *"a qualidade de dados deve ser declarada como **constraints** sobre
> os dados, similar ao que constraints de banco de dados fazem para schemas relacionais."*

Na prática, os frameworks modernos são **híbridos**: DSL declarativa para os casos comuns +
extensão imperativa em Python.

**Shift Left:** validar o mais cedo possível no pipeline — **idealmente na fonte** — para
evitar que problemas se propaguem e amplifiquem. Os frameworks são a implementação técnica
desse princípio.

### Os quatro frameworks

**Great Expectations (GX)** — criado em 2017 por Abe Gong e James Campbell; v1.0 em 2024
reformulou a API. Quatro conceitos: **Data Source** (conexão) → **Asset** (dataset) →
**Batch** (fatia, ex.: os dados de um dia); **Expectations** (regras em linguagem quase-inglesa);
**Expectation Suite** (coleção); **Validation Definition** (executa).

Categorias de expectation:
- **Column Map** — verificam cada linha (`expect_column_values_to_match_regex`)
- **Column Aggregate** — estatísticas da coluna (`expect_column_mean_to_be_between`)
- **Table** — a tabela como um todo (`expect_table_row_count_to_be_between`)
- **Voltadas a ML** — `expect_column_chisquare_test_p_value_to_be_greater_than` verifica se a
  distribuição de uma categórica **não mudou significativamente** (detector de drift embutido!)

**Diferencial: Data Docs** — documentação interativa em HTML gerada automaticamente com suites
e resultados. *A documentação vira subproduto natural da validação*, publicável em S3/GCS/Azure
Blob e compartilhável com stakeholders não técnicos.

**Amazon Deequ** (Schelter et al., 2018) — sobre Apache Spark, valida **bilhões de linhas**.
Diferença crucial: enquanto o GX faz **uma passagem por constraint**, o Deequ computa **todas
as métricas em uma única passagem**. Três abstrações: **Analyzers** (calculam métricas),
**Constraints** (verificam critérios), **ConstraintVerification** (orquestra).

```python
check = Check(spark, CheckLevel.Warning, 'Validação Imóveis') \
    .isComplete('preco').isPositive('preco').isComplete('cidade') \
    .hasMin('preco', lambda x: x >= 50000) \
    .hasMean('area_m2', lambda x: 50 <= x <= 500).isUnique('id')
VerificationSuite(spark).onData(df).addCheck(check).run()
```

**Diferencial: `ConstraintSuggestionRunner`** — analisa o dataset e **sugere constraints
automaticamente** com nível de confiança, para o engenheiro revisar. Útil para datasets novos
sem conhecimento prévio profundo.

**Pandera** — criado por Niels Bantilan, mantido pela Union.ai. Baseado em **anotações de tipo**
— natural para equipes que usam mypy/pyright. `DataFrameSchema` (declarativo) e
`DataFrameModel` (orientado a objetos, à la Pydantic). `lazy=True` coleta todos os erros.
**Único da lista com suporte nativo a Polars.** Suporta checks customizados (element-wise e
agregados) e **Hypothesis** (teste t de Student entre colunas ou contra o histórico — detector
de drift ou verificação de regra de negócio).

**TFDV** (Caveness et al., 2020; parte do TFX) — único **orientado a estatística**, não a
constraints. Resolve três problemas: entender as características estatísticas, detectar
anomalias contra um schema/histórico, e **identificar skew e drift entre treino e produção**.
**Drift comparators** com limites de divergência aceitáveis (**L-infinito, Jensen-Shannon**).
Visualização interativa revela problemas difíceis de capturar com regras estáticas.

### Tabela comparativa

| Critério | **Great Expectations** | **Amazon Deequ** | **Pandera** | **TFDV** |
|---|---|---|---|---|
| Backend | Pandas, Spark, SQL, Cloud | **Apache Spark** | Pandas, **Polars**, Modin | TensorFlow/TFX, Pandas |
| Escalabilidade | Média-Alta (com Spark) | **Muito Alta** (bilhões de linhas) | Média (limitada pela memória) | Alta (TFRecords/Beam) |
| Curva de aprendizado | Baixa (v1.x) / Alta (v0.x) | Alta (requer Spark) | **Baixa** (pythônico) | Média-Alta |
| Análise estatística | Boa | Excelente | Boa | **Excelente (foco em ML)** |
| Documentação automática | **Excelente (Data Docs)** | Limitada | Boa (integra com mypy) | Boa (visualização) |
| **Melhor para** | pipelines Python/SQL com múltiplas fontes | big data com Spark/AWS | DataFrames Python com **type safety** | ecossistema TF/TFX, **análise de skew** |

### Integração com pipelines

Validar em **quatro níveis**: ingestão (raw data) → feature engineering (após transformação) →
treinamento → **serving (em tempo real**, garantindo que as entradas estão no domínio de treino).

**Prática `fail fast`:** exceção imediata nas validações **críticas** (impedindo avanço dos
dados inválidos) e registro das **não-críticas** (warnings) num sistema de observabilidade
(MLflow) para análise posterior.

> **Dica:** defina claramente quais violações são **blockers** e quais são **warnings**.
> Nulos em campos obrigatórios → blocker. "Distribuição mudou 5%" → warning que notifica
> para revisão humana.

Orquestração: Airflow (com **`GreatExpectationsOperator`** do Astronomer Cosmos, que executa
suites como etapas do DAG e registra resultados como artefatos), Lakeflow Jobs (Databricks)
ou Dagster.

### Os fundamentos de engenharia de software por trás

| Princípio | Origem | Como se aplica |
|---|---|---|
| **Design by Contract (DbC)** | Bertrand Meyer, Eiffel, anos 1980 | pré-condições, pós-condições e invariantes → **o produtor garante regras (pré-condição), o consumidor pode assumi-las (pós-condição)** |
| **Sketching Algorithms** | Cormode & Hadjieleftheriou (2010) | o que torna o Deequ escalável: **HyperLogLog** (cardinalidades), **Count-Min Sketch** (frequências), **t-Digest** (percentis) — uma passagem, erro *bounded* |
| **Integration Patterns** | Hohpe & Woolf (2003) | **Message Validator** = validação como etapa do pipeline; **Dead Letter Channel** = redirecionar dados inválidos para uma **quarentena** em vez de descartá-los |

### Cases e tendências

**Amazon** usa o Deequ internamente em data lakes com **mais de 60 petabytes**.
**Great Expectations**: milhões de downloads semanais no PyPI, 11k+ estrelas no GitHub;
**Trivago** o adotou em ambientes multi-cloud. **Airbnb** documentou o sistema **Midas** e um
**Data Quality Score calculado diariamente** — qualidade em escala exige tanto ferramentas
robustas quanto **processos culturais consolidados**.

Três tendências 2024–2025: (i) **Data Contracts** como padrão de governança (Soda, dbt);
(ii) **observabilidade de dados estendida a IA generativa** (Monte Carlo monitorando dados não
estruturados em fluxos de RAG e fine-tuning); (iii) **curadoria para LLMs** — **Argilla**
(anotação e feedback humano, ecossistema Hugging Face) complementando os frameworks tradicionais.

---

## Aula 8 — Validação Contínua com MLOps, MLflow, AWS e Azure

### Por que ML exige uma extensão do DevOps

> Modelos de ML não são apenas código — são a interseção de **três artefatos que evoluem
> independentemente: código, dados e hiperparâmetros**. Uma mudança em qualquer um deles pode
> alterar o comportamento em produção **mesmo com todos os testes de software passando.**

**Três níveis de maturidade MLOps** (Kreuzberger et al., 2022):

| Nível | O que existe |
|---|---|
| **0** | processos manuais, modelos estáticos |
| **1** | pipelines de treinamento automatizados, retreinamento contínuo |
| **2** | **CI/CD totalmente automatizado para ML** |

> É na transição do **nível 1 para o 2** que a validação contínua se torna essencial.

**Analogia da manufatura:** o código é a máquina, os dados a matéria-prima, o modelo o produto
— e **uma fábrica moderna não testa só o produto: monitora a matéria-prima e a máquina em cada
etapa.**

### CI, CD e CT

| | **CI** | **CD** | **CT** (exclusivo de ML) |
|---|---|---|---|
| Gatilho | novo commit de código **ou dados** | validação bem-sucedida no CI | **drift detectado** ou novos dados |
| Objetivo | validar integridade de dados e código | automatizar deploy de modelos | manter o modelo atualizado |
| Ferramentas | GitHub, GitLab | MLflow, Azure DevOps | Lakeflow Jobs, DABs, Kubeflow |
| Frequência | a cada push/PR | a cada merge aprovado | contínua, baseada em drift |

**CI para dados** vai além de testes unitários: a cada chegada de dados (batch ou streaming),
validar schema, **distribuições estatísticas** ("os dados de hoje são semelhantes aos de
ontem?"), completude e consistência das transformações.

> Chen et al. (2021): **a falta de CT é uma das principais causas de degradação silenciosa** —
> o modelo continua funcionando tecnicamente, mas suas previsões ficam progressivamente menos
> precisas à medida que o mundo evolui e os dados de treino envelhecem.

### MLflow para validação contínua

Quatro componentes: **Tracking** (experimentos), **Projects** (empacotamento reproduzível),
**Models** (formato agnóstico a frameworks), **Model Registry** (governança).
Para validação contínua, os pilares são **Tracking** e **Registry**.

**Hierarquia:** `Tracking Server → Experiments → Runs → (Params, Metrics, Artifacts)`

**Data logging (subestimado):** desde o MLflow 2.4, **`mlflow.log_input()`** associa um dataset
a uma run, registrando nome, **hash SHA256** (para verificar se o mesmo dataset foi usado em
runs diferentes), fonte e contexto de uso — **o primeiro passo para um pipeline de validação
contínua rastreável**.

**`mlflow.autolog()`** captura automaticamente parâmetros, métricas e artefatos de **15+
frameworks** (scikit-learn, XGBoost, LightGBM, TensorFlow, PyTorch) e combina com logging
manual de métricas de validação do domínio.

**Padrão-chave — artifact-driven pipeline gates:** usar artefatos como **"contrato de qualidade"
entre etapas**. A ingestão registra o relatório de validação do Great Expectations, e a etapa
seguinte **só prossegue se o artefato existir e `validation_passed` for `True`** — impedindo que
dados inválidos avancem e criando uma **cadeia auditável**.

**Model Registry:** estágios clássicos (Staging/Production/Archived) ou, a partir do MLflow 2.9,
**aliases customizáveis** — `champion`, `challenger`, `baseline` — permitindo fluxos de promoção
mais flexíveis.

### Databricks + Unity Catalog

**Unity Catalog** (2022) — metastore centralizado para tabelas/views SQL, volumes (dados não
estruturados), modelos MLflow, features do Feature Store e arquivos em storage cloud, com
**RBAC granular, auditoria completa e lineage automático**.

Dois recursos críticos para validação contínua:
1. **Models in Unity Catalog** — histórico de versões, runs e datasets de treinamento;
   **cadeia de rastreabilidade fundamental para auditorias de LGPD, GDPR e EU AI Act**
2. **Data Quality Constraints** impostas diretamente nas tabelas Delta Lake

### Data Observability — os 5 pilares (Barr Moses, Monte Carlo)

| Pilar | Pergunta |
|---|---|
| **Freshness** | os dados estão atualizados? |
| **Distribution** | as distribuições estão dentro do esperado? |
| **Volume** | o volume está dentro do esperado? |
| **Schema** | a estrutura não mudou inesperadamente? |
| **Lineage** | conseguimos rastrear origem e caminho? |

No Databricks: **Delta Lake** (transações ACID; o transaction log permite auditoria e **rollback
para qualquer versão anterior — time travel**) + **Lakeflow** (orquestrador nativo) + integração
com Great Expectations.

### Lineage tracking

Documenta automaticamente a origem de um dado e seu caminho da fonte ao destino, respondendo:
**"qual modelo foi treinado com esse dataset?"** e **"se este dado estiver errado, quais
modelos em produção foram impactados?"**

O Unity Catalog captura em três níveis — **column-level**, **table-level** (grafo de
dependências entre tabelas, views e modelos) e **model lineage** — **a partir das queries
executadas, sem anotações manuais no código**.

### Alertas

> Um pipeline de validação contínua sem alertas é como **um alarme de incêndio sem sirene**.

O alerta deve ser **proativo** (disparar antes de chegar ao usuário), **informativo** (contexto
para diagnóstico rápido) e **acionável** (direcionar a pessoa certa à ação certa). *Um alerta
que só diz "algo deu errado" é quase tão inútil quanto nenhum.*

Padrão recomendado: **logar as métricas de qualidade no MLflow e acionar as notificações pelo
Lakeflow** (e-mail, webhooks → PagerDuty, Slack, Teams).

### Databricks Asset Bundles (DABs) — IaC para MLOps

Define, versiona e implanta recursos do workspace como **YAML declarativo versionado no Git**,
em vez de clicar na interface. Um único repositório contém o código dos modelos **e** a
definição de jobs, schedules, permissões e artefatos MLflow.

`databricks.yml` define nome, variáveis, arquivos de recursos e **targets** (ambientes) — cada
target corresponde a um workspace e pode sobrescrever variáveis, permitindo **o mesmo código
em dev, staging e prod sem duplicação** (`${var.catalog_name}`, `${bundle.target}`).

```bash
databricks bundle validate                       # sintaxe e referências, sem deploy (ideal em CI)
databricks bundle deploy --target dev            # cria/atualiza recursos
databricks bundle run model_training_job --target dev
databricks bundle destroy --target dev
```
No modo *development*, os jobs recebem prefixo do usuário e **os schedules ficam pausados**.

### O pipeline completo do hands-on (mlops-stacks)

```
DataSetup.py              → 3 tabelas Delta no Unity Catalog (train, val, scoring)
EDA.py                    → nulos, duplicatas, estatísticas, correlações
Train.py                  → LightGBM + mlflow.lightgbm.autolog() + registro no UC Model Registry
ModelValidation.py        → mlflow.evaluate() contra thresholds
                            (RMSE ≤ 4,00 · MAE ≤ 3,00 · R² ≥ 0,85 · MAPE ≤ 25%)
                            → se passa, alias 'challenger'
ModelDeployment.py        → promove 'challenger' → 'champion'
batch inference           → mlflow.pyfunc.spark_udf() → tabela Delta com versão e timestamp
MonitoredMetricViolationCheck.py
                          → consulta _profile_metrics (Lakehouse Monitoring);
                            se RMSE excede o threshold em ≥ 2 das últimas 5 janelas,
                            dispara novo ciclo de treinamento  ← CONTINUOUS TRAINING
```

> Em um pipeline real, a etapa de validação incluiria também **constraints de schema, data
> quality e data drift** (Great Expectations), complementadas pelas expectations nativas dos
> pipelines Databricks (ações `warn`, `drop` ou `fail`).

⚠️ O repositório da aula usa tokens PAT por facilidade didática — **em produção, prefira
Service Principals com escopo reduzido.**

### Compliance e governança

GDPR, LGPD e o **EU AI Act** (em vigor desde 2024) impõem obrigações sobre como sistemas
automatizados de decisão devem ser **documentados, auditados e explicados aos afetados**.

- Unity Catalog → **audit log** detalhado de acessos e transformações
- Model Registry → registra **quem promoveu cada versão, quando e com qual justificativa**
- MLflow no Databricks suporta **HIPAA, PCI-DSS e FedRAMP Moderate** (junho/2025)

**Model Card** (Mitchell et al., 2019) — documento estruturado descrevendo dados de treinamento,
**métricas por subgrupo**, limitações e considerações éticas; registrável no Model Registry.
Integrá-lo ao CI/CD garante que **cada versão em produção tenha documentação de compliance
atualizada automaticamente**.

### Pipeline CI/CD completo — "pipeline as code"

```
PR com mudanças no código → GitHub Actions: testes unitários + treino em ambiente de testes
       ↓ aprovado e mergeado em main
GitHub Actions → pipeline completo no Databricks:
       treinamento → registro no MLflow → promoção automática ao alias "challenger"
```

> **O pipeline de CI/CD é código de primeira classe**, versionado junto com o modelo e os
> testes — qualquer mudança é revisada em PR, testada e documentada no histórico do Git.

O repositório `databricks/mlops-stacks` já inclui workflows que executam `bundle validate` +
testes a cada PR e `bundle deploy` ao mergear em `main` (staging) e `release` (prod).

### Mercado

Mercado global de MLOps: **US$ 2,98 bi em 2025 → projeção de US$ 89,91 bi até 2034**
(Fortune Business Insights).

**Nubank** — ML para crédito e personalização para 100M+ clientes em 3 países sobre Databricks;
a capacidade de rastrear lineage (**saber exatamente com quais dados cada modelo foi treinado**)
é fundamental para atender às exigências do **Banco Central** sobre explicabilidade de decisões
de crédito automatizadas.

**Airbnb — Zipline** (Zanoyan & Shapiro, 2019): plataforma declarativa de feature engineering
como **fonte única de verdade**; garante consistência treino/inferência via **computações
point-in-time corretas**, evitando data leakage. Reduziu de **meses para dias** o tempo de criar
novas features, incorporando monitoramento automático de qualidade às próprias definições.

**MLflow 3.0 (2025)** trouxe observabilidade nativa para IA generativa — **MLflow Tracing**
instrumenta aplicações LangChain, LlamaIndex e APIs de LLM com observabilidade comparável à de
modelos supervisionados. **LLMOps está convergindo rapidamente com o MLOps clássico.**

---

## Referências recorrentes do módulo

- **Zhou, Y. et al. (2024)** — *A Survey on Data Quality Dimensions and Tools for Machine
  Learning*. IEEE AITest. arXiv:2406.19614.
- **Sambasivan, N. et al. (2021)** — *"Everyone wants to do the model work, not the data work":
  Data Cascades in High-Stakes AI*. ACM CHI.
- **Wang, R. Y.; Strong, D. M. (1996)** — *Beyond Accuracy: What Data Quality Means to Data
  Consumers*. JMIS 12(4).
- **Rubin, D. B. (1976)** — *Inference and Missing Data*. Biometrika 63(3).
- **Van Buuren, S. (2018)** — *Flexible Imputation of Missing Data*, 2ª ed. CRC Press.
- **Fan, J. (2025)** — *An Interdisciplinary and Cross-Task Review on Missing Data Imputation*.
  arXiv:2511.01196.
- **Buolamwini, J.; Gebru, T. (2018)** — *Gender Shades*. FAT* / PMLR 81.
- **Schelter, S. et al. (2018)** — *Automating Large-Scale Data Quality Verification* (Deequ).
- **Caveness, E. et al. (2020)** — *TensorFlow Data Validation*. ACM SIGMOD.
- **Chandola, V.; Banerjee, A.; Kumar, V. (2009)** — *Anomaly Detection: A Survey*. ACM CSUR 41(3).
- **Breunig, M. M. et al. (2000)** — *LOF: Identifying Density-Based Local Outliers*. SIGMOD.
- **Liu, F. T.; Ting, K. M.; Zhou, Z.-H. (2008)** — *Isolation Forest*. IEEE ICDM.
- **Zhao, Y. et al. (2019)** — *PyOD: A Python Toolbox for Scalable Outlier Detection*. JMLR 20.
- **Elmagarmid, A. K. et al. (2007)** — *Duplicate Record Detection: A Survey*. IEEE TKDE 19(1).
- **Indyk, P.; Motwani, R. (1998)** — *Approximate Nearest Neighbors* (LSH). ACM STOC.
- **Lee, K. et al. (2022)** — *Deduplicating Training Data Makes Language Models Better*. ACL.
- **Kreuzberger, D.; Kühl, N.; Hirschl, S. (2022/2023)** — *MLOps: Overview, Definition, and
  Architecture*. IEEE Access.
- **Mitchell, M. et al. (2019)** — *Model Cards for Model Reporting*. ACM FAccT.
- **NIST (2023)** — *AI Risk Management Framework (AI RMF 1.0)*. NIST AI 100-1.
- **ISO/IEC 25012:2008** e **ISO/IEC 5259-4:2024**.
