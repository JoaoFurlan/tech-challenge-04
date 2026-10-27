# Data Drift — resumo do módulo (Aulas 1 a 8)

> Fontes: `POSTECH_DATA_DRIFT_AULA_1..8.pdf`
>
> ⚠️ Nota: os PDFs 6 e 7 vêm com os títulos trocados em relação à ementa. O arquivo
> `AULA_6` traz no cabeçalho "Detecção de Concept Drift vs. Covariate/Label Drift" mas o
> conteúdo é **Monitoramento Contínuo e Data SLOs**; o `AULA_7` traz "Monitoramento
> Contínuo e Data SLOs" mas o conteúdo é **drift em escala / Big Data / streaming**.
> Abaixo seguem os conteúdos reais.

---

## 0. O mapa mental do módulo

```
            MUNDO MUDA
                │
   ┌────────────┼────────────┐
DETECTAR    DIAGNOSTICAR   MITIGAR
   │            │            │
KS, χ², PSI   que tipo?   retreino, online learning,
KL, JS,       covariate/  ensembles adaptativos,
Wasserstein,  label/      ajuste de limiar, outliers,
MMD, LSDD,    concept     regras de negócio
ADWIN/CUSUM
                │
         OPERACIONALIZAR
   SLIs/SLOs • streaming • dashboards • alertas • retreino automático
   Evidently • NannyML • Great Expectations • Alibi-Detect • SageMaker/Azure
```

---

## Aula 1 — Introdução ao Drift em Dados e Modelos

### Definições formais

| Tipo | O que muda | Formalização | Também chamado |
|---|---|---|---|
| **Data drift / feature drift** | distribuição das entradas | `P_treino(X) ≠ P_prod(X)`, com `P(Y\|X)` constante | *covariate shift* |
| **Concept drift** | a relação entrada→saída | `P_treino(Y\|X) ≠ P_prod(Y\|X)` | *real concept drift* |
| **Label drift** | distribuição do alvo | `P_treino(Y) ≠ P_prod(Y)` | *prior probability shift* |

Visão guarda-chuva: `P_treino(X, Y) ≠ P_producao(X, Y)` — o modelo "envelheceu"
(*AI aging*). Na prática o termo "data drift" é usado genericamente, mas **diferenciar os
tipos é o que define a estratégia de resposta**.

Dado citado: pesquisa de 2023 (MIT/Harvard; Nature Communications) apontou que **91% dos
modelos de ML em produção sofrem degradação significativa** ao longo do tempo.

### Por que os dados mudam (ambientes não-estacionários)

A hipótese IID quebra. Causas comuns:

1. **Comportamento humano / mercado** — modas, gírias, mudanças culturais, novos hábitos
   de consumo (fonte clássica de concept drift).
2. **Fatores externos** — crises econômicas, inflação, novas leis, tecnologias novas,
   "cisnes negros" (COVID-19 quebrou padrões históricos de vendas e tráfego).
3. **Evolução da coleta de dados** — novos sensores, mudança de definição de métricas,
   bugs silenciosos no pipeline, unidades diferentes, categorias novas. Gera **drift
   artificial**: o conceito X→Y não mudou, mas a entrada do modelo mudou.
4. **Sazonalidade e ciclos** — Natal vs. janeiro, dia útil vs. fim de semana. Drift
   *previsível* e recorrente, que ainda assim degrada modelos não adaptativos.

### Padrões temporais de drift

- **Abrupto** — salto instantâneo (lockdown de março/2020). Fácil de detectar, impacto máximo.
- **Gradual / incremental** — mudança lenta e acumulativa. **Perigoso porque passa
  despercebido** sem monitoramento constante.
- **Sazonal / recorrente** — padrões que vão e voltam. Resposta: dar **consciência de
  contexto** ao modelo (feature de dia da semana/hora, ou modelos por contexto).

### Consequências

- Queda de acurácia / aumento de erro (dívida técnica oculta — Sculley et al., 2015).
- Decisões de negócio equivocadas (crédito, manutenção preditiva).
- Experiência do usuário degradada e perda de confiança no produto.
- **Riscos de compliance e éticos**: modelo pode se tornar enviesado após o drift;
  atacantes podem explorar "pontos cegos" de um modelo que sabidamente não é atualizado.

### Ciclo de resposta

**Monitorar → Detectar → Adaptar**, com quatro alavancas de adaptação:
retreino periódico programado; aprendizado online/incremental (`partial_fit`, River,
scikit-multiflow); ensembles adaptativos (Adaptive Random Forest, Dynamic Weighted
Majority); intervenção humana com novas features/fontes de dados.

> "Nem todo drift deve ser tratado com retreino imediato: parte da arte de MLOps é
> determinar quanta mudança justifica uma intervenção." Retreinar demais introduz ruído;
> de menos deixa o modelo obsoleto. Daí a categorização **leve / moderado / severo**.

### Código-chave

```python
from scipy.stats import ks_2samp
stat, p_value = ks_2samp(X_treino, X_novo)
if p_value < 0.05:
    print("Drift: distribuição de X mudou significativamente!")
```

```python
from sklearn.linear_model import SGDClassifier
modelo = SGDClassifier(loss="log_loss")
for X_batch, y_batch in fluxo_dados():
    modelo.partial_fit(X_batch, y_batch, classes=[0, 1])
```

---

## Aula 2 — Técnicas de Detecção de Drift (Estatística)

**Caso-guia:** fintech de crédito em que a inadimplência sobe sem mudança de política —
o perfil etário dos tomadores se deslocou (covariate shift).

### Duas famílias complementares

| | Métricas de divergência | Testes de hipótese |
|---|---|---|
| Respondem | **quanto** mudou e **onde** (qual variável) | **se** a mudança é estatisticamente significativa |
| Exemplos | KL, JS, PSI, Wasserstein | KS, Qui-quadrado (χ²) |
| Saída | score de magnitude | p-valor |

> Estratégia recomendada: *screening* contínuo com métricas de divergência (para priorizar
> alertas) + confirmação com testes formais quando um possível drift aparece.

### Métricas

- **KL (Kullback–Leibler, 1951)** — `D_KL(P‖Q) = Σ P(i)·log(P(i)/Q(i))`.
  Não negativa, **não simétrica**, não limitada superiormente; diverge se `Q(x)=0` onde
  `P(x)>0`. Boa para *ranquear* quais variáveis mais mudaram.
- **JS (Jensen–Shannon)** — média simétrica de duas KLs contra `M = ½(P+Q)`.
  Varia de **0 a 1** (log base 2); a raiz quadrada é uma métrica de distância válida.
  Estável e comparável entre atributos.
- **PSI (Population Stability Index)** — `PSI = Σ (p_j − q_j)·ln(p_j/q_j)`, com a variável
  dividida em bins. Derivado da KL, padrão no setor bancário.

  | PSI | Leitura |
  |---|---|
  | < 0,10 | estável |
  | 0,10 – 0,25 | drift moderado — monitorar |
  | > 0,25 | drift substancial — ação imediata |

  Vantagens: interpretável, simétrico, sem suposição de distribuição.
  **Limitação:** pode não perceber mudanças de *forma* (ex.: mesma média e variância, mas
  distribuição virou multimodal). Não fornece p-valor.
- **Wasserstein / Earth Mover's Distance** — custo de transformar P em Q; no caso 1D é a
  área entre as CDFs. Mais estável e sensível a mudanças sutis de forma; robusta em alta
  dimensão.

### Testes

- **Kolmogorov–Smirnov (KS)** — `D = sup_x |F_ref(x) − F_novo(x)|`, não paramétrico,
  para variáveis **contínuas**. Sensível a posição, dispersão e assimetria.
  Limitação: **uma variável por vez**, fraco em alta dimensão.
- **Qui-quadrado (χ²)** — para **categóricas** ou binadas. Exige contagem esperada
  ≳ 5 por categoria (agrupar categorias raras).

### Código-chave

```python
from scipy import stats
estatistica, p_valor = stats.ks_2samp(dados_ref, dados_novos)

psi = np.sum((dist_atual - dist_referencia) * np.log(dist_atual / dist_referencia))
```

Dado citado: modelos sem atualização por mais de 6 meses tiveram **aumento médio de 35%
na taxa de erro**. Bancos que automatizaram a detecção relatam **redução de 70–80%** no
tempo de validação manual de modelos.

---

## Aula 3 — Estratégias de Mitigação de Drift

**Caso-guia:** churn em telecom; oferta agressiva da concorrência provoca choque abrupto
seguido de reorganização gradual dos hábitos de uso.

> "Mitigar drift não é um detalhe operacional; é uma decisão de arquitetura."
> Entenda como um **ciclo de controle adaptativo: monitorar → diagnosticar → intervir → revalidar.**

### Reativo vs. proativo

- **Reativo** — age depois que a degradação apareceu; depende de detecção ou evidência do negócio.
- **Proativo** — reduz a sensibilidade a mudanças futuras por design (adaptação, redundância,
  tolerância à incerteza).
- Organizações maduras **compõem camadas**: núcleo proativo + conjunto reativo.

### As camadas de resposta (modelo mental central da aula)

```
1. OBSERVABILIDADE   → produz sinais confiáveis de que algo mudou
2. EMERGENCIAL       → ajuste de limiar, isolamento de anômalos, regras de negócio
3. ADAPTATIVA        → aprendizado incremental, ensembles
4. ESTRUTURAL        → retreino, revisão de features, nova janela temporal, revalidação
```

> "O profissional maduro não pergunta apenas 'qual algoritmo usar?', mas 'em que sequência
> de alavancas este sistema deve operar quando a realidade mudar?'"

### Tabela de estratégias (Tabela 1 da aula)

| Estratégia | Natureza | Tempo de resposta | Depende de rótulos recentes | Risco principal |
|---|---|---|---|---|
| Retreinamento sob demanda | Reativa | Dias | Alta | Reagir tarde demais |
| Aprendizado contínuo | Proativa | Segundos a dias | Média a alta | Esquecer padrões úteis |
| Ensemble adaptativo | Proativa | Minutos a dias | Média | Complexidade operacional |
| Ajuste de limiar | Reativa | Minutos | Baixa | Queda de precisão |
| Tratamento de outliers | Híbrida | Minutos a horas | Baixa | Descartar sinal legítimo |
| Regras de negócio | Reativa | Imediata | Nenhuma | Baixa escalabilidade |

### Detalhes importantes

**Retreinamento** — o problema central é a **escolha da janela**: curta o suficiente para
captar a nova realidade, longa o suficiente para não capturar só o ruído do evento.

**Reponderação de instâncias** (útil quando rótulos demoram, sob covariate shift):

```
R̂_t(f) ≈ (1/n) Σ_i [p_t(x_i) / p_s(x_i)] · ℓ(f(x_i), y_i)
```
Exemplos que ficaram mais prováveis no presente pesam mais na atualização.

**Validação não é opcional.** A pergunta certa não é "o novo modelo está melhor em média?",
mas **"ele está melhor nos segmentos críticos, sob o custo de erro relevante, e sem
regressões inaceitáveis?"**. Ferramentas: validação segmentada, *shadow mode*,
*champion/challenger*, *canary release*.

**Aprendizado incremental** — `θ_{t+1} = θ_t − η_t ∇_θ ℓ(f_θt(x_t), y_t)`.
Traz o dilema **plasticidade × estabilidade**: rígido demais envelhece, plástico demais
aprende ruído. Mitigadores: janelas deslizantes, fatores de esquecimento, detectores de
mudança, validação contínua. *"Adaptação rápida é desejável, mas adaptação cega é perigosa."*

**Ensembles adaptativos** — redundância estatística: se um membro perde aderência, outros
seguram a performance. Adaptive Random Forest distribui detectores entre os membros, então
**apenas partes do sistema são reinicializadas** — ganho de governança, não só preditivo.

**Ajuste de limiar como política de decisão** (não detalhe de implementação):
```
t* = argmin_t [ C_FN·π·(1 − TPR(t)) + C_FP·(1 − π)·FPR(t) ]
```

**Outliers ≠ drift.** Um caso extremo pode ser ruído, fraude, erro de integração — ou o
**início de um novo comportamento relevante**. Anomalia deve servir como **mecanismo de
triagem**, não como "borracha estatística". Técnicas: Isolation Forest, One-Class SVM.

**Regras de negócio** como *guardrails* organizacionais durante transições abruptas —
válidas como contenção **temporária**, revisada e conectada a métricas.

### As 4 perguntas que definem a estratégia

1. Quão rápido o ambiente muda?
2. Quão rápido os rótulos chegam?
3. Quanto custa errar durante a adaptação?
4. Qual complexidade operacional a organização sustenta?

### Código-chave

```python
from river import drift
detector = drift.ADWIN()
for i, value in enumerate(stream):
    detector.update(value)
    if detector.drift_detected:
        print(f"Drift detectado em {i}; iniciar resposta")
        break
```

```python
from river import ensemble, linear_model, preprocessing
model = ensemble.ADWINBoostingClassifier(
    model=preprocessing.StandardScaler() | linear_model.LogisticRegression(),
    n_models=5, seed=42,
)
```

```python
from evidently import Report
from evidently.metrics import ValueDrift, DriftedColumnsCount
report = Report([ValueDrift(column="prediction"), DriftedColumnsCount()])
run = report.run(reference_data=reference_ds, current_data=current_ds)
```

> Checkpoint da aula: *"detectar a mudança sem saber como responder a ela é apenas uma
> forma elegante de assistir à degradação acontecer."*

---

## Aula 4 — Métricas Avançadas para Detecção de Drift

**Caso-guia:** fintech em que KS e PSI **não acusam nada** nas marginais de "idade",
"renda" e "dívidas", mas o modelo degrada — a **correlação conjunta** mudou.

### O ponto cego das métricas univariadas

- KS e PSI analisam **uma variável por vez**; são cegos a mudanças na *relação* entre variáveis.
- Em alta dimensão, distâncias euclidianas se concentram (**maldição da dimensionalidade** —
  Bellman 1961; Aggarwal, Hinneburg & Keim 2001), e testes univariados perdem poder.
- Exemplo canônico (Figura 1 da aula): idade e renda têm as mesmas marginais em 2025 e 2026,
  mas a **correlação inverte de positiva para negativa**. Nenhuma métrica univariada captura.

### MMD (Maximum Mean Discrepancy) — Gretton et al., 2012

Teste de duas amostras em espaço de kernel (RKHS). Projeta os dados num espaço de dimensão
possivelmente infinita via kernel e compara as **médias embutidas**:

```
MMD²(P,Q) = E_{x,x'~P}[k(x,x')] + E_{y,y'~Q}[k(y,y')] − 2·E_{x~P, y~Q}[k(x,y)]
```

Com kernel gaussiano `k(x,x') = exp(−γ‖x−x'‖²)`. Esperança **zero se e somente se P = Q**.
Significância via **permutações** ou aproximação assintótica.

### LSDD (Least-Squares Density Difference) — Sugiyama et al., 2013

Em vez de divergências indiretas via densidades, **estima diretamente** `p(x) − q(x)`
minimizando o erro quadrático médio. Mais eficiente computacionalmente em alta dimensão;
supera o KS em vários cenários.

> MMD e LSDD são **testes não supervisionados**: não precisam de rótulos nem de queda de
> acurácia para sinalizar problema. Ideais quando o feedback demora (fraude, manutenção preditiva).

### Panorama das métricas de divergência/distância

| Métrica | Simétrica? | Limitada? | Multivariada nativa? |
|---|---|---|---|
| KL | ✗ | ✗ | ✗ (supõe independência) |
| JS | ✓ | ✓ (0–1) | ✗ |
| PSI | ✓ | ✗ | ✗ |
| Hellinger | ✓ | ✓ | parcial |
| Wasserstein | ✓ | ✗ | ✓ (robusta em alta dim.) |
| **MMD** | ✓ | — | **✓** |
| **LSDD** | ✓ | — | **✓** |

### Código-chave

```python
def rbf_mmd(X, Y, gamma=1e-5):
    XX = np.exp(-gamma * np.sum((X[:, None] - X[None, :])**2, axis=2)).mean()
    YY = np.exp(-gamma * np.sum((Y[:, None] - Y[None, :])**2, axis=2)).mean()
    XY = np.exp(-gamma * np.sum((X[:, None] - Y[None, :])**2, axis=2)).mean()
    return XX + YY - 2 * XY
```

### Mercado

- **D3Bench** (Müller et al., 2024): **Evidently AI** se destaca na detecção geral de data
  drift; **NannyML** tem vantagem em localizar **o momento exato** do drift e avaliar o
  **impacto na performance**.
- Ferramentas: Evidently AI, NannyML, **Alibi-Detect**.
- Caso **Instacart** (COVID-19): modelo de previsão de estoque caiu de **93% → 61%** de
  acurácia em semanas; as distribuições individuais de produtos pouco mudaram, mas o
  **padrão agregado de compras** mudou completamente — drift multivariado puro.

---

## Aula 5 — Drift em Embeddings e Testes de Duas Amostras

**Caso-guia:** classificador de tópicos em rede social quebra quando um grande evento
mundial faz posts políticos dominarem o feed.

### Drift superficial vs. drift semântico

| | Drift superficial | Drift semântico |
|---|---|---|
| O que muda | formatação, estilo, abreviações, flutuação sazonal de frequência | **sentido/contexto** dos dados |
| Impacto | o modelo frequentemente tolera | degrada de forma silenciosa |
| Detecção | estatísticas simples podem bastar | **só aparece nas coordenadas do embedding** |

Exemplo canônico: **"máscara"** — pré-2020 aparecia em contextos de fantasia/cosmético;
pós-COVID migra no espaço vetorial para perto de termos médicos.

### Embeddings como representação de significado

Vetores densos de alta dimensionalidade (BERT-base → **768 dimensões**) onde proximidade
geométrica ≈ semelhança semântica (Mikolov et al., 2013; Devlin et al., 2019).
Acompanhando a evolução dos embeddings (vetores médios, distribuições de distâncias)
quantifica-se o deslocamento do "espaço semântico".

### As três formas do drift (recapitulação formal)

Há drift sempre que `P_ref(X, y) ≠ P_novo(X, y)`; manifesta-se como
(1) `P(X)` — covariate/virtual drift, (2) `P(y|X)` — concept drift real,
(3) `P(y)` — drift de proporção de saída.

> Em produção, rótulos demoram ou nem existem, então **apenas o tipo (1) é diretamente
> mensurável sem supervisão** — e ainda assim pequenos deslocamentos em `P(X)` costumam
> preceder quedas de performance.

### Testes de duas amostras aplicados a embeddings

- **KS** — `D_{n,m} = sup_x |F_ref(x) − F_novo(x)|`. Para embeddings seria preciso aplicar
  **por dimensão** e corrigir para múltiplas comparações — o que **reduz a sensibilidade**.
  Alternativa simplificada: testar uma estatística agregada (norma do vetor, 1ª componente
  principal) — mas perde estrutura multivariada.
- **MMD** — opera direto no espaço de alta dimensão, **sem redução de dimensionalidade**.
  Maior poder estatístico para drifts sutis em embeddings (Lopez-Paz & Oquab, 2017;
  Feldhans et al., 2021). Implementada na **Alibi-Detect**.

### Código-chave

```python
from transformers import BertTokenizer, BertModel
# embedding = vetor [CLS] da última camada oculta
return outputs.last_hidden_state[:, 0, :].numpy()
```

```python
from sklearn.metrics.pairwise import rbf_kernel
def compute_mmd_rbf(X, Y, gamma=0.5):
    return rbf_kernel(X, X, gamma).mean() + rbf_kernel(Y, Y, gamma).mean() \
           - 2 * rbf_kernel(X, Y, gamma).mean()
```

### Estado da arte

- **DriftLens** (Greco et al., 2024) — framework não supervisionado de detecção em
  **tempo real** sobre distâncias entre distribuições de embeddings de deep learning.
  Até **5× mais rápido** e com forte correlação entre drift detectado e magnitude real.
  Funciona em texto, imagem e voz.
- **Zero-shot Embedding Drift Detection** (NeurIPS 2023) — detecção em embeddings de LLMs
  sem dados rotulados, mirando mitigação de **prompt injection** em chatbots.
- Azure ML, SageMaker e Google Cloud ML já trazem monitores nativos de data drift.
- NannyML foca em **target drift não supervisionado** (sem rótulos verdadeiros).

---

## Aula 6 (conteúdo real) — Monitoramento Contínuo de Modelos e Data SLOs

### Por que monitorar modelos é diferente de monitorar software

Software tradicional falha com erro explícito. Modelos de ML **degradam em silêncio**:
são sensíveis a mudanças no ambiente e nos dados, sofrem *decay*, enfrentam dados fora do
escopo, *feedback loops* e **ataques adversariais**. Monitoramento é a **extensão em
runtime das atividades de teste**.

**Case Zillow (2021):** modelo de precificação imobiliária mal supervisionado não captou a
mudança do mercado na pandemia → **mais de US$ 500 milhões** de prejuízo, fim do programa
iBuyer e perda de confiança dos usuários nas estimativas da plataforma.

### SLI vs. SLO vs. Data SLO

| Conceito | Definição | Exemplo em ML |
|---|---|---|
| **SLI** (Service Level Indicator) | a grandeza medida | acurácia, latência média, taxa de erro, desvio-padrão da previsão |
| **SLO** (Service Level Objective) | a meta/faixa aceitável para o SLI | "acurácia ≥ 85%" |
| **Data SLO** | SLO aplicado à **qualidade dos dados** de entrada | "≤ 1% de nulos por coluna"; "dataset de pagamentos atualizado a cada 12h" |

> O SLO funciona como **buffer interno** para proteger o SLA externo: se o SLA promete
> atualização diária (24h), o SLO interno de 12h dispara o alerta com folga para agir.

**Dimensões de qualidade que sustentam Data SLOs:** *usefulness*, acurácia, completude,
consistência, unicidade, validade e atualidade (*freshness*).

### Calibração: o SLI esquecido

Além de acertar, as probabilidades precisam ser fiéis. **Brier Score**:

```
BS = (1/N) Σ_i (p_i − y_i)²     (quanto menor, melhor)
```

Ovadia et al. (2019): sob *dataset shift*, métodos modernos de deep learning e inferência
bayesiana não só perdem acurácia como ficam **super ou sub-confiantes**. Logo, monitore
**confiança/incerteza**, não só performance nominal.

### Arquitetura típica de monitoramento

1. **Log** de previsões + rótulos reais em produção → cálculo dos SLIs
2. **Verificação periódica** de qualidade e integridade dos dados
3. **Detecção automática de drift** nos dados e no comportamento do modelo
4. **Dashboards + alertas** (Prometheus, ELK, Grafana; notificação em Slack/Teams)

Ferramentas citadas: Arize AI, Fiddler (plataformas); **Evidently** (relatórios de drift e
performance), **Great Expectations** (qualidade de dados), **Deepchecks** (integridade e
consistência), TFDV/TFX, MLflow Monitoring.

### Cases positivos

- **LinkedIn** — sistema interno de monitoramento **reduziu pela metade** o tempo de
  detecção de bugs em algoritmos de recomendação.
- **Uber Michelangelo** — módulos de tracking de dados e performance com ajuste rápido e
  automatizado quando métricas saem do padrão.

### Tendências

- Fechamento do ciclo MLOps: pipelines que **detectam e já corrigem** (disparar retreino,
  reverter para versão anterior) sem intervenção humana.
- **Deep Isolation Forest** (Xu et al., 2023): +15–20% de AUC-ROC sobre o Isolation Forest clássico.
- **LLMs para root cause analysis** (Roy et al., 2024): agentes que diagnosticam falhas a
  partir de logs e métricas.
- Lacuna apontada pela literatura: aspectos de **ML justo e ética** ainda sub-representados
  nas soluções de monitoramento.
- Integração com DevOps/SRE: **violação de SLO vira gatilho** de procedimento de recuperação.

---

## Aula 7 (conteúdo real) — Drift em escala: Big Data e Streaming

**Caso-guia:** **FinBank**, fintech com centenas de milhões de transações diárias que precisa
detectar mudança nos padrões de uso do cartão em tempo real.

### Lambda vs. Kappa

| Característica | **Lambda** (Marz, 2011–13) | **Kappa** (Kreps, 2014) |
|---|---|---|
| Visão geral | batch + streaming, duas camadas reconciliadas | **apenas streaming**, pipeline unificado |
| Latência × precisão | batch preciso mas lento; stream rápido mas aproximado | contínuo, baixa latência; reprocessa histórico reexecutando o mesmo pipeline |
| Complexidade | **duas bases de código** para lógica similar; dois sistemas para manter | um código, infra unificada, menor overhead |
| Uso típico | exige alta precisão histórica e recomputação de grandes janelas | agilidade e simplicidade — ex.: **fraude em tempo real** |

> Para data drift, **Kappa tende a ser mais atraente**: um fluxo unificado, sem conciliar
> duas vias. Para recalcular o drift dos últimos 6 meses, reaplica-se o mesmo pipeline de
> streaming sobre o data lake.

### Stack de streaming

**Apache Kafka** (barramento de eventos) + **Apache Flink** / Spark Structured Streaming
(processamento, estado distribuído, janelas) → métricas de drift em janelas deslizantes →
banco de métricas → serviço de regras → alerta / retreino.

### Otimização de custo e desempenho

- **Processamento distribuído** — dividir o cálculo de métricas entre nós.
- **Janelas móveis** — calcular sobre as últimas 1.000 transações ou 60 minutos, não sobre todo o histórico.
- **Amostragem otimizada** — amostras estatisticamente representativas em vez de cada evento.
  (A 100 mil TPS, calcular PSI/KS por transação é inviável.)
- **Estado agregado in-memory** (windows do Flink/Spark), escalando horizontalmente.
- **Trade-off custo × frequência**: a cada minuto é caro e geralmente desnecessário; janelas
  mensais deixam o modelo degradar sem detecção. Considerar **frequência adaptativa** conforme
  a estabilidade observada.

### PSI vs. KS — quando usar cada um

| | PSI | KS |
|---|---|---|
| Entrega | magnitude interpretável pelo negócio ("termômetro") | rigor estatístico embutido (p-valor) |
| Simetria | sim | — |
| Suposições | nenhuma | nenhuma (não paramétrico), mas exige **n suficiente** |
| Fraqueza | cego a mudanças de forma (ex.: virou multimodal) | p-valor não diz **tamanho do efeito** |
| Limiares | 0,1 / 0,25 são **regras de bolso sem fundamento estatístico rigoroso** (Yurdakul & Naranjo, 2020 propõem limiares por taxa de erro tipo I/II) | α = 0,05 |

**Prática recomendada: combinar os dois.**

### Governança

Reguladores (FATF, BIS) já exigem evidência de acompanhamento de performance e robustez.
Instituições financeiras documentam **PSI em relatórios de validação de modelos**.
Uber Michelangelo tem o módulo **Hue**, que monitora latência, erros, distribuição das
previsões e drift — e **bloqueia automaticamente a promoção de novos modelos** quando detecta anomalias.

### Código-chave

```python
def calcular_psi(baseline, atual, bins=10):
    limites = np.percentile(baseline, np.linspace(0, 100, bins + 1))
    prop_base  = np.histogram(baseline, bins=limites)[0] / len(baseline)
    prop_atual = np.histogram(atual,    bins=limites)[0] / len(atual)
    return np.sum((prop_atual - prop_base) * np.log(prop_atual / prop_base))

if psi_valor > 0.25 or p_value < 0.05:
    print("Drift significativo identificado! Modelo pode requerer atualização.")
```

---

## Aula 8 — Ferramentas para Drift e Validação de Dados

Aula de síntese. Três ferramentas open-source, cada uma respondendo a uma pergunta diferente:

| Ferramenta | Pergunta que responde | Mecanismo |
|---|---|---|
| **Evidently AI** | *O que mudou nos dados?* | Relatórios automatizados comparando referência × atual; dezenas de métricas (KS, Anderson-Darling, Wasserstein, Jensen-Shannon) por coluna, com histogramas comparativos |
| **NannyML** | *Qual o efeito disso no modelo?* | **CBPE** (Confidence-Based Performance Estimation): estima acurácia/AUC **sem rótulos**, por *chunk* temporal, com intervalos de confiança |
| **Great Expectations** | *Esse dado pode entrar?* | "Expectativas" declarativas de qualidade executadas **antes** da predição; falha crítica interrompe o pipeline |

### Como o CBPE funciona

Se o modelo está **bem calibrado**, a distribuição dos *scores* de saída carrega informação
sobre o próprio desempenho. O NannyML compara as distribuições atuais de `p̂` para classes
positiva/negativa com as históricas e estima TPR/AUC por segmento temporal — alertando
**antes** da confirmação manual dos rótulos. *"Uma babá do modelo."*

### Tipos de expectativa em Great Expectations

1. **Integridade de schema** — colunas obrigatórias existem e têm o tipo certo.
2. **Valores e faixas** — `valor_da_transacao` não-nulo e ≥ 0; `pais_do_cliente` em lista válida.
3. **Distribuições e dependências** — "o número de transações por hora não muda mais que 20%
   semana a semana" (**detector precoce de data drift!**) ou `total_pago = preço × quantidade`.

Polyzotis et al. (2019): a validação contínua nos pipelines do Google produziu detecção
precoce de erros, correções mais ágeis e **ganho de qualidade do modelo** ao evitar treino
com dados corrompidos. *"Prevenir é melhor que remediar."*

### Código-chave

```python
from evidently import Report
from evidently.metric_preset import DataDriftPreset
report = Report(metrics=[DataDriftPreset()])
report.run(reference_data=ref_data, current_data=curr_data)
```

```python
import nannyml as nml
cbpe = nml.CBPE(model_metadata=model_meta, metrics=['roc_auc'])
cbpe.fit(reference_data=ref_data, y_true='fraude_real')
estimated_perf = cbpe.estimate(curr_data)
```

```python
import great_expectations as gx
df.expect_column_values_to_not_be_null('transaction_id')
df.expect_column_values_to_be_between('valor_transacao', min_value=0, max_value=10000)
validation_result = df.validate()
```

### Open-source × comercial

| | Open-source (Evidently, NannyML, GX, Alibi-Detect) | Comercial (SageMaker Model Monitor, Azure Monitor for ML, Fiddler, WhyLabs, Arize) |
|---|---|---|
| Prós | customização total, sem lock-in, custo | integração nativa com a cloud, UI pronta, suporte empresarial |
| Contras | integração e operação por conta da equipe | **lock-in**, menor customização |

Critérios de escolha: facilidade de integração, custo, expertise do time, requisitos específicos.
**O essencial é ter algum mecanismo de observabilidade contínua.**

### Cases e tendências

- **Uber** — detecta em minutos quando mudanças de comportamento afetam predições de ETA.
- **LinkedIn** — retreino semanal do algoritmo de recomendação de vagas ao detectar drift no
  perfil das interações.
- **Meta/Facebook** — concept drift nos modelos de detecção de discurso impróprio durante a
  COVID; resposta foi monitoramento agressivo + atualizações frequentes.
- **Gartner (2024):** até o fim da década, **mais de 80%** das organizações maduras em IA
  terão **SLOs específicos para dados e modelos** em produção.
- Consolidação de **AIOps** e **DataOps** unindo data science, engenharia e TI em processos de
  resposta a incidentes de dados.

---

## Referências recorrentes do módulo

- **Gama, J. et al. (2014)** — *A Survey on Concept Drift Adaptation*. ACM Computing Surveys 46(4).
- **Lu, J. et al. (2019)** — *Learning under Concept Drift: A Review*. IEEE TKDE 31(12).
- **Sculley, D. et al. (2015)** — *Hidden Technical Debt in Machine Learning Systems*. NIPS 28.
- **Gretton, A. et al. (2012)** — *A Kernel Two-Sample Test*. JMLR 13. (MMD)
- **Rabanser, S.; Günnemann, S.; Lipton, Z. (2019)** — *Failing Loudly*. NeurIPS 32.
- **Moreno-Torres, J. G. et al. (2012)** — *A Unifying View on Dataset Shift in Classification*.
- **Quionero-Candela, J. et al. (2009)** — *Dataset Shift in Machine Learning*. MIT Press.
- **Yurdakul, B.; Naranjo, J. (2020)** — *Statistical Properties of the PSI*. J. Risk Model Validation.
- **Gomes, H. M. et al. (2017)** — *Adaptive Random Forests for Evolving Data Stream Classification*.
- **Greco, S. et al. (2024)** — *Unsupervised Concept Drift Detection from Deep Learning
  Representations in Real-Time* (DriftLens). arXiv:2406.17813.
- **Müller, R. et al. (2024)** — *Open-Source Drift Detection Tools in Action* (D3Bench). arXiv:2404.18673.
- **Ovadia, Y. et al. (2019)** — *Can You Trust Your Model's Uncertainty?*. NeurIPS 32.
- **Breck, E. et al. (2017)** — *The ML Test Score*. IEEE Big Data.
- **Montiel, J. et al. (2021)** — *River: Machine Learning for Streaming Data in Python*. JMLR 22.
