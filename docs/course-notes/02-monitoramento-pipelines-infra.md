# Monitoramento de Pipelines e Infraestrutura — resumo do módulo (Aulas 1 a 8)

> Fontes: `Aula 1_Monitoramento_Observabilidade_ML.pdf` … `Aula 8_Automacao_Melhoria_Continua_AIOps.pdf`

---

## 0. Fio condutor do módulo

```
Aula 1  POR QUÊ        → MLOps, dívida técnica, 3 pilares, SRE (SLI/SLO/error budget)
Aula 2  ORQUESTRAÇÃO   → Airflow (DAGs, métricas StatsD, SLAs) + Datadog
Aula 3  ALERTAS        → taxonomia de falhas, data contracts, incident mgmt, circuit breakers
Aula 4  MODELO         → métricas de produção, tipos de drift, janelas, champion/challenger
Aula 5  OBSERVABILIDADE→ logs estruturados, métricas de negócio, OpenTelemetry/traces
Aula 6  RASTREABILIDADE→ MLflow Tracking + Model Registry + CI/CD
Aula 7  INFRA          → CloudWatch, Azure Monitor, Prometheus/Grafana, GPU, FinOps
Aula 8  AUTOMAÇÃO      → Continuous Training, guardrails, deploy seguro, AIOps, self-healing
```

---

## Aula 1 — Por que Monitorar Pipelines e Infraestrutura?

### A premissa

> Software tradicional: um bug causa um **erro visível**.
> ML: o modelo **degrada silenciosamente**. O endpoint devolve HTTP 200, com latência
> aceitável, e as predições estão completamente erradas.

Cenário de abertura: modelo com 95% de acurácia aprovado e implantado; três meses depois
as conversões caem 20% — ninguém percebeu porque não havia monitoramento.

### MLOps e o ciclo de vida

MLOps (Kreuzberger, Kühl & Hirschl, 2023) = práticas de DevOps aplicadas ao ciclo de vida
de ML, com **dados e modelos** acrescentados ao ciclo. Princípios: **automação,
versionamento, monitoramento contínuo e reprodutibilidade**.

```
Dados → Feature Engineering → Treino → Validação → Deploy → Monitoramento → Retreino
   ↑                                                                            │
   └────────────────────────────────────────────────────────────────────────────┘
```

**O ciclo não termina no deploy — o monitoramento fecha o loop.**

### Riscos específicos de ML (Sculley et al., 2015)

| Risco | O que é |
|---|---|
| **Erosão de fronteiras** | o modelo mistura sinais de várias fontes; é difícil isolar o impacto de uma mudança |
| **Emaranhamento (entanglement)** | alterar uma feature afeta o comportamento de todas as outras (**CACE**) |
| **Feedback loops ocultos** | o modelo influencia o ambiente que gera os dados que alimentam o próprio modelo |
| **Dependências de dados não declaradas** | mudanças upstream alteram silenciosamente a distribuição de entrada |

### Os três pilares da observabilidade

Observabilidade (origem na teoria de controle) = **capacidade de entender o estado interno
de um sistema a partir de suas saídas externas**.

| Pilar | O que é | Em ML |
|---|---|---|
| **Logs** | registros discretos de eventos | início/fim de treino, predições, erros de validação, decisões de retreino |
| **Métricas** | valores numéricos agregados no tempo | infra (CPU, memória, latência), modelo (acurácia, precisão, recall), dados (volume, distribuição, completude) |
| **Traces** | caminho de uma requisição por um sistema distribuído | do dado bruto até a predição final, passando por cada transformação |

Shankar & Parameswaran (2022): observabilidade para ML deve ir **além** dos três pilares,
incluindo **detecção de bugs específicos de ML, diagnóstico assistido e reação automatizada**.

### SRE aplicado a ML (Beyer et al., 2016; Chen et al., 2022)

- **SLIs** para ML: latência de inferência (p50/p95/p99), taxa de erro de predição,
  throughput, **freshness do modelo** (tempo desde o último retreino).
- **SLOs**: "99,5% das inferências com latência < 200ms"; "acurácia em produção > 85%".
- **Error Budget** = 100% − SLO. Se o SLO de disponibilidade é 99,9%, o budget é 0,1%.
  Permite **equilibrar velocidade de inovação com confiabilidade** — enquanto houver budget,
  a equipe pode fazer deploys mais arriscados.

### As quatro camadas a monitorar

| Camada | Métricas |
|---|---|
| **Infraestrutura** | CPU/GPU, memória, I/O de disco e rede, custos |
| **Pipeline** | duração de cada etapa, taxa de sucesso/falha de tasks, volume processado, latência end-to-end |
| **Modelo** | acurácia/F1/AUC, distribuição de predições, latência de inferência, taxa de erros |
| **Dados** | completude, distribuição de features, schema compliance, freshness |

### Amazon CloudWatch — vocabulário

- **Namespaces** — agrupamento lógico (`MLPipeline/Training`, `MLPipeline/Inference`)
- **Dimensions** — pares nome-valor (`ModelName=fraud-detector`, `Environment=production`)
- **Alarms** — thresholds ou anomalias
- **Dashboards** / **Logs Insights**

**Padrões de coleta:** **Push** (a aplicação envia, via SDK/agente) vs. **Pull**
(um coletor busca métricas expostas — padrão Prometheus).

### Toil em ML

*Toil* (SRE) = trabalho manual, repetitivo, automatizável e sem valor duradouro.
Exemplos em ML: retreinar manualmente quando a performance cai; validar dados à mão antes
de cada treino; **investigar alertas não acionáveis**; gerar relatórios manualmente.
**Reduzir toil é objetivo central do MLOps.**

---

## Aula 2 — Monitoramento de Pipelines de Dados (Airflow + Datadog)

### DAG e arquitetura do Airflow

Um **DAG** (Directed Acyclic Graph) é um conjunto de tasks com dependências, sem ciclos.

```
ingest_data → validate_data → feature_engineering → train_model → evaluate_model → deploy_model
     │                                                                   │
     └── alert_on_failure                                                └── register_model
```

| Componente | Papel |
|---|---|
| **Scheduler** | o cérebro — monitora DAGs/tasks, verifica dependências, horários e estados |
| **Executor** | executa tasks: `LocalExecutor` (dev), `CeleryExecutor` (produção, fila de mensagens), `KubernetesExecutor` (um pod por task, escalabilidade) |
| **Workers** | processos que rodam o código das tasks |
| **Metadata DB** | PostgreSQL com estado de DAGs, tasks, variáveis, conexões e logs |
| **Web Server** | UI de visualização e gerenciamento |

**Conceitos:** Operators (`PythonOperator`, `BashOperator`, `S3ToRedshiftOperator`);
Sensors (`S3KeySensor`, `ExternalTaskSensor`, `HttpSensor`); **XComs** (troca de pequenos
dados entre tasks); **Pools** (limitam concorrência por recurso compartilhado);
**Connections** (credenciais e configs).

### Métricas nativas (via StatsD)

| Grupo | Métricas |
|---|---|
| Scheduler | `scheduler_heartbeat`, `dag_processing.total_parse_time`, `dag_processing.processes` |
| Tasks | `ti.finish.<dag_id>.<task_id>.<state>`, `ti.duration.<dag_id>.<task_id>`, `ti.start.*` |
| Pools | `pool.open_slots.<pool>`, `pool.used_slots.<pool>`, **`pool.starving_tasks.<pool>`** |
| DAG | `dagrun.duration.success.<dag_id>`, `dagrun.duration.failed.<dag_id>`, `dag.callback_exceptions` |

### SLAs no Airflow

```python
task = PythonOperator(
    task_id='train_model',
    python_callable=train_fn,
    sla=timedelta(hours=2),          # deve completar em até 2 horas
    on_failure_callback=alert_team,
)
```
Violação de SLA → e-mail, callback customizado, registro no metadata DB.

### Padrões de falha comuns

| Falha | O que é | Mitigação |
|---|---|---|
| **Dependency Hell** | task depende de sistema externo indisponível | timeouts, retries com **backoff exponencial**, circuit breakers |
| **Resource Starvation** | tasks demais competindo por CPU/memória/conexões | **pools**, prioridades, resource quotas |
| **Zombie Tasks** | o scheduler acha que roda, mas o processo morreu | detecção nativa de zombies (pode demorar) |
| **Sensor Deadlocks** | sensor nunca satisfeito ocupa slot indefinidamente | `timeout` e **`mode='reschedule'`** em vez de `mode='poke'` |

### Integração com Datadog

```ini
# airflow.cfg
[metrics]
statsd_on = True
statsd_host = localhost
statsd_port = 8125
statsd_prefix = airflow
```

**Tags** para filtrar/agrupar: `dag_id`, `task_id`, `environment`.
**Dashboards:** heatmap de execução de DAGs, top N tasks mais lentas, taxa de sucesso/falha
por DAG, utilização de pools, tendências de duração.

**Alertas inteligentes (além de thresholds estáticos):**

- **Anomaly Detection** — aprende o padrão normal da métrica; alerta em desvios, sem threshold fixo.
- **Forecast Alerts** — prevê o valor futuro (ex.: "o disco vai encher antes da próxima janela de manutenção").
- **Composite Alerts** — combinam condições (ex.: duração do DAG > SLA **E** volume de dados menor que o esperado).

### Correlação pipeline ↔ qualidade do modelo (prática avançada)

- Feature engineering demorou 3× mais que o normal → **os dados podem ter mudado**; verificar data drift.
- Volume ingerido caiu 50% → o modelo pode estar sendo treinado com dados insuficientes.
- Task de validação foi **skipada** → features podem estar com valores inesperados.

> Essa correlação detecta problemas de qualidade **antes** que apareçam nas métricas de performance.

### Alternativas ao Airflow

**Prefect** (simplicidade, Python puro) · **Dagster** (orientado a *assets*, tipagem forte,
integração nativa com data quality) · **Argo Workflows** (nativo Kubernetes) ·
**AWS Step Functions** (serverless). Airbnb (criador), Spotify e Twitter usam Airflow.

---

## Aula 3 — Alertas de Falha em Ingestão/Transformação

### Cenário de abertura

Uma fonte muda o schema: um campo inteiro passa a vir como string. **O pipeline não falha** —
converte silenciosamente, gera features incorretas, o modelo é retreinado com dados
corrompidos e as predições degradam gradualmente.

### Taxonomia de falhas

| Categoria | Descrição |
|---|---|
| **Loud failures** (ruidosas) | task falha, exceção é lançada, processo termina com erro — **as "fáceis"**, detectadas na hora |
| **Silent failures** (silenciosas) | o pipeline completa com sucesso, mas os dados estão errados — **as mais perigosas**, passam semanas ou meses despercebidas |

**Tipos específicos:** *schema drift* (campos novos/removidos/tipo alterado);
*data quality issues* (estruturalmente correto, semanticamente errado); *missing data*
(partições vazias, nulos inesperados); **duplicatas** (especialmente em streaming com
garantia *at-least-once*); **latência excessiva** (dados chegam, mas tarde — tão prejudicial
quanto dados ausentes em sistemas com features em tempo real).

### Data Contracts

Acordos **formais** entre produtores e consumidores de dados, especificando:

1. **Schema** — campos, tipos, constraints
2. **SLAs** — freshness, latência máxima
3. **Qualidade** — completude, unicidade, ranges
4. **Semântica** — significado dos campos e regras de negócio

> Tornam explícitas expectativas antes implícitas, habilitando **detecção automática de violações**.

### Ferramentas de validação

| Ferramenta | Característica |
|---|---|
| **Great Expectations** | "suites" de expectativas declarativas executadas automaticamente |
| **Deequ** (Amazon) | validação em escala, integrada ao Apache Spark |
| **Pandera** | validação de DataFrames pandas, tipagem forte, integração com type hints |

```python
expect_column_values_to_be_between("age", min_value=0, max_value=150)
expect_column_values_to_not_be_null("customer_id")
expect_column_values_to_be_unique("transaction_id")
expect_column_mean_to_be_between("purchase_amount", min_value=10, max_value=500)
```

### Teoria de alertas

Equilíbrio entre **sensibilidade** (pegar problemas reais) e **especificidade** (evitar falsos positivos).

| Eixo | Níveis |
|---|---|
| **Severidade** | `Critical` (impacto imediato em produção) · `Warning` (potencial problema) · `Info` (awareness) |
| **Prioridade** | `P1` resposta em minutos · `P2` em horas · `P3` em dias |
| **Routing** | infra → SRE/DevOps · qualidade de dados → data engineering · performance de modelo → ML engineering |
| **Escalation** | 15 min sem resposta → líder técnico · 30 min → gerente · 1h → diretor |

### Duas regras de ouro

1. **Todo alerta deve ter uma ação clara associada.** Se não há nada que alguém possa fazer
   ao recebê-lo, **ele não deveria existir**.
2. **Todo alerta deve ter link para um runbook**: o que o alerta significa, como investigar a
   causa raiz, passos de mitigação e quando escalar.

**Redução de falsos positivos:** múltiplos períodos de avaliação antes de disparar; composite
alarms; thresholds calibrados por dados históricos; anomaly detection em vez de threshold estático.

### Incident Management

```
Detecção → Triagem → Mitigação → Resolução → Post-mortem
```

Perguntas de **triagem**: o modelo em produção está afetado? Dados corrompidos já foram
usados para treino? Qual o **blast radius**?

**Mitigação imediata:** reverter para o último modelo válido; pausar o pipeline para evitar
propagação; ativar fallback.

### Circuit Breakers e Fallbacks

**Circuit breakers** interrompem o processamento em condições anômalas:
se registros inválidos > 5% → parar a ingestão; se o schema mudou → pausar o pipeline;
se a latência da dependência excede o timeout → usar cache.

**Fallback strategies para ML:**
- **Último modelo válido** em standby para rollback rápido
- **Cache de features** quando o pipeline de features falha
- **Degradação graceful** — predições com menor confiança ou regras de negócio
- **Default predictions** quando o modelo não está disponível

### Política de alerta (o "como")

| Canal | Uso |
|---|---|
| PagerDuty / OpsGenie | alertas críticos, resposta imediata |
| Slack / Teams | warnings e informativos |
| E-mail | relatórios e tendências |
| Dashboard | visibilidade contínua |

Horários: críticos **24/7 com rotação de on-call**; warnings em horário comercial.
**Agrupamento e deduplicação** evitam flood: agrupar relacionados, suprimir duplicados numa
janela, correlacionar alertas de sistemas diferentes que indicam o mesmo problema.

### Tendências

Plataformas de **data observability** dedicadas: **Monte Carlo, Bigeye, Anomalo**.
Movimento de **"shift left"** em qualidade de dados — validar em **cada etapa** criando
**quality gates**, em vez de só no fim do pipeline. E **ML para monitorar ML** (anomaly
detection que aprende o normal, sem regras manuais).

---

## Aula 4 — Monitoramento de Modelos em Produção e Detecção de Drift

### Métricas de treino ≠ métricas de produção

Razões: **distribuição diferente**, **escala diferente** (volume e velocidade),
**feedback atrasado** (nem sempre há ground truth na hora).

| Com ground truth | Métricas de proxy (sem ground truth) |
|---|---|
| Acurácia, precisão, recall, F1 | distribuição de predições (histograma de scores) |
| AUC-ROC, AUC-PR | confiança média das predições |
| RMSE, MAE (regressão) | taxa de predições em zonas de incerteza |
| | proporção de classes preditas |

### Monitorar a distribuição de predições

- **Histogramas de scores** — se um classificador binário normalmente produz scores
  **bimodais** (perto de 0 e 1) e de repente eles se concentram no meio, **algo mudou**.
- **Percentis** — p50, p90, p95, p99 dos scores ao longo do tempo.
- **Alertas de anomalia** quando a distribuição desvia da baseline.

### Tipos de drift (recapitulação)

| Tipo | Formalmente |
|---|---|
| **Data drift / covariate shift** | `P(X)` muda, `P(Y\|X)` constante |
| **Concept drift** | `P(Y\|X)` muda, mesmo com `P(X)` constante |
| **Feature drift** | uma ou mais features específicas mudam — causado por mudanças upstream, bugs de feature engineering ou comportamento de usuários |

### Janelas de referência e análise — o trade-off central

| Janela | O que é |
|---|---|
| **Referência (baseline)** | período em que o modelo performava bem: dados de treino/validação, primeiro mês em produção, ou janela deslizante de N dias |
| **Análise (current)** | últimas 24h, última semana, último batch |

- Janelas **muito pequenas** → alta sensibilidade, **muitos falsos positivos**
- Janelas **muito grandes** → baixa sensibilidade, **detecção tardia**
- Janelas **deslizantes** → adaptam-se a mudanças graduais, mas **podem mascarar drift lento**

### Ferramentas

| Ferramenta | Foco |
|---|---|
| **Evidently AI** | relatórios de drift, qualidade de dados e performance; integrável como step de validação |
| **WhyLogs** | logging estatístico — captura **perfis de dados** de forma eficiente para comparação temporal |
| **NannyML** | estimativa de performance **sem ground truth** (CBPE) |

### Mitigação

**Retraining triggers:** schedule-based · performance-based · drift-based · event-based.

**Shadow Models (Champion/Challenger):**
- **Champion** — modelo atual em produção, servindo predições reais
- **Challenger** — novo modelo recebendo os mesmos dados, **sem servir predições**
- Promover o challenger quando superar o champion

**A/B Testing** — dividir tráfego e comparar métricas de **negócio**. Cuidado com tamanho de
amostra para significância, duração do teste e escolha das métricas de decisão (técnicas × negócio).

### Regulação

A **EU AI Act** está impulsionando a adoção de monitoramento, pois exige que organizações
**demonstrem que seus sistemas de IA continuam funcionando conforme esperado após o deploy**.

---

## Aula 5 — Logs, Métricas e Traces: Observabilidade de Sistemas de ML

> "Observabilidade é a capacidade de responder perguntas que você não sabia que precisaria fazer."

### Logs estruturados

❌ Não estruturado: `2026-01-15 03:00:01 INFO Model prediction completed in 1.2s`

✅ Estruturado:
```json
{
  "timestamp": "2026-01-15T03:00:01Z",
  "level": "INFO",
  "event": "prediction_completed",
  "model_name": "fraud-detector",
  "model_version": "v2.3.1",
  "latency_ms": 1200,
  "input_features_count": 45,
  "prediction_score": 0.87,
  "request_id": "req-abc123",
  "trace_id": "trace-xyz789"
}
```

Permite: filtrar por qualquer campo; **agregar métricas a partir de logs**; correlacionar com
traces via `trace_id`; buscar padrões (ex.: predições com score alto que foram lentas).

### O que logar em cada etapa

| Etapa | O que logar |
|---|---|
| **Ingestão** | volume (registros, bytes), fonte e timestamp, erros de parsing/validação, latência |
| **Feature engineering** | features calculadas + estatísticas (min/max/mean/nulls), tempo por feature, **features que falharam ou caíram em default**, versão do pipeline |
| **Treino** | hiperparâmetros, métricas por época, tempo total e por etapa, tamanho do dataset, hardware |
| **Inferência** | input (**ou hash do input, por privacidade**), predição, score/confiança, latência, versão do modelo |

### Níveis de log

`DEBUG` (valores intermediários — só em dev/troubleshooting) · `INFO` (eventos normais) ·
`WARNING` ("feature X tem 10% de nulos, usando default") · `ERROR` (falha numa operação
específica, com fallback) · `CRITICAL` ("modelo não encontrado, endpoint indisponível").

### Centralização

**ELK** (Elasticsearch armazena/indexa + Logstash coleta/transforma + Kibana visualiza) ·
cloud-native (CloudWatch Logs + Logs Insights, Google Cloud Logging, Azure Monitor Logs) ·
**Loki** (Grafana) — alternativa leve, **indexa só labels**, reduzindo custo de armazenamento.

### Métricas de negócio (Nandamuri, 2025)

> Métricas técnicas sozinhas não contam a história completa.

Revenue impact · conversion rate · user engagement · customer satisfaction (NPS, CSAT) ·
operational efficiency.

**Correlações que importam:**
- Latência de inferência ↔ taxa de abandono
- Acurácia do modelo ↔ receita
- Disponibilidade ↔ satisfação do cliente
- **Freshness do modelo ↔ relevância das recomendações**

### Dashboards efetivos

**Hierarquia de informação:**
1. **Overview** — saúde geral: verde/amarelo/vermelho
2. **Drill-down** — métricas por componente (pipeline, modelo, infra)
3. **Detail** — métricas granulares para investigação

Boas práticas: cores consistentes; mostrar **tendências**, não só valores instantâneos;
incluir comparação com baseline (semana anterior); **limitar a 5–7 painéis por dashboard**.

### OpenTelemetry e traces

**OTel** é o padrão aberto da **CNCF** que unifica traces, métricas e logs.
Um **trace** é o caminho completo de uma requisição, composto por **spans**:

```
Trace: predict_request (total: 1200ms)
├── Span: receive_request       (5ms)
├── Span: preprocess_input      (150ms)
│   ├── validate_schema         (10ms)
│   ├── normalize_features      (80ms)
│   └── encode_categoricals     (60ms)
├── Span: model_inference       (800ms)
│   ├── load_model              (50ms)
│   └── predict                 (750ms)
├── Span: postprocess_output    (200ms)
│   ├── apply_business_rules    (100ms)
│   └── format_response         (100ms)
└── Span: send_response         (45ms)
```

Cada span tem: **nome**, **timestamps**, **atributos** (`model_version`, `input_size`),
**eventos** (`cache_hit`, `fallback_used`), **status** (OK/ERROR/UNSET) e **contexto**
(`trace_id`, `span_id`).

```python
from opentelemetry import trace
tracer = trace.get_tracer("ml-pipeline")

def predict(input_data):
    with tracer.start_as_current_span("predict") as span:
        span.set_attribute("model.name", "fraud-detector")
        span.set_attribute("model.version", "v2.3.1")
        with tracer.start_as_current_span("preprocess"):
            features = preprocess(input_data)
            span.set_attribute("features.count", len(features))
        with tracer.start_as_current_span("inference"):
            prediction = model.predict(features)
            span.set_attribute("prediction.score", float(prediction))
        return prediction
```

### A correlação é o que transforma dados em observabilidade

```
Alerta de métrica  →  "latência p99 > 5s"
Investigar traces  →  filtrar traces > 5s, achar o span mais lento
Analisar logs      →  buscar os logs daquele span, achar a causa raiz
```

IDs compartilhados que habilitam isso: `trace_id` (traces + logs), `span_id` (operação
específica), `request_id` (correlaciona com métricas de negócio).

### Observability as Code

Versionar dashboards, alertas e instrumentação **junto com o código** — Grafana
(provisioning) e Terraform (CloudWatch).

### Tendências

OTel consolidado como padrão (adotado por Datadog, New Relic, Dynatrace, Grafana) — reduz
**vendor lock-in**. **"Observabilidade generativa"**: LLMs analisando logs, sugerindo causas
raiz e gerando runbooks (Datadog **Bits AI**, New Relic AI).

---

## Aula 6 — Rastreamento de Experimentos e Modelos com MLflow

### O problema (Zaharia et al., 2018)

> "Qual modelo está em produção agora? Quem treinou? Com quais hiperparâmetros? Usando
> quais dados?"

Desafios únicos de ML: **experimentação massiva**, **múltiplos ambientes**, **dependências
complexas** (versões de bibliotecas, dados e código) e **colaboração**.
O antipadrão: `model_final_v2_FINAL_realmente_final.ipynb`.

### MLflow Tracking

| Conceito | O que é |
|---|---|
| **Experiment** | agrupamento lógico de runs (`"fraud-detection-model"`) |
| **Run** | uma execução específica, com parâmetros, métricas e artefatos |
| **Parameters** | inputs (hiperparâmetros, configurações) |
| **Metrics** | outputs numéricos (acurácia, loss, F1) |
| **Artifacts** | arquivos produzidos (modelo, gráficos, datasets) |
| **Tags** | metadados (autor, versão do dataset, objetivo) |

```python
import mlflow
mlflow.set_experiment("fraud-detection")

with mlflow.start_run(run_name="xgboost-v1"):
    mlflow.log_param("learning_rate", 0.01)
    mlflow.log_param("dataset_version", "2026-01-15")
    model = train_model(X_train, y_train, params)
    mlflow.log_metric("accuracy", 0.95)
    mlflow.log_metric("auc_roc", 0.98)
    mlflow.log_artifact("confusion_matrix.png")
    mlflow.sklearn.log_model(model, "model")
```

### Tracking Server

- **Local** — filesystem local, bom para desenvolvimento individual.
- **Remoto** — **backend store** (PostgreSQL/MySQL para parâmetros e métricas) +
  **artifact store** (S3/GCS/Azure Blob para artefatos e modelos).

```bash
mlflow server \
  --backend-store-uri postgresql://user:pass@host:5432/mlflow \
  --default-artifact-root s3://mlflow-artifacts/ \
  --host 0.0.0.0 --port 5000
```

Consulta programática:
```python
runs = client.search_runs(
    experiment_ids=["1"],
    filter_string="metrics.f1_score > 0.9",
    order_by=["metrics.f1_score DESC"],
    max_results=10,
)
```

### Model Registry

Conceitos: **Registered Model** (`"fraud-detector"`), **Model Version** (versão 3),
**Stages/Aliases**, **Descriptions**.

**Workflow de promoção:** `Development → Staging → Production → Archived`

```python
mlflow.register_model(model_uri="runs:/abc123/model", name="fraud-detector")
client.transition_model_version_stage(name="fraud-detector", version=3, stage="Production")
```

> O **Model Registry é o ponto de controle entre desenvolvimento e operação** —
> garante que apenas modelos validados cheguem à produção.

### Model Signatures e Input Examples

```python
from mlflow.models import infer_signature
signature = infer_signature(X_train, model.predict(X_train))
mlflow.sklearn.log_model(model, "model", signature=signature, input_example=X_train[:5])
```

A **signature documenta o contrato de entrada/saída** (nomes, tipos, shapes das features e da
predição), **prevenindo erros de integração**.

### Versionamento completo (reprodutibilidade total)

| Dimensão | Ferramenta |
|---|---|
| Código | Git (commit hash registrado automaticamente pelo MLflow) |
| Dados | **DVC** ou referência ao dataset no artifact store |
| Modelo | MLflow Model Registry |
| Ambiente | Conda env / `requirements.txt` (registrado pelo MLflow) |
| Configuração | parâmetros logados no MLflow |

### Flavors e serving

`python_function` (genérico), `sklearn`, `pytorch`, `tensorflow`, `xgboost`, `transformers`.
Serving: `mlflow models serve` (REST local), container Docker, endpoints cloud
(SageMaker, Azure ML, Databricks).

### Integração com CI/CD

**CI para ML:** push de código → testes de dados (schema, qualidade) → treino com tracking →
testes de modelo (performance mínima, fairness) → registro no registry se passar.

**CD para ML:** promoção para Production no registry → build de container → deploy em staging
(smoke tests) → deploy em produção (canary ou blue-green) → monitoramento pós-deploy.

### Mercado

MLflow é mantido pela **Databricks**. Alternativas: **Weights & Biases, Neptune.ai, Comet ML,
ClearML**. O MLflow 2.x introduziu o **MLflow Deployments Server** (ex-AI Gateway) para
gerenciar endpoints, incluindo LLMs. Tendência: **feature stores** (Feast, Tecton) integradas
ao MLflow para eliminar **training-serving skew**.

---

## Aula 7 — Monitoramento de Infraestrutura: Cloud e Open-Source

### Métricas de infraestrutura essenciais para ML

| Domínio | Métricas |
|---|---|
| **Computação** | CPU utilization, **GPU utilization**, **GPU memory (VRAM — frequentemente o gargalo)**, RAM |
| **Storage** | Disk I/O (crítico no carregamento de datasets), disk latency, storage capacity (datasets e checkpoints consomem muito) |
| **Rede** | network throughput, network latency entre componentes distribuídos, API request rate |
| **Específicas de ML** | **inference latency (p50/p95/p99)**, batch processing time, **queue depth**, **model load time** |

### AWS CloudWatch

**SageMaker Endpoints:** `Invocations`, `InvocationModelErrors`, **`ModelLatency`**,
**`OverheadLatency`** (serialização, rede), `CPU/Memory/GPUUtilization`.
**Lambda** (inferência serverless): `Duration`, `Errors`, `Throttles`, `ConcurrentExecutions`.
**ECS/EKS:** `CPUUtilization`/`MemoryUtilization` por task/pod, `RunningTaskCount`,
custom metrics via CloudWatch Agent.

### Azure Monitor

**Azure ML Endpoints** (latência, throughput, error rates por tipo, utilização dos compute
targets) · **AKS** (métricas por pod, por node, cluster autoscaler) · **Application Insights**
(distributed tracing, custom events, **Smart Detection** = anomaly detection automática).

### FinOps para ML

> Kreuzberger et al. (2023): **custos de infraestrutura são frequentemente subestimados** em projetos de ML.

**Onde o dinheiro vai:** treino (GPUs — um treino grande custa milhares de dólares);
inferência (endpoints *always-on*); storage (datasets, checkpoints, artefatos, logs);
**data transfer** entre serviços e regiões.

**Práticas:** tagging por projeto/equipe/modelo · dashboards de custo por pipeline e modelo ·
alertas de custo (treino excedeu budget) · **spot/preemptible instances para treino** ·
autoscaling para inferência (**scale to zero** quando possível) · **right-sizing** por
utilização real.

### Prometheus

Arquitetura **pull** — o Prometheus busca métricas nos targets periodicamente.

**Componentes:** Prometheus Server (TSDB) · **Exporters** (expõem métricas de sistemas que não
falam Prometheus nativamente) · **Alertmanager** (routing, silencing, grouping) ·
**Pushgateway** (jobs curtos que não podem ser *scraped*).

```promql
# Taxa de requisições por segundo nos últimos 5 minutos
rate(http_requests_total{job="ml-inference"}[5m])

# Percentil 99 de latência
histogram_quantile(0.99, rate(http_request_duration_seconds_bucket[5m]))

# Uso de GPU acima de 90%
nvidia_gpu_utilization > 90
```

### Grafana

**Data sources** (Prometheus, CloudWatch, Elasticsearch, InfluxDB…) · **Dashboards**
(séries temporais, heatmaps, tabelas, gauges) · **Alerting** integrado (Slack, PagerDuty,
e-mail, webhooks) · **Provisioning** — dashboards e data sources como código (YAML/JSON).

### Stack open-source completa

```
Prometheus (métricas) + Loki (logs) + Tempo (traces) + Grafana (visualização)
                              +
                     Alertmanager (alertas)
```

| Vantagens | Desvantagens |
|---|---|
| sem vendor lock-in | overhead operacional de manter a stack |
| custo controlado | escalabilidade exige expertise |
| customização total | menos features "out of the box" que SaaS |

### Case: 10M requisições/dia

**Arquitetura:** NVIDIA DCGM → métricas de GPU para o Prometheus; node exporter (CPU,
memória, disco); custom exporter (latência, throughput, scores do modelo); Prometheus armazena;
Grafana visualiza; Alertmanager envia para Slack e PagerDuty.

**Autoscaling por métrica:** scale **up** com GPU utilization > 80% por 5 min; scale **down**
com GPU < 30% por 15 min; **warm pools** para reduzir o tempo de scale up.

**Capacity planning:** análise de tendências de tráfego; right-sizing por utilização real vs.
provisionada; **reserva de capacidade para picos** (Black Friday, eventos).

### Monitoramento de GPU — NVIDIA DCGM

GPU Utilization (% de tempo executando kernels) · Memory Utilization (VRAM) ·
**Temperature** (thermal throttling reduz performance) · Power Usage ·
**ECC Errors** (podem indicar hardware degradado) · SM Clock.

### Tendências

Microsoft publicou a arquitetura de referência **MLOps v2** com monitoramento como componente
central. **FinOps para ML** ganhando tração (Kubecost, CloudHealth). Com LLMs, monitorar
**comunicação inter-GPU** (NVLink, InfiniBand) e **eficiência de paralelismo** vira essencial.

---

## Aula 8 — Automação e Melhoria Contínua (Retraining Automático e AIOps)

### Os três pilares do MLOps

| Pilar | O que faz |
|---|---|
| **CI** — Continuous Integration | testar **código, dados e modelos** a cada mudança |
| **CD** — Continuous Delivery | deployar modelos validados automaticamente |
| **CT** — **Continuous Training** | **retreinar modelos automaticamente** em resposta a triggers |

### Triggers de retreino

| Trigger | Vantagem | Desvantagem |
|---|---|---|
| **Schedule-based** | simples e previsível | retreina desnecessariamente ou tarde demais |
| **Performance-based** | retreina só quando necessário | **exige ground truth**, nem sempre disponível rápido |
| **Drift-based** | proativo — detecta antes de afetar performance | **drift nem sempre implica degradação de performance** |
| **Event-based** | responde a nova fonte de dados, mudança de política, feedback de usuários, atualização de dependências | requer integração com eventos de negócio |

### Pipeline de retreino automático + guardrails

```
Trigger → Coleta de Dados → Validação de Dados → Feature Engineering
       → Treino → Avaliação → Validação → Registro → Deploy
```

**Quality gates em cada etapa:**

| Gate | Pergunta |
|---|---|
| Validação de dados | Schema correto? Volume suficiente? Qualidade aceitável? |
| Validação de treino | O modelo convergiu? Métricas dentro do esperado? |
| Validação de modelo | Superior ao modelo atual? **Sem regressões em subgrupos?** |
| Validação de deploy | Smoke tests passam? Latência aceitável? Sem erros? |

> **Se qualquer guardrail falha, o pipeline para e alerta — o modelo em produção permanece inalterado.**

### CI para ML vai além de testar código

**Testes de dados:** schema validation · distribution tests · completeness tests · freshness tests.
**Testes de modelo:** performance mínima · regression tests (não piorou em subgrupos) ·
**fairness tests** (não introduziu bias em grupos protegidos) · latency tests (dentro do SLO) ·
size tests (modelo cabe no deploy).
**Testes de integração:** end-to-end · API contract · compatibility com o serving atual.

### CD para ML — estratégias de deploy

| Estratégia | Como funciona |
|---|---|
| **Canary** | 5% do tráfego para o novo modelo, monitorar, aumentar gradualmente |
| **Blue-Green** | dois ambientes idênticos; "blue" serve o tráfego, "green" recebe o novo modelo; após validação, troca |
| **Progressive Rollout** | canary + automação: aumenta o tráfego sozinho se as métricas estão saudáveis, **faz rollback se detecta problemas** |
| **Shadow Mode** | o novo modelo recebe o mesmo tráfego, **mas suas predições não são servidas** — comparação sem risco |

```
Tráfego → Load Balancer ─┬→ 95% → Modelo v1 (champion)
                         └→  5% → Modelo v2 (canary)
```

### Versionamento completo (auditabilidade)

Código → **Git** · Dados → **DVC** · Modelo → **MLflow Model Registry** ·
Infraestrutura → **Terraform/CloudFormation** · Configuração → Git ·
Ambiente → **imagens Docker com versões fixas**.

### AIOps (Notaro, Cardoso & Gerndt, 2021)

**IA para operar IA** — três capacidades:

1. **Anomaly Detection** — Isolation Forest, autoencoders, LSTM para séries temporais.
   Detecta spikes de latência, padrões incomuns de erro, uso anômalo de recursos.
2. **Root Cause Analysis (RCA)** — correlação temporal, grafos de causalidade, análise de
   propagação. *"A latência subiu porque o disco está cheio porque o job de limpeza falhou."*
3. **Auto-Remediation** — restart de serviços, scale up, rollback de deploy.
   **Requer: playbooks automatizados, guardrails de segurança e limites de ação.**

### Self-Healing Pipelines

| Mecanismo | Resolve |
|---|---|
| **Auto-retry** com backoff exponencial | falhas transientes (timeout de rede, indisponibilidade temporária) |
| **Auto-scaling** | problemas de capacidade |
| **Auto-rollback** | novo modelo degradando métricas → reverte sozinho |
| **Auto-retraining** | drift detectado → aciona o pipeline de retreino **com guardrails** |

### LLMs para AIOps e observabilidade generativa

Análise de logs em linguagem natural · **geração automática de runbooks** a partir de um
alerta · chat-based troubleshooting · **sumarização de incidentes / post-mortems automáticos**.

**Observabilidade generativa:** o sistema não apenas coleta e exibe — **gera insights
proativamente**, identificando tendências, prevendo problemas e sugerindo otimizações antes
que humanos percebam a necessidade.

### Mercado

Players de AIOps: **Datadog Bits AI, New Relic AI, Dynatrace Davis AI, PagerDuty AIOps**.
Tendência de **closed-loop MLOps** (monitoramento → retreino → deploy → monitoramento), que
reduz o tempo entre detecção e resolução de **dias para horas ou minutos**.
A **EU AI Act** exige auditabilidade e rastreabilidade — organizações precisam **demonstrar
controle sobre pipelines de retreino automático e decisões de deploy**.

---

## Referências recorrentes do módulo

- **Beyer, B.; Jones, C.; Petoff, J.; Murphy, N. R. (2016)** — *Site Reliability Engineering:
  How Google Runs Production Systems*. O'Reilly.
- **Chen, C. et al. (2022)** — *Reliable Machine Learning: Applying SRE Principles to ML in
  Production*. O'Reilly.
- **Kreuzberger, D.; Kühl, N.; Hirschl, S. (2023)** — *MLOps: Overview, Definition, and
  Architecture*. IEEE Access 11, 31866–31879.
- **Sculley, D. et al. (2015)** — *Hidden Technical Debt in Machine Learning Systems*. NeurIPS 28.
- **Shankar, S.; Parameswaran, A. G. (2022)** — *Towards Observability for Production ML
  Pipelines*. arXiv:2108.13557.
- **Zaharia, M. et al. (2018)** — *Accelerating the Machine Learning Lifecycle with MLflow*.
  IEEE Data Engineering Bulletin 41(4).
- **Notaro, P.; Cardoso, J.; Gerndt, M. (2021)** — *A Survey of AIOps Methods for Failure
  Management*. ACM TIST 12(6).
- **Nandamuri, S. (2025)** — *Monitoring and Observability in ML Infrastructure: Metrics, Logs,
  and Traces*. WJARR.
- **Pulicharla, M. R. (2019)** — *Detecting and Addressing Model Drift: Automated Monitoring
  and Real-Time Retraining in ML Pipelines*. WJARR.
- **Vuppu, S.; Achanta, V. (2025)** — *Building Robust Data Pipelines*. IJCTT.
- **Gama, J. et al. (2014)** — *A Survey on Concept Drift Adaptation*. ACM CSUR 46(4).
- **Alla, S.; Adari, S. K. (2021)** — *Beginning MLOps with MLflow*. Apress.
- **Intorf, D.; van Doorn, K.; Storey, D. (2024)** — *Apache Airflow Best Practices*. Packt.
