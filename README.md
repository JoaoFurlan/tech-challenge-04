# credit-drift

Camada de sustentação e confiabilidade (contratos de dados, detecção de drift, observabilidade
e governança LGPD) em torno de um modelo de credit scoring. Projeto do Tech Challenge — Fase 4.

**Cenário.** Uma fintech de crédito tem um modelo de credit scoring em produção e suspeita de degradação silenciosa depois
de mudanças econômicas. Este repositório constrói a camada de **sustentação e confiabilidade** ao redor desse modelo: contratos
de dados, simulação e detecção de drift (de dados e de conceito) ao longo de 16 semanas sintéticas, observabilidade, retreino com
portão de promoção e governança (LGPD) com análise causal. Conjunto de dados: Home Credit Default Risk (`application_train.csv`,
307 mil linhas, classificação binária). O plano de implementação está em `docs/build-plan.md`, as decisões em
`docs/decisions.md` e `CLAUDE.md`.

### Mapa do README pela rubrica do Tech Challenge

| Critério (peso) | Onde está |
|---|---|
| **Validação de dados** (25%) | [Contratos de dados](#contratos-de-dados): Pandera na entrada e Great Expectations com Data Docs (`docs/reports/data_docs/index.html`); lote corrompido bloqueado (`uv run python -m credit_drift.contracts.demo`) e bloqueio no laço de produção (semana 9) |
| **Detecção de drift** (25%) | [Simulação de drift](#simulação-de-drift) (cronograma e verdade em `docs/reports/drift_spec.json`), [Detecção de drift](#detecção-de-drift) (PSI, KS, qui-quadrado, JS com Evidently, relatórios HTML em `docs/reports/drift/`) e [Calibração dos limiares](#calibração-dos-limiares) |
| **Observabilidade e logs** (20%) | [Dashboard (Grafana)](#dashboard-grafana), [Métricas de saúde operacionais](#métricas-de-saúde-operacionais), [Monitoramento semanal](#monitoramento-semanal-airflow), experimentos e registro no MLflow, DAGs no Airflow |
| **Governança (LGPD)** (15%) | [Governança e LGPD](#governança-e-lgpd) (dados pessoais, base legal, retenção, viés) e [Causalidade](#causalidade) |
| **Vídeo STAR** (15%) | [Vídeo](#vídeo-star) |

Requisitos obrigatórios do desafio: pipeline de validação (contratos), scripts de simulação e detecção estatística de drift,
dashboards e relatórios HTML, governança consolidada neste README e histórico de commits semânticos. Boas práticas: Referência
e Produção separadas em arquivos distintos; mais de três regras rígidas de qualidade (idade 18–100, renda não nula e positiva,
sem IDs duplicados, esquema e tipos, valores positivos); relatório estatístico de mudança de distribuição (PSI e KS); plano de
privacidade.

## Reproduzindo do zero

Pré-requisitos: git, [uv](https://docs.astral.sh/uv/) e Docker Desktop.

```bash
git clone https://github.com/JoaoFurlan/tech-challenge-04.git
cd tech-challenge-04
uv sync          # instala o Python 3.14 indicado em .python-version e as dependências travadas
```

Em seguida, obtenha os dados por **uma** das opções abaixo.

### Opção A — sem AWS (recomendada para quem avalia o projeto)

Não é necessária nenhuma conta ou credencial AWS.

1. Baixe `application_train.csv` e `HomeCredit_columns_description.csv` da competição
   [Home Credit Default Risk](https://www.kaggle.com/competitions/home-credit-default-risk/data)
   no Kaggle (é preciso aceitar as regras da competição) e salve-os em `data/raw/`.
2. Confirme a integridade: `uv run dvc status data/raw/application_train.csv.dvc data/raw/HomeCredit_columns_description.csv.dvc`.
   O DVC compara o hash dos arquivos com os ponteiros versionados e deve informar que os dados
   estão atualizados (as demais pastas em `data/` só existem depois do passo 3).
3. Todo o restante (partes de referência, lotes semanais com drift simulado) é **regenerado pelo
   código** a partir de sementes fixas; não há dados derivados para baixar. Por exemplo, a divisão
   Referência/Produção: `uv run python -m credit_drift.split`.

### Opção B — com um bucket S3 próprio

O remoto padrão do DVC aponta para o bucket do autor, que é privado: `dvc pull` falhará com
"acesso negado" para qualquer outra pessoa, de propósito. Para usar um bucket seu:

```bash
uv run dvc remote modify --local s3store url s3://SEU-BUCKET/dvc-store
# credenciais suas em ~/.aws/credentials (perfil [default]); nunca versionadas
uv run dvc push   # depois de colocar os dados em data/raw/ pela Opção A
```

### Opção C — autor (bucket do projeto)

Com as credenciais do usuário IAM do projeto em `~/.aws/credentials` e `region = us-east-1` em
`~/.aws/config`: `uv run dvc pull`.

### Serviços locais

```bash
docker compose up -d
```

Os passos seguintes (treino, pipelines, dashboards) serão acrescentados aqui conforme forem
implementados. O estado do MLflow (experimentos e registro de modelos) fica em um volume Docker
local e **não** é compartilhado entre máquinas: o treino é reexecutado em cada ambiente, com
sementes fixas, a partir do que está versionado no git e no DVC.

## Experimentos e modelo campeão

O treino roda em quatro etapas, cada uma um *experimento* do MLflow (`http://127.0.0.1:5000`),
e o melhor resultado de uma etapa alimenta a seguinte (decisões em `docs/decisions.md`, seção 8):

```bash
docker compose up -d                                  # MLflow + Postgres
uv run python -m credit_drift.training.run all        # ou: features | models | tune | select
```

| Experimento | O que compara | Resultado |
|---|---|---|
| `feature_engineering` | variantes A (bruta), B (engenhada), C (selecionada), cada uma com regressão logística e LightGBM | **B** vence (AUC-PR 0,2402 com LightGBM); C empata, mas custa mais; A é claramente pior (0,2333) |
| `model_selection` | regressão logística, LightGBM e XGBoost na variante B | LightGBM e XGBoost empatam (0,2402 e 0,2404); ambos seguem para o ajuste |
| `hyperparameter_tuning` | Optuna (TPE, 25 tentativas por família, poda de tentativas fracas) | LightGBM e XGBoost terminam empatados (AUC-PR 0,2462 na validação cruzada) |
| `champion` | regra de seleção, calibração, avaliação única no teste, registro | **variante B + LightGBM**, calibração de Platt |

**Regra de seleção** (aplicada em todas as etapas): ordenar por AUC-PR em validação cruzada de 3
partes, manter tudo que estiver a até 1 erro-padrão do melhor e escolher o de menor **custo
esperado** (um inadimplente não detectado custa 5 vezes mais que um alarme falso; é uma suposição
que uma instituição real substituiria pelos seus números). O Brier é uma verificação de sanidade:
o modelo calibrado precisa superar a previsão constante da taxa de inadimplência.

**Resultado do campeão no conjunto de teste** (usado uma única vez, depois da escolha):

| Métrica | Valor |
|---|---|
| AUC-PR | 0,246 (IC 95%: 0,232–0,261); um modelo sem poder preditivo teria 0,081 |
| AUC-ROC | 0,760 |
| Brier calibrado | 0,0680 (constante: 0,0745) |
| Probabilidade média prevista | 8,09% (observada: 8,10%) |

Interpretação prática: os 10% de solicitantes com maior risco têm 27,9% de inadimplência (3,4
vezes a média) e concentram 34% de todos os inadimplentes. Com o limiar de decisão de 1/6 (custo
5:1), o modelo sinaliza 12% dos solicitantes e captura 38,5% dos inadimplentes, com 26% de
precisão. O desempenho é moderado de propósito: usamos só a tabela principal do Home Credit, sem
as demais tabelas de histórico, porque o foco do projeto é o monitoramento em torno do modelo.

## Simulação de drift

Os dados de produção (100 mil linhas separadas do treino) são cortados em **16 semanas** de 5.000
clientes, sem sobreposição, e um estado de "mundo" é aplicado a cada semana
(`uv run python -m credit_drift.drift.simulate`; decisões em `docs/decisions.md`, seção 10). O tempo é
sintético: os dados não têm datas.

| Semanas | O que acontece |
|---|---|
| 1–2 | controle: nenhum drift (os detectores **não** devem disparar) |
| 3–4 | drift gradual de covariáveis: renda ×0,90 → ×0,86 e tempo de emprego ×0,92 → ×0,85; a mistura de tipos de renda começa a mudar |
| 5 | choque abrupto: parcela ×1,15 (a razão parcela/renda dispara) e mistura completa (pensionistas de 18% para 12%) |
| 6–7 | novo normal: o regime da semana 5 continua |
| 8, 10–16 | **drift de conceito**: para clientes até 45 anos com renda abaixo da mediana (≈26%), a relação de `EXT_SOURCE_2` com a inadimplência é invertida e reduzida à metade; nenhuma variável muda |
| 9 | lote corrompido (12% de rendas nulas, idades < 18, IDs duplicados, coluna numérica como texto): o contrato bloqueia o lote e os rótulos nunca chegam |

Cada semana gera um arquivo de *features* e outro de *rótulos* (em produção os rótulos chegam depois),
e `drift_spec.json` registra o que foi injetado: é o gabarito usado **apenas offline**
(`docs/reports/drift_spec.json`).

O que isso faz com o modelo campeão (resultado medido, não suposto):

- O drift de covariáveis mexe nas distribuições e na probabilidade média prevista (de 8,2% para
  cerca de 8,8%, enquanto a inadimplência observada fica perto de 8,3%), mas quase não muda o AUC-PR
  (0,251 → 0,238 só com o choque da parcela).
- O drift de conceito faz o contrário: nenhuma distribuição se mexe, mas o AUC-PR cai cerca de 0,04
  e o AUC-ROC vai de 0,76–0,78 para 0,72–0,74. PSI e KS **não** veem isso; só o desempenho (com
  rótulos) mostra.
- Uma única semana de 5.000 clientes é ruidosa (intervalo de confiança do AUC-PR de ±0,04), por isso
  as métricas por fatia usam janelas de duas semanas.

## Monitoramento semanal (Airflow)

O DAG **`production_monitoring`** é o laço de produção: cada execução é uma semana simulada (1 a 16, em ordem), e o
conjunto roda sozinho em cerca de 25 minutos, com os retreinos. Por semana: o lote passa pelo **contrato de dados** (Pandera e Great
Expectations); se for rejeitado, vai para **quarentena** e nem é pontuado; se passar, é **pontuado pelo servidor do
modelo via HTTP** e gera o relatório de drift. Em paralelo, como os rótulos chegam com **uma semana de atraso**, a
execução da semana N fecha a semana N-1: AUC-PR, Brier, taxa prevista contra observada e 14 fatias (janela de duas
semanas). Em seguida vêm as **regras de alerta**, o **diagnóstico** do tipo de drift (comparando com a semana
anterior, não com a Referência), o registro no MLflow e o envio das métricas ao Pushgateway/Prometheus.

| Semanas | O que o laço disse (limiares congelados, decisão #17) |
|---|---|
| 1–2 | nenhum alerta (controle) |
| 3 | sem alerta (PSI logo abaixo de 0,10), mas já diagnosticado como mudança de covariáveis |
| 4 | primeiro alerta (`data_drift_moderate`) |
| 5 | `data_drift_major` (parcela/crédito) |
| 6–7 | o alerta de PSI continua (a base ainda é a do treino), mas o diagnóstico é "nada": o novo normal parou de mudar |
| 8 | diagnóstico "nada": a média móvel de 3 semanas ainda não caiu abaixo da base (atraso aceito, decisão #16) |
| 9 | lote bloqueado pelo contrato: não pontuado, em quarentena, alerta crítico e diagnóstico "técnico" |
| 10–15 | drift de conceito diagnosticado; `performance_drop` a partir da semana 11 (sobre a semana 10) |
| 16 | só o alerta de PSI: os rótulos chegariam em uma semana 17 que não roda |

Limitações assumidas e registradas em `docs/decisions.md` (seção 14): o alerta de PSI se repete até o modelo ser
retreinado e a base ser trocada; o ranking de fatias aponta renda baixa e idade baixa (as variáveis que definem o
segmento injetado) por volta da semana 11, mas o limiar de sinalização é conservador e assim foi mantido (decisão #16); a matriz de fatias do Grafana faz o ranking.

## Dashboard (Grafana)

O Grafana (`http://localhost:3000` pelo túnel SSH) reúne a história das 16 semanas em uma tela, provisionado como
código (`docker/grafana/`): severidade do drift (PSI contra a Referência e contra a janela do campeão, com as faixas
0,10 e 0,25), variáveis acima dos limites, drift do score, AUC-PR e Brier contra a linha de base do campeão, taxa
prevista contra observada, uma matriz de fatias (queda de AUC-PR por fatia e semana), a tabela de diagnóstico e de
alertas por semana, o resultado do contrato de dados e tráfego/latência do servidor. As semanas são o eixo (rótulo
`week`), e os painéis que dependem de rótulos só mostram a semana N depois da semana N+1, refletindo o atraso real.
Decisões em `docs/decisions.md` (seção 15).

### Métricas de saúde operacionais

Documentadas aqui porque a Etapa 3 pede as métricas de saúde escolhidas. Todas chegam ao Prometheus (Pushgateway para as
tarefas em lote do Airflow, `/metrics` para o servidor) e aparecem no Grafana.

| Dimensão | Métrica | O que mostra |
|---|---|---|
| Execução do pipeline | estado de cada tarefa do DAG (Airflow), `batch_blocked`, `batch_contract_warnings` | sucesso ou falha na ingestão e nas transformações; lote bloqueado pelo contrato |
| Volume de dados | `batch_rows`, `predict_rows_total`, `predict_requests_total` | linhas processadas por semana e tráfego do servidor |
| Latência e infraestrutura | `predict_latency_seconds` (histograma, p95 no painel), `model_info`, `model_reloads_total` | estabilidade e tempo de resposta do servidor, versão servida |
| Drift de dados | `drift_max_psi`, `drift_mean_psi`, `drift_n_psi_moderate/major`, `drift_score_psi/ks`, `drift_feature_psi/ks` | quanto as entradas e o score se afastaram da janela de treino do campeão |
| Desempenho do modelo (com rótulos, uma semana depois) | `model_auc_pr`, `model_brier`, `model_mean_pred`, `model_observed_rate`, `slice_auc_pr` | queda de AUC-PR, calibração e fatias piores |
| Alertas e diagnóstico | `alert_firing{rule,severity}`, `diagnosis{cause}` | quantos alertas e a causa provável por semana |

Os logs de execução ficam no Airflow (por tarefa e por tentativa) e cada semana gera uma execução no experimento
`production_monitoring` do MLflow com as métricas e os artefatos.

## Retreino e recalibração

O `retraining_pipeline` roda dentro do laço, com atraso de rótulos de **uma semana**: dispara no primeiro alerta forte de
drift de dados (semana 5, rótulos até a semana 4) e depois em alertas de queda de desempenho (semanas 11 e 14, com
carência de três semanas rotuladas entre tentativas). Cada execução constrói dois candidatos: **A**, o campeão
recalibrado, e **B**, a mesma receita reajustada na Referência mais as semanas rotuladas (peso maior para as recentes) e
recalibrada. Os dois passam por um **portão de bootstrap pareado** numa fatia separada de 8 mil linhas (intervalos de
97,5%, com correção para dois candidatos): o ganho de custo (C_FN = 5 × C_FP) precisa ter o intervalo inteiro acima de
zero, sem piora significativa de AUC-PR. Se um passar, o alias `champion` muda e o servidor recarrega; o primeiro
campeão continua pontuando offline como linha de comparação.

Resultado na execução de referência (16 semanas): **o portão rejeitou todos os candidatos nas três tentativas**, então o
campeão original serviu todas as semanas. Depois do drift de covariáveis, recalibrar foi neutro e reajustar com 4 semanas
foi pior (AUC-PR −0,010). Depois de cinco semanas rotuladas de drift de conceito, o candidato B **melhora de fato** nas
semanas 14–16, que nenhum modelo viu: AUC-PR +0,011 (IC 95% +0,002 a +0,020), concentrado no segmento afetado (0,189 →
0,216), e neutro fora dele. Ainda assim o portão não o aprovou: na fatia dele a estimativa foi −0,005 (IC −0,017 a +0,009),
compatível com o ruído. Com cerca de 5 mil clientes por semana e 8% de inadimplência, um ganho de um ponto de AUC-PR não
é certificável por uma validação curta; em produção isso pede um período de sombra mais longo, acumulando rótulos antes
de promover (não implementado aqui). Análise em `notebooks/02_retraining.ipynb`, que cobre as três tentativas (HTML em
`docs/reports/`); decisões, incluindo o que foi ajustado depois de ver resultados (post hoc), na seção 18 de
`docs/decisions.md`. Limitação: o caminho de promoção (mudança de alias, recarga, nova janela de referência de drift) não
foi exercitado no laço real, só em testes unitários; e a ordem de grandeza do ganho (+0,011) é indicativa, não um teste
limpo.

## Calibração dos limiares

Os limiares dos alertas foram calibrados **offline** em simulações com sementes diferentes da oficial: 101 e 202 para
escolher, 303 e 404 para validar uma única vez (`notebooks/04_drift_evaluation.ipynb`, HTML em `docs/reports/`;
decisão #16). Resultado: nenhum falso alarme nas semanas de controle, drift de conceito detectado em 5/5 e 4/5 das
semanas nas sementes de validação e diagnóstico correto em 13/13 e 12/13. A regra de desempenho passou a olhar a
**média de 3 semanas** do AUC-PR com margem de 1,645 erro-padrão, porque uma semana isolada (±0,04) tem o mesmo tamanho
da queda injetada. Limitações assumidas: a semana 8 pode ser detectada com um a dois semanas de atraso, o alerta de
fatia é fraco (a matriz do Grafana faz o ranking) e as quatro sementes vêm do mesmo simulador. A verdade do simulador só
é usada nessa calibração offline; em produção o equivalente seria o histórico de incidentes.

## Orquestração (Airflow)

O Airflow 3 roda no servidor Linux (`docker compose`, perfil `airflow`; interface em `http://localhost:8080` por túnel
SSH). O projeto fica em um ambiente virtual próprio dentro da imagem, para que as versões fixas de dependências do
Airflow não briguem com as nossas (LightGBM, Great Expectations, MLflow, Evidently), e as tarefas dos DAGs são finas:
chamam `python -m credit_drift...`, e a lógica fica em `src/`.

O DAG **`training_pipeline`** (disparo manual) valida os dados de referência contra o contrato completo (e para se algo
for rejeitado), executa os experimentos em etapas no MLflow (engenharia de features, seleção de modelo, ajuste de
hiperparâmetros) e registra o campeão com o limiar de decisão. O parâmetro `quick` faz um teste rápido (10 mil linhas, 3
tentativas, nomes separados). Foi verificado de ponta a ponta no servidor: execução completa em cerca de 2,3 minutos.

## Servidor do modelo

Um serviço FastAPI (`docker compose up -d`, porta local 8010) carrega o campeão do registro do MLflow
(`models:/credit_scoring@champion`) e o serve por HTTP; é ele que o DAG de monitoramento chamará.

| Endpoint | Função |
|---|---|
| `POST /predict` | recebe registros de clientes e devolve, para cada um, a probabilidade de inadimplência e `high_risk` (probabilidade acima do limiar de decisão) |
| `GET /metrics` | métricas Prometheus: requisições por resultado, linhas pontuadas, clientes sinalizados, latência, distribuição das probabilidades, versão do modelo no ar |
| `POST /reload` | consulta o alias `champion` de novo e troca o modelo sem reiniciar (exige o cabeçalho `X-Reload-Token`); se falhar, continua com o modelo anterior |
| `GET /health`, `/ready`, `/model` | vivo, pronto (503 até haver modelo) e metadados do modelo (versão, limiar) |

```bash
curl -s localhost:8010/model
curl -s -X POST localhost:8010/predict -H 'content-type: application/json' -d '{"records": [ {...colunas do cliente...} ]}'
```

- **O contrato de dados vale na entrada:** cada requisição passa pelas mesmas regras do lote semanal; uma
  violação devolve 422 com a coluna, a regra e a contagem de linhas, **nunca os valores** (LGPD).
- **O limiar de decisão** (1/6, custo 5:1) fica registrado como *tag* na versão do modelo no MLflow; ao promover
  um modelo com outro limiar, as decisões mudam sem alterar código. `high_risk` é uma sinalização para
  análise, não uma recusa automática: a decisão final deve permitir revisão humana.
- Medido pelo contêiner: 500 clientes em 178 ms e uma semana inteira (5.000) em 0,68 s; as probabilidades
  coincidem com as do modelo rodando localmente.

## Detecção de drift

Cada semana é validada pelo contrato e, se passar, pontuada pelo modelo campeão e comparada com duas
bases (`uv run python -m credit_drift.monitoring.run_drift`; decisões em `docs/decisions.md`, seção 11):

- a **Referência original** (informativa: quanto o mundo mudou desde o treino);
- a **janela de treino do campeão** (é a que dispara alertas; muda de lugar quando um novo modelo é
  promovido). Para o primeiro campeão as duas são os mesmos dados.

São medidos, para todas as colunas de entrada do modelo (inclusive as que ele mesmo cria, como a razão
parcela/renda) e para o **score**: PSI, KS (numéricas), qui-quadrado (categóricas) e distância de
Jensen-Shannon. O **PSI** decide os alertas (faixas: abaixo de 0,10 sem drift, 0,10–0,25 moderado, acima de
0,25 forte; provisórias até a etapa de calibração), porque com milhares de linhas o p-valor acusa
qualquer diferença, enquanto o PSI mede o tamanho da mudança. Valores ausentes têm faixa própria no PSI,
e colunas numéricas com muitos valores repetidos (61% das rendas estão em 8 números redondos) recebem uma
perturbação de ±2% com semente fixa para que o PSI não oscile conforme o valor cai na borda de uma faixa.

Os relatórios HTML do **Evidently** (PSI, KS e qui-quadrado, 32 variáveis escolhidas por uma regra sobre a
Referência, mais o score) das semanas 1, 5 e 8 estão em `docs/reports/drift/`.

O alerta "drift no conjunto de dados" do Evidently só dispara quando 50% das colunas mudam, o que nunca
acontece com 3 de 32 variáveis; por isso, no teste de PSI ele segue a nossa regra (2 ou mais variáveis com
PSI ≥ 0,10) e, no teste de KS/qui-quadrado, mantém o padrão de 50% de propósito: os p-valores marcam 4 de 32
colunas até na semana de controle e 14 de 32 nas semanas 5 e 8, e por isso servem de evidência, não de alerta.

| Semanas | Variáveis monitoradas com PSI ≥ 0,10 | Leitura |
|---|---|---|
| 1–2 | 0 (máximo 0,012) | controle: os detectores ficam quietos |
| 3 | 0 (renda 0,086) | drift gradual ainda abaixo do limite |
| 4 | 2 (renda 0,156; parcela/renda 0,10) | primeiro alerta |
| 5–8, 10–16 | 3, duas delas fortes (parcela/crédito 0,80–0,94; parcela/renda 0,27–0,34) | o choque abrupto e o novo normal |
| 9 | lote bloqueado pelo contrato | sem pontuação e sem relatório |

Duas lições para o vídeo: (1) a **parcela bruta** sozinha (PSI 0,08) teria escondido o choque; as razões
financeiras o revelam. (2) As semanas 8 e 10–16 (drift de conceito) são **idênticas** às 5–7 para todos os
detectores de distribuição: nenhuma entrada mudou, e só o desempenho, quando os rótulos chegam, mostra a
queda. Já a mudança na mistura de tipos de renda tem PSI baixo (0,031), mas p-valor de qui-quadrado da ordem
de 1e-26: PSI e testes de hipótese respondem a perguntas diferentes.

## Contratos de dados

Toda carga de dados passa por dois validadores com papéis diferentes (regras em
`src/credit_drift/contracts/`, decisões em `docs/decisions.md`):

- **Pandera:** porta de entrada rápida (*fail fast*), só com as regras **bloqueantes**.
- **Great Expectations:** suíte completa, com regras bloqueantes e **avisos**, e geração do
  **Data Docs** (trilha de auditoria).

| Severidade | Regras | Efeito |
|---|---|---|
| Bloqueante | colunas e tipos esperados; `SK_ID_CURR` único e não nulo; renda não nula e > 0; idade entre 18 e 100 anos; `DAYS_EMPLOYED` ≤ 0 ou o código 365243; valores de crédito positivos (parcela ≥ 0) | O lote é rejeitado e não é pontuado |
| Aviso | renda acima de 10 milhões (tolerância de 0,1%); gênero fora de {F, M, XNA}; aumento de mais de 5 pontos na taxa de nulos de uma coluna; categorias nunca vistas; tamanho do lote | Registrado e alertado; o lote segue |

Há dois contratos: o do **lote de pontuação** (sem `TARGET`) e o dos **rótulos**, aplicado quando
eles amadurecem (`TARGET` binário, IDs únicos). Os resultados e o Data Docs registram apenas
**contagens e percentuais**, nunca os valores que violaram a regra (LGPD).

Demonstração (o treino e um lote limpo passam; um lote corrompido, com 12% de rendas nulas,
idades abaixo de 18 anos, 3% de IDs duplicados e uma coluna numérica como texto, é rejeitado
pelos dois validadores):

```bash
uv run python -m credit_drift.contracts.demo
```

O relatório gerado está em `docs/reports/data_docs/index.html`.

## Causalidade

A pergunta da diretoria é "o que causou a degradação?". Com um simulador, temos a verdade do que foi injetado (`drift_spec.json`),
então dá para comparar causa injetada, causa atribuída pelos detectores e causa estimada por contrafactual. A verdade só é usada
**offline** (`notebooks/04_drift_evaluation.ipynb`, HTML em `docs/reports/`); em produção ela não existe, e o equivalente seria
histórico de incidentes.

**1. Injetado contra detectado** (execução de referência, 16 semanas, PSI ≥ 0,10).

| Semanas | O que foi injetado | O que o laço atribuiu |
|---|---|---|
| 1–2 | nada (controle) | nenhum alerta, nenhuma variável sinalizada |
| 3–4 | renda ×0,90 → ×0,86, emprego ×0,92 → ×0,85, mistura de tipos de renda começa | covariáveis; primeiro alerta (moderado) na semana 4, forte na 5 |
| 5 | parcela ×1,15 e mistura completa | covariáveis, `data_drift_major` (parcela/crédito e parcela/renda) |
| 6–7 | o regime da semana 5 continua | "nada": o novo normal parou de mudar |
| 8, 10–16 | drift de conceito num segmento (≈ 26% dos clientes), sem mudar nenhuma entrada | conceito, **mas só quando os rótulos chegam** (semana 8 com atraso: a média de 3 semanas ainda não caíra) |
| 9 | lote corrompido | técnico (contrato bloqueou) |

Precisão de 100% (nenhum falso positivo em 16 semanas) e **recall de ≈ 0,38**: dos oito sinais injetados ou derivados, o PSI nomeia
a razão parcela/crédito, a razão parcela/renda e a renda. O tempo de emprego (PSI ≈ 0,05) e a mistura de tipos de renda (≈ 0,03)
mudaram de verdade, mas ficam abaixo da faixa de 0,10. O diagnóstico semana a semana acerta 14 das 15 semanas avaliáveis.

**2. Contrafactual por reponderação de importância.** Reponderamos a Referência para que suas variáveis se pareçam com as da
produção e medimos o AUC-PR do campeão nela: é o que **as entradas deslocadas, sozinhas**, produziriam.

| | AUC-PR | IC 95% |
|---|---|---|
| Referência (base) | 0,246 | 0,233 a 0,260 |
| Referência reponderada (efeito das entradas) | 0,262 | 0,242 a 0,281 |
| Produção, semanas 5–7 (só covariáveis) | 0,261 | 0,237 a 0,283 |
| Produção, semanas 8 e 10–16 (drift de conceito) | **0,192** | 0,182 a 0,203 |

Nas semanas 5–7 as entradas explicam **tudo** (diferença de 0,002, intervalo que inclui zero): o choque de covariáveis não
custou desempenho, até o elevou levemente. Nas semanas de conceito, as variáveis são idênticas às das semanas 5–7, e ainda assim
o AUC-PR cai **0,070** (IC 95% 0,045 a 0,092) em relação ao que as entradas dariam. Esse resíduo só pode vir da relação entre
variáveis e resposta: **a causa da degradação de desempenho é o drift de conceito**, e é por isso que PSI e KS não o veem.

**3. O que ajudou.** Recalibrar não muda o AUC-PR; reajustar o modelo ajuda só depois de várias semanas rotuladas do regime novo
(seção Retreino e recalibração). Mitigações enquanto os rótulos amadurecem, não implementadas: limiares mais rígidos ou revisão manual
no segmento afetado, reduzir o peso das variáveis afetadas e ponderar dados recentes no próximo retreino.

Limite assumido: a atribuição causal aqui é validada contra um simulador (mesmos tipos de drift, mesmo segmento). Em produção a
causa teria de ser investigada com dados de negócio, não só com métricas.

## Governança e LGPD

> **Ressalva.** O conjunto de dados é público (Home Credit, Kaggle) e não identifica titulares brasileiros: a LGPD não se aplica a
> ele em si. O projeto o trata **como se** fossem dados de clientes de uma fintech brasileira, e este plano descreve o que seria
> feito em produção. É documentação de engenharia, **não parecer jurídico**: os pontos marcados com (*) pedem validação do
> jurídico e do encarregado (DPO) antes de uso real.

### Inventário e classificação dos dados

| Categoria | Colunas (exemplos) | Classificação |
|---|---|---|
| Identificador | `SK_ID_CURR` | chave pseudonimizada; sem nome, CPF, endereço ou contato. Nunca entra no modelo |
| Dados pessoais comuns | idade (`DAYS_BIRTH`), renda, tempo de emprego, escolaridade, ocupação, tipo de organização, dados do crédito pedido | dado pessoal (art. 5º, I) |
| Dado com risco de discriminação | `CODE_GENDER` | **não é entrada do modelo** (`training/features.py`); permanece no dado só para a verificação de viés |
| Scores externos | `EXT_SOURCE_1/2/3` | vêm de terceiros; origem e base legal de uso precisam ser rastreáveis (*) |
| Região e moradia | indicadores agregados do imóvel e da região | dado pessoal de baixo risco |
| Alvo | `TARGET` (inadimplência) | rótulo, chega com atraso |

Nenhuma coluna é dado pessoal **sensível** no sentido do art. 5º, II (origem racial ou étnica, saúde, vida sexual, convicções,
biometria etc.), portanto o art. 11 não se aplica; gênero e idade pedem cuidado com o princípio da não discriminação.

### Como os dados pessoais foram tratados neste projeto

- **Minimização (art. 6º, III).** O ID e o gênero são excluídos das entradas do modelo; só colunas necessárias para o escore
  entram; o ID serve apenas como chave de junção entre as features e os rótulos.
- **Dados fora do git e controle de acesso.** O git guarda só ponteiros do DVC; os dados ficam num bucket privado (bloqueio de
  acesso público, criptografia em repouso, usuário IAM restrito). As interfaces (Airflow, MLflow, Grafana, Prometheus) escutam
  apenas em 127.0.0.1 e são acessadas por túnel SSH; na AWS, grupos de segurança limitados ao IP do operador.
- **Nenhuma linha de cliente em saídas.** Relatórios, notebooks, métricas, MLflow e logs guardam agregados (distribuições,
  métricas, hashes do conjunto de dados), nunca linhas. O servidor valida cada requisição com o contrato e, ao rejeitar, cita
  as regras quebradas e nunca os valores.
- **Quarentena.** O lote rejeitado pelo contrato é guardado à parte (`artifacts/monitoring/quarantine`) para o dono do dado;
  ele **contém linhas de clientes** e por isso tem acesso restrito e retenção curta (tabela abaixo).
- **Rastreabilidade (art. 6º, X e art. 37).** `mlflow.log_input()` liga cada modelo ao hash do conjunto de dados, o registro de
  modelos guarda versões, janela de treino e resultado do portão, o DVC fixa os dados de cada semana e o Great Expectations gera
  as Data Docs (relatório de auditoria).
- **Segredos.** Nenhuma credencial no repositório; quem replica o projeto usa a opção A ou as próprias credenciais.

### Base legal para a decisão de crédito

| Tratamento | Base legal (LGPD, art. 7º) | Observação |
|---|---|---|
| Avaliar o pedido de crédito e pontuar o cliente | **V** (execução de contrato ou procedimentos preliminares a pedido do titular) e **X** (proteção ao crédito) | bases principais; consentimento seria frágil, pois pode ser revogado |
| Monitorar e melhorar o modelo (drift, retreino) | **IX** (legítimo interesse do controlador) | exige teste de balanceamento e minimização; usa os mesmos dados do escore (*) |
| Guardar registros para auditoria e obrigações do setor financeiro | **II** (obrigação legal ou regulatória) | prazos definidos pela regulação aplicável (*) |

**Decisão automatizada (art. 20).** O titular pode pedir a revisão de decisões tomadas unicamente com base em tratamento
automatizado que afetem seus interesses, inclusive de crédito, e o controlador deve informar os critérios e procedimentos (respeitados
os segredos comercial e industrial). Consequências de projeto: o escore é só uma recomendação (`high_risk` quando a probabilidade
atinge 1/6, o limiar de custo documentado); a decisão final deve ter fluxo de revisão por analista; o modelo é versionado e suas
variáveis, explicáveis. Esse fluxo de revisão **não está implementado** aqui (*).

**Princípios aplicados (art. 6º).** Finalidade e adequação: os dados servem só ao escore e ao monitoramento. Necessidade:
minimização acima. Qualidade dos dados: contratos de dados bloqueiam lotes ruins e o monitoramento acompanha a degradação.
Segurança e prevenção: acesso restrito, sem linhas em saídas, alertas de degradação. Não discriminação: verificação de viés
abaixo. Responsabilização: linhagem, registro de modelos e relatórios.

### Direitos dos titulares (art. 18)

Acesso, correção, anonimização, bloqueio ou eliminação, portabilidade e informação sobre compartilhamento dependem do sistema de
origem; este projeto os apoia assim: os dados de treino e de produção são versionados por semana, então um titular que peça
eliminação pode ser retirado da janela do próximo retreino (`retraining/window.py` controla quais semanas entram); modelos já
treinados deixam de conter o dado depois do próximo retreino (*). A revisão de decisões automatizadas está descrita acima.

### Plano de retenção (proposta, a validar (*))

| Artefato | Contém dado pessoal? | Retenção proposta | Motivo |
|---|---|---|---|
| Dados brutos e Referência de treino | sim | enquanto durar a finalidade e a obrigação legal; depois eliminação ou anonimização (art. 15 e 16) | prazos legais do setor definem o teto |
| Lotes semanais de produção (features e rótulos) | sim | até a maior janela de retreino (proposta: 12 meses) | o retreino usa as semanas rotuladas |
| **Lote em quarentena** | sim | 30 dias, só para o dono do dado analisar | evita guardar dado rejeitado sem finalidade |
| Escores por cliente | sim (indiretamente) | prazo de auditoria e de revisão de decisão (art. 20) | permite reconstruir a decisão |
| Métricas, relatórios, notebooks, modelos | não (agregados) | longa, como evidência de auditoria | sem dado pessoal |
| Logs do pipeline | não (sem valores) | 90 dias (proposta) | diagnóstico operacional |

### Mitigação de vieses

`CODE_GENDER` não entra no modelo, mas a verificação de viés exige medir por gênero. Resultado do campeão (v1) na parte de teste
da Referência, no limiar de decisão (1/6), calculado por `credit_drift.monitoring.fairness` (`docs/calibration/fairness.json`):

| Grupo | Linhas | Inadimplência observada | Probabilidade média | Sinalizados como alto risco | Recall | Falso positivo | AUC-PR |
|---|---|---|---|---|---|---|---|
| Mulheres (F) | 27.387 | 7,0% | 7,4% | 9,9% | 35,0% | 8,1% | 0,225 |
| Homens (M) | 14.116 | 10,2% | 9,4% | 16,1% | 43,3% | 13,0% | 0,275 |

Os homens são sinalizados **1,61 vez** mais, e a taxa de falso positivo deles é maior (13,0% contra 8,1%): clientes homens que
**não** inadimplem são barrados com mais frequência. Parte da diferença acompanha a inadimplência observada de cada grupo (10,2%
contra 7,0%), e o modelo subestima um pouco os homens (9,4% previsto contra 10,2% observado), mas a disparidade nas taxas de erro é
um risco de discriminação indireta que **o gênero estar fora do modelo não elimina**: outras colunas podem servir de proxy.
Isso é um achado a levar ao jurídico e ao comitê de risco (*), não algo que este projeto resolve.

O que existe: o gênero é uma fatia do monitoramento semanal (a matriz do Grafana mostra a queda de AUC-PR por gênero a cada semana),
então uma piora desigual aparece. O que **não** foi feito, por exigir decisão de negócio e jurídica: remover variáveis proxy,
reponderar o treino, ou limiares por grupo (que, em crédito, podem ser eles mesmos discriminatórios). Limite assumido: só há
gênero binário no dado, e outras características protegidas (idade, por exemplo) não foram auditadas por grupo.

### Acesso e reprodução dos dados


O projeto trata os dados como se fossem dados reais de clientes de uma fintech, mesmo sendo um
conjunto público, e a estrutura de reprodução segue esse princípio:

- **Minimização e controle de acesso:** os dados brutos nunca entram no git; apenas os arquivos
  de ponteiro do DVC (`*.dvc`). O bucket S3 do autor é privado, com bloqueio de acesso público e
  criptografia em repouso, e é acessado por um usuário IAM restrito a esse bucket.
- **Sem compartilhamento de credenciais:** quem replica o projeto usa a Opção A ou suas próprias
  credenciais; chaves de acesso nunca são compartilhadas, versionadas ou publicadas.
- **Sem redistribuição:** o conjunto de dados pertence à competição do Kaggle e a seus termos de
  uso. O repositório e o bucket não o redistribuem publicamente; cada pessoa o obtém na fonte
  original.
- **Sem dados pessoais nas saídas:** relatórios, notebooks e logs versionados não exibem linhas
  brutas de clientes.

## Vídeo STAR

Link do vídeo (até 5 minutos, método STAR): _a ser inserido depois da gravação_. Roteiro em preparação: **Situation** (o risco da
degradação silenciosa de modelos de crédito), **Task** (interceptar dados ruins, detectar drift, garantir conformidade),
**Action** (Pandera e Great Expectations, Evidently, Prometheus e Grafana, simulação de 16 semanas, governança) e **Result**
(o painel mostrando o drift, os alertas, o diagnóstico e as lições aprendidas).
