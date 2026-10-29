# credit-drift

Camada de sustentação e confiabilidade (contratos de dados, detecção de drift, observabilidade
e governança LGPD) em torno de um modelo de credit scoring. Projeto do Tech Challenge — Fase 4.

> Em construção. O plano de implementação está em `docs/build-plan.md` e as decisões de
> arquitetura em `CLAUDE.md`.

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

Os dados de produção (100 mil linhas separadas do treino) são cortados em **13 semanas** de 5.000
clientes, sem sobreposição, e um estado de "mundo" é aplicado a cada semana
(`uv run python -m credit_drift.drift.simulate`; decisões em `docs/decisions.md`, seção 10). O tempo é
sintético: os dados não têm datas.

| Semanas | O que acontece |
|---|---|
| 1–2 | controle: nenhum drift (os detectores **não** devem disparar) |
| 3–4 | drift gradual de covariáveis: renda ×0,90 → ×0,86 e tempo de emprego ×0,92 → ×0,85; a mistura de tipos de renda começa a mudar |
| 5 | choque abrupto: parcela ×1,15 (a razão parcela/renda dispara) e mistura completa (pensionistas de 18% para 12%) |
| 6–7 | novo normal: o regime da semana 5 continua |
| 8, 10–13 | **drift de conceito**: para clientes até 45 anos com renda abaixo da mediana (≈26%), a relação de `EXT_SOURCE_2` com a inadimplência é invertida e reduzida à metade; nenhuma variável muda |
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

## Dados e privacidade (LGPD)

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
