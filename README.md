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
