# credit-drift

Camada de sustentação e confiabilidade (contratos de dados, detecção de drift, observabilidade
e governança LGPD) em torno de um modelo de credit scoring. Projeto do Tech Challenge — Fase 4.

> Em construção. O plano de implementação está em `docs/build-plan.md` e as decisões de
> arquitetura em `CLAUDE.md`.

## Reproduzindo do zero

Pré-requisitos: git, [uv](https://docs.astral.sh/uv/), Docker Desktop e credenciais AWS de um
usuário IAM com acesso ao bucket de dados.

```bash
# 1. Código e ambiente (o uv instala o Python 3.14 indicado em .python-version)
git clone https://github.com/JoaoFurlan/tech-challenge-04.git
cd tech-challenge-04
uv sync

# 2. Credenciais AWS (nunca versionadas): arquivo ~/.aws/credentials com o perfil [default]
#    [default]
#    aws_access_key_id = ...
#    aws_secret_access_key = ...
#    e ~/.aws/config com: region = us-east-1

# 3. Dados versionados com DVC (remoto: s3://driftsys/dvc-store)
uv run dvc pull

# 4. Serviços locais
docker compose up -d
```

Os passos seguintes (treino, pipelines, dashboards) serão acrescentados aqui conforme forem
implementados. O estado do MLflow (experimentos e registro de modelos) fica em um volume Docker
local e **não** é compartilhado entre máquinas: o treino é reexecutado em cada ambiente, com
sementes fixas, a partir do que está versionado no git e no DVC.

## Dados e privacidade

- Os dados brutos nunca entram no git; apenas os arquivos de ponteiro do DVC (`*.dvc`).
- O bucket S3 é privado, com bloqueio de acesso público e criptografia em repouso, e o acesso é
  feito por um usuário IAM restrito a esse bucket.
