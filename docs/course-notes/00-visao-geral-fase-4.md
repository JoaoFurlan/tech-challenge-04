# Fase 4 — Model Monitoring and Security (visão geral)

> Fonte: `MLET - Capitulo de Projeto - Fase 4.pdf` (Welcome to Machine Learning Engineering)

## Onde essa fase se encaixa

A Fase 3 cobriu engenharia de deploy, CI/CD, orquestração e infraestrutura de nuvem.
A Fase 4 trata do que vem **depois do deploy**: sustentação, observabilidade contínua e
governança em produção.

> "Colocar um modelo no ar é apenas o primeiro passo; mantê-lo confiável, preciso, seguro e
> gerando impacto real de negócio em um mundo dinâmico é o que define um(a) Engenheiro(a) de
> Machine Learning sênior."

## Os quatro blocos de conteúdo

| Bloco | Pergunta central | Resumo |
|---|---|---|
| **Data Drift** | O mundo mudou? | Data/Concept/Label drift, testes estatísticos (KS, PSI, Wasserstein, KL/JS), detecção de mudança de regime (ADWIN, CUSUM), drift em embeddings, mitigação | [`01-data-drift.md`](01-data-drift.md) |
| **Monitoramento de Pipelines e Infraestrutura** | O sistema está de pé? | Observabilidade (logs/métricas/traces), Airflow + Datadog, Prometheus/Grafana, MLflow, cloud, AIOps e retreino automático | [`02-monitoramento-pipelines-infra.md`](02-monitoramento-pipelines-infra.md) |
| **Validação de Dados e Qualidade** | O dado é confiável? | Dimensões de qualidade, duplicatas/nulos, contratos e schema, outliers, frameworks (GX, Pandera, Deequ, TFDV), validação contínua | [`03-validacao-dados-qualidade.md`](03-validacao-dados-qualidade.md) |
| **Privacy by Design e by Default** | Temos direito de usar esse dado? | PbD (7 princípios), LGPD/GDPR, dados sintéticos, DP, PETs e o futuro da privacidade | [`04-privacy-by-design.md`](04-privacy-by-design.md) |

## O que o projeto prático da fase pede

Construir **uma stack de observabilidade, qualidade de dados, monitoramento de drift e
governança** para garantir que modelos em produção permaneçam **robustos, auditáveis e
resilientes ao longo do tempo**.

Checklist implícito no capítulo de projeto:

- [ ] Diagnosticar e monitorar **Data Drift, Concept Drift e Label Drift**
- [ ] Aplicar testes estatísticos e métricas avançadas (KS, PSI, Wasserstein, KL/JS, ADWIN, CUSUM)
- [ ] Cobrir dados **tabulares** e **embeddings** (texto, imagem, áudio)
- [ ] Pipelines contínuos de detecção + mitigação com ferramentas de mercado
      (Evidently AI, WhyLabs, Arize, Great Expectations)
- [ ] **Contratos de dados** rigorosos: interceptar duplicatas, anomalias e nulos críticos
      antes de chegarem ao modelo
- [ ] Observabilidade de ML: logs estruturados, telemetria, métricas de inferência,
      dashboards analíticos e operacionais em **Prometheus + Grafana**
- [ ] Monitoramento de pipelines (**Apache Airflow**) e infraestrutura (**Datadog**),
      com alertas preventivos de falha de ingestão, gargalo de transformação e
      degradação de recursos
- [ ] Governança e compliance (**LGPD/GDPR**): catalogação, *data lineage*, versionamento
      estrito, auditoria contínua e proteção à privacidade

## Frase-síntese da fase

> "Prepare-se para dominar a disciplina que separa os modelos que quebram em silêncio
> daqueles que sustentam operações críticas de alta escala."
