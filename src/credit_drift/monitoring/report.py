"""Evidently HTML report for one week against one baseline: PSI report plus KS (numeric) / chi-square (categorical)."""
from pathlib import Path

import pandas as pd
from evidently import DataDefinition, Dataset, Report
from evidently.presets import DataDriftPreset

REFERENCE_SAMPLE = 20_000  # the baseline is subsampled for the HTML report only; the tables use all of it
# Evidently calls a dataset "drifted" when this share of columns drifted. Its default (0.5) can never fire for a few
# drifted inputs out of 32, so for the PSI preset the banner follows our alert rule: one or more columns at
# PSI >= 0.10. Half a column's share is used as the threshold so that exactly one column trips it whatever the column
# count. The KS / chi-square preset keeps the default: p-values flag about 4 of 33 columns even in the clean control
# week, which is exactly why they are evidence and not the alert signal.
EXPLAINER = """
<div style="font-family:Georgia,serif;max-width:1100px;margin:24px auto;padding:20px 28px;border:2px solid #1f4e79;
border-radius:8px;background:#f7fafc;color:#1a1a1a;line-height:1.5">
<h2 style="margin:0 0 8px">Como ler este relatório: duas réguas diferentes sobre os mesmos dados</h2>
<p>As mesmas colunas desta semana são comparadas com a janela de treino do campeão por <b>dois testes</b>. Cada um tem
o seu próprio quadro "Dataset Drift", com a sua própria regra, por isso os números diferem. Não é contradição.</p>
<p><b>1. Primeiro quadro: PSI (Population Stability Index), o sinal de alerta.</b> Divide cada variável em faixas
(10 faixas com a mesma quantidade de linhas na base) e compara a <i>fração de linhas em cada faixa</i> hoje com a da
base: PSI = soma de (fração atual − fração base) × ln(fração atual ÷ fração base). Mede <i>o quanto</i> a distribuição
andou, não só se andou: abaixo de 0,10 é estável, de 0,10 a 0,25 é mudança moderada, acima de 0,25 é grande. Não
cresce só porque a amostra é grande. A coluna conta como "drifted" com PSI ≥ 0,10, e o quadro acusa drift quando
<b>pelo menos uma</b> coluna passa disso (é a regra de alerta do projeto).</p>
<p><b>2. Segundo quadro: KS e χ² (testes de hipótese), a evidência complementar.</b> Para colunas numéricas, o
<b>KS (Kolmogorov-Smirnov)</b> desenha a curva acumulada da base e a da semana e mede a <i>maior distância vertical</i>
entre elas (0 a 1); para categóricas, o <b>χ² (qui-quadrado)</b> compara as contagens por categoria. Ambos devolvem um
<i>p-valor</i>: abaixo de 0,05 a coluna é marcada "drifted". Eles respondem "a diferença existe?", não "é grande?":
com milhares de linhas até diferenças minúsculas dão p &lt; 0,05, e com 33 colunas algumas são marcadas por acaso mesmo
numa semana limpa. Por isso este quadro só acusa drift quando metade das colunas é marcada (padrão do Evidently).</p>
<p><b>Resumo:</b> PSI mede o tamanho da mudança e decide o alerta; KS e χ² dizem se a mudança é estatisticamente
detectável e servem de segunda opinião. Na semana 1 (controle), o PSI não marca nenhuma coluna, enquanto KS e χ² marcam
algumas por acaso: é o comportamento esperado.</p>
</div>
"""


def build_report(current: pd.DataFrame, baseline: pd.DataFrame, kinds: dict[str, str], title: str, path: Path) -> Path:
    """`current` and `baseline` hold the monitored columns plus SCORE; `kinds` maps column -> numeric/categorical."""
    columns = [c for c in current.columns if c in kinds or c == "SCORE"]
    definition = DataDefinition(
        numerical_columns=[c for c in columns if c == "SCORE" or kinds[c] == "numeric"],
        categorical_columns=[c for c in columns if c != "SCORE" and kinds[c] == "categorical"])
    reference = baseline[columns].sample(min(len(baseline), REFERENCE_SAMPLE), random_state=0)
    report = Report([DataDriftPreset(method="psi", drift_share=0.5 / len(columns)),
                    DataDriftPreset(num_method="ks", cat_method="chisquare")])
    snapshot = report.run(Dataset.from_pandas(current[columns], data_definition=definition),
                          Dataset.from_pandas(reference, data_definition=definition), name=title)
    path.parent.mkdir(parents=True, exist_ok=True)
    snapshot.save_html(str(path))
    html = path.read_text(encoding="utf-8")
    path.write_text(html.replace("<body>", "<body>" + EXPLAINER, 1), encoding="utf-8")
    return path
