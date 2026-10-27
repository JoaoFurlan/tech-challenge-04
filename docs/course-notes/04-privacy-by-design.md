# Privacy by Design e by Default — resumo do módulo (Aulas 1 a 5)

> Fontes: `Aula 1 - Fundamentos Privacy by Design e by Default.pdf` …
> `Aula 5 - Privacidade_Novas_Tecnologias_Desafios_PbD.pdf`

---

## 0. Fio condutor do módulo

```
Aula 1  O PRINCÍPIO   → os 7 princípios de Cavoukian; PbD vs. PbDf; casos práticos
Aula 2  O CONTEXTO    → Data Driven Society: benefícios × dilemas éticos
Aula 3  A LEI         → GDPR e LGPD: princípios, bases legais, direitos, sanções
Aula 4  A FERRAMENTA  → dados sintéticos vs. anonimização/pseudonimização/DP
Aula 5  O FUTURO      → IA, IoT, blockchain, biometria, RV/RA, quântica, PETs
```

---

## Aula 1 — Fundamentos de Privacy by Design e by Default

### Contexto: a sociedade da vigilância

Organizações usam dados para personalizar serviços, prever comportamentos e impulsionar
marketing; governos empregam tecnologias de vigilância. **Cada interação digital pode ser
rastreada, armazenada e analisada.**

Exemplos citados: o escândalo da **Cambridge Analytica** (dados de milhões de usuários do
Facebook usados para manipulação eleitoral); o sistema de **crédito social chinês**; o aumento
do **reconhecimento facial em espaços públicos no Brasil**.

> A verdadeira proteção da privacidade depende **não apenas de leis**, mas da conscientização
> e ação coletiva de indivíduos, empresas e governos.

### Impacto das violações

| Para os indivíduos | Para as organizações |
|---|---|
| prejuízos financeiros (fraude bancária, roubo de identidade) | prejuízos financeiros e **multas regulatórias** |
| danos emocionais e psicológicos (saúde, fotos pessoais expostas) | **danos irreparáveis à reputação** |
| **discriminação** quando dados sensíveis (orientação sexual, histórico médico) são usados indevidamente em seleção ou crédito | dificuldade de atrair clientes e parceiros; **perda de confiança** |

**Casos:** **Netshoes (2018)** — vazamento de nomes, endereços e dados financeiros de milhões
de clientes. **Equifax (2017)** — **147 milhões** de pessoas expostas, multa de **US$ 575
milhões** e perda de confiança de consumidores e investidores.

### Por que a privacidade importa

Direito fundamental garantido pela **Declaração Universal dos Direitos Humanos (art. 12)**,
essencial para **liberdade, autonomia e dignidade humana**. É pela privacidade que as pessoas
exercem a individualidade sem medo de monitoramento ou julgamento, e é ela que **protege
grupos vulneráveis** de discriminação por etnia, orientação sexual ou histórico médico.

### Os 7 princípios de Privacy by Design (Ann Cavoukian, 1990s)

| # | Princípio | O que significa |
|---|---|---|
| 1 | **Proativo, não reativo; preventivo, não corretivo** | antecipar e prevenir problemas, em vez de reagir depois do fato |
| 2 | **Privacidade como padrão (Privacy by Default)** | a configuração padrão protege, **sem exigir intervenção do usuário** |
| 3 | **Privacidade incorporada ao design** | componente **essencial** do design, não um add-on |
| 4 | **Funcionalidade total — soma positiva, não soma zero** | privacidade **sem** comprometer funcionalidade |
| 5 | **Segurança de ponta a ponta** | proteger o **ciclo de vida completo** dos dados, da coleta à eliminação |
| 6 | **Visibilidade e transparência** | processos de tratamento visíveis e verificáveis |
| 7 | **Respeito pela privacidade do usuário** | o usuário no centro do design |

**Exemplo canônico:** a **"Localização Aproximada"** do iOS 14 — o usuário compartilha
localização aproximada em vez de exata, reduzindo a coleta de dados sensíveis ao mínimo
necessário.

### Privacy by Default (PbDf) em detalhe

> As configurações **mais protetivas** devem ser o **padrão**, sem que o usuário precise
> configurá-las manualmente.

**Por que importa:** protege justamente **quem não tem conhecimento técnico** nem
familiaridade com configurações de privacidade.

Exemplos: Chrome bloqueando cookies de terceiros no modo anônimo por padrão; Telegram com
autodestruição de mensagens ativada por padrão em chats secretos; **Apple exigindo
autorização explícita para rastreamento de apps no iOS (ATT)**; Instagram tornando contas de
menores privadas por padrão; Facebook limitando o alcance padrão de publicações a "Amigos".

**Minimização da coleta** é o princípio fundamental do PbDf: coletar apenas o estritamente
necessário para a finalidade específica. **A coleta excessiva aumenta os riscos de violação e
dificulta a gestão de segurança.**

### PbD na prática — cinco frentes

| Frente | Técnica | Exemplos |
|---|---|---|
| **Criptografia** | ponta a ponta (E2EE); TLS em trânsito | WhatsApp, Signal (protocolo **open-source**, permitindo auditorias externas), Telegram; bancos e e-commerces com TLS |
| **Anonimização** | remove/altera informações identificadoras, tornando-as não associáveis a um indivíduo | dados anonimizados de celulares para monitorar distanciamento social na COVID-19 |
| **Pseudonimização** | substitui identificadores diretos (nome, CPF) por **códigos únicos**, mantendo a possibilidade de **reidentificação autorizada** | CRMs de e-commerce que pseudonimizam dados de compra |
| **Ferramentas de controle** | bloqueio de rastreadores por padrão | Firefox (**Enhanced Tracking Protection**), Brave, **DuckDuckGo** (projetado desde o início sem rastrear consultas) |
| **IoT** | firewalls para dispositivos conectados, criptografia em trânsito e repouso, MFA | Nest (Google) com configurações detalhadas; Amazon Echo passou a permitir que o usuário **exclua gravações** após críticas |

Dado citado: navegadores com bloqueio automático de rastreadores reduziram em até **35%** o
volume de dados coletados sem consentimento.

### Benefícios

**Para as organizações:** redução de riscos e penalidades (multas, litígios); mitigação de
riscos legais e operacionais; cultura de conformidade com GDPR/LGPD; **aumento da confiança e
da reputação**; estímulo à inovação.

> Consumidores são cada vez mais criteriosos, priorizando empresas comprometidas com
> privacidade. **A privacidade vira vantagem competitiva** — o Signal conquistou milhões de
> usuários exatamente assim, especialmente após as denúncias de Edward Snowden sobre a coleta
> massiva da NSA.

### Cases do mercado

| Caso | Relevância PbD/PbDf |
|---|---|
| **Apple App Tracking Transparency (ATT)** | aplicativos **não podem rastrear sem consentimento explícito**; privacidade integrada à experiência central do produto |
| **Brave Browser** | bloqueio de rastreadores e anúncios invasivos **por padrão**, sem configuração adicional |
| **Zoom** | após críticas na pandemia, adicionou E2EE e controles aprimorados — privacidade integrada ao design de forma proativa |

---

## Aula 2 — Data Driven Society

### A escala

A cada **minuto**: 188 milhões de e-mails, 41,6 milhões de mensagens no WhatsApp, 3,8 milhões
de buscas no Google.

### Os três pilares

```
1. COLETA          → IoT e plataformas digitais coletam massivamente
2. ANÁLISE         → Big Data e IA processam e encontram padrões
3. TOMADA DE DECISÃO → dados informam negócios, políticas públicas e o cotidiano
```

### Dados em ação — por setor

| Setor | Aplicações | Exemplos citados |
|---|---|---|
| **Saúde** | diagnósticos precisos, medicina personalizada, monitoramento remoto | IA do MIT detectando câncer de mama com **94% de precisão**; 23andMe (dados genômicos); Apple Watch alertando sobre fibrilação atrial |
| **Educação** | ensino adaptativo, análise preditiva de evasão | Khan Academy, Duolingo, Coursera; **Georgia State University reduziu a evasão em 20%** |
| **Negócios** | marketing direcionado, otimização de processos, novos produtos | Netflix (recomendação); FedEx (sensores na frota preveem falhas e otimizam rotas); Nike |
| **Meio ambiente** | monitoramento ambiental, modelagem climática | sensores de poluição em São Paulo; supercomputadores da NASA |
| **Segurança pública** | prevenção criminal, resposta a emergências | **PredPol** em Los Angeles; alertas sísmicos via celular no Japão |

### Os quatro dilemas éticos

**1. Privacidade — um direito ameaçado**
Vazamentos (**Equifax, 2017 — 143 milhões**), vigilância em massa (câmeras com IA na China,
usadas segundo críticos para suprimir dissidências), uso antiético (**Cambridge Analytica**
influenciando as eleições de 2016 nos EUA).

**2. Vieses algorítmicos — perpetuando inequidades**
Algoritmos treinados em dados históricos **refletem e amplificam preconceitos existentes**.
- **Amazon (2018)** — descontinuou o sistema de recrutamento por IA ao descobrir que favorecia
  homens para cargos de tecnologia, pois aprendeu padrões a partir de currículos majoritariamente
  masculinos.
- **COMPAS (EUA)** — classificava desproporcionalmente réus negros como mais propensos a
  reincidir, levantando preocupações sobre equidade no sistema de justiça.

**3. Exclusão digital — a nova desigualdade**
**2,7 bilhões de pessoas** ainda sem acesso à internet. Mesmo entre conectados, a falta de
**letramento digital** limita o aproveitamento: segundo a UNESCO, 60% da população global tem
conectividade, mas muitos não compreendem questões básicas de segurança digital e proteção de
dados.

**4. Desinformação — dados contra a verdade**
Fake news minam a confiança na mídia e no processo democrático (eleições brasileiras de 2018 no
WhatsApp; Twitter sinalizando conteúdo falso desde 2020).

### Como enfrentar

| Frente | O que fazer |
|---|---|
| **Regulamentação** | GDPR e LGPD oferecem o arcabouço — mas a **aplicação prática ainda enfrenta desafios** |
| **Educação e transparência** | educar usuários sobre o uso de dados; adotar **PbD e PbDf** |
| **Governança algorítmica** | garantir que algoritmos sejam **auditáveis e livres de vieses** |

### As quatro habilidades essenciais

1. **Pensamento crítico** — questionar origem, validade e intenção dos dados; analisar
   metodologias e contexto de coleta. *Dados podem ser manipulados para servir a interesses
   específicos.*
2. **Alfabetização de dados** — interpretar gráficos, tabelas, mapas, infográficos e extrair
   insights.
3. **Competências digitais** — usar ferramentas para pesquisar, organizar, analisar e
   comunicar dados. *Sua ausência exacerba a exclusão digital.*
4. **Ética e responsabilidade** — respeitar privacidade e direitos ao usar dados. *O uso
   irresponsável leva a violações, discriminação e perda de confiança.*

### Cases

**Waze** (dados colaborativos em tempo real para otimizar rotas); **Netflix** (dados de
visualização → recomendação, campanhas direcionadas e **decisão sobre qual conteúdo original
produzir**); **ANA — Agência Nacional de Águas** (nível de reservatórios, vazão, precipitação
e uso da água → prever e prevenir crises hídricas).

> "Os dados não são apenas números: eles contam histórias e representam vidas."

---

## Aula 3 — Fundamentos da LGPD e do GDPR

### GDPR — alcance extraterritorial

Aplica-se a **todas as empresas que coletam ou processam dados de cidadãos da UE,
independentemente de onde estejam localizadas**.

> *Exemplo:* uma startup brasileira de delivery que expande para Portugal **precisa se adequar
> ao GDPR** para processar dados de clientes portugueses.

### Conceitos essenciais do GDPR

| Conceito | Definição |
|---|---|
| **Dados pessoais** | qualquer informação relativa a pessoa singular **identificada ou identificável**, direta ou indiretamente (nome, nº de identificação, **dados de localização, identificador on-line**, ou fatores da identidade física, fisiológica, genética, mental, econômica, cultural ou social) |
| **Dados genéticos** | características genéticas herdadas ou adquiridas que fornecem informação única sobre fisiologia ou saúde |
| **Dados biométricos** | resultantes de tratamento técnico de características físicas, fisiológicas ou comportamentais que **permitem ou confirmam a identificação única** (imagens faciais, dados datiloscópicos) |
| **Dados de saúde** | relacionados à saúde física ou mental, incluindo a prestação de serviços de saúde |
| **Controlador** | determina **as finalidades e os meios** do tratamento |
| **Processador** | processa dados **em nome do** controlador |

### Os 7 princípios do GDPR

| Princípio | Significado | Exemplo |
|---|---|---|
| **Licitude, justiça e transparência** | base legal + informação clara e acessível | consentimento explícito e informado para marketing personalizado |
| **Limitação da finalidade** | finalidades determinadas, explícitas e legítimas; sem uso incompatível | rede social não pode usar dados de **autenticação** para publicidade direcionada sem consentimento |
| **Minimização de dados** | "menos é mais" | num cadastro de e-commerce, pedir só nome, endereço e e-mail — não estado civil |
| **Exatidão** | dados precisos e atualizados; titular pode corrigir | área de conta para atualizar contato e preferências |
| **Limitação do armazenamento** | guardar apenas pelo tempo necessário | **políticas de retenção**; excluir dados de clientes inativos |
| **Integridade e confidencialidade** | proteger contra acesso não autorizado | criptografia, controle de acesso, firewalls, treinamento |
| **Responsabilização (accountability)** | **demonstrar** conformidade | **a palavra-chave é documentação**: registros das atividades de tratamento, auditorias, políticas |

### As 6 bases legais do GDPR

1. **Consentimento** — livre, específico, informado e inequívoco.
   ⚠️ Requisitos: demonstrado por **declaração ou ação afirmativa clara**; deve ser **tão fácil
   retirar quanto dar**; **não pode ser base legal em relações de poder desigual**
   (ex.: empregador-empregado).
2. **Execução de um contrato** — ex.: loja on-line coleta o endereço para entregar.
3. **Cumprimento de obrigação legal** — ex.: dados de funcionários para previdência social.
4. **Proteção de interesses vitais** — ex.: hospital trata dados de paciente inconsciente.
5. **Tarefa de interesse público / autoridade pública** — ex.: prefeitura e planejamento urbano.
6. **Interesses legítimos** do controlador ou de terceiros.
   ⚠️ Requisitos: exige **teste de balanceamento**; **não pode ser usado por autoridades
   públicas** no desempenho de suas funções; atenção redobrada quando o titular é **criança**.

### Direitos dos titulares (GDPR)

Acesso · Retificação · **Eliminação ("direito ao esquecimento")** · Limitação do tratamento ·
**Portabilidade** (receber em formato estruturado e transferir) · Oposição (marketing direto,
decisões automatizadas) · **Direito de não ser submetido a decisões unicamente automatizadas,
incluindo definição de perfis**.

### LGPD (Lei nº 13.709/2018)

Inspirada no GDPR; aplicável a **qualquer organização que trate dados pessoais no Brasil**,
independentemente de onde esteja sediada.

**Conceitos:** **dado pessoal** (relacionado a pessoa natural identificada ou identificável);
**dado pessoal sensível** (origem racial ou étnica, convicção religiosa, opinião política,
filiação sindical ou a organização religiosa/filosófica/política, dado de saúde ou vida sexual,
dado genético ou biométrico **quando vinculado a pessoa natural**); **controlador** (a quem
competem as decisões); **operador** (trata em nome do controlador).

### Os 10 princípios da LGPD

| | Princípio |
|---|---|
| 1 | **Finalidade** — propósitos legítimos, específicos, explícitos e informados |
| 2 | **Adequação** — tratamento compatível com as finalidades informadas |
| 3 | **Necessidade** — só o estritamente necessário |
| 4 | **Livre acesso** — acesso facilitado às informações sobre o tratamento |
| 5 | **Qualidade dos dados** — exatos, claros, relevantes e atualizados |
| 6 | **Transparência** — informações claras e acessíveis |
| 7 | **Segurança** — medidas contra acesso não autorizado e incidentes |
| 8 | **Prevenção** — prevenir a ocorrência de danos |
| 9 | **Não discriminação** — vedado uso para fins discriminatórios |
| 10 | **Responsabilização e prestação de contas** — demonstrar conformidade |

### As 10 hipóteses de tratamento da LGPD

Consentimento · cumprimento de obrigação legal ou regulatória · execução de políticas públicas ·
**estudos por órgão de pesquisa (garantida a anonimização sempre que possível)** · execução de
contrato ou diligências preliminares · exercício regular de direitos em processo judicial,
administrativo ou arbitral · proteção da vida ou incolumidade física · **tutela da saúde** ·
**legítimo interesse** do controlador (desde que não prevaleçam direitos fundamentais do
titular) · **proteção ao crédito**.

**Dados pessoais sensíveis** exigem consentimento **específico e destacado**. Sem consentimento,
apenas quando **indispensável** para: obrigação legal/regulatória; políticas públicas; estudos
por órgão de pesquisa (com anonimização sempre que possível); exercício regular de direitos;
proteção da vida; tutela da saúde por profissionais/serviços de saúde ou autoridade sanitária;
**garantia da prevenção à fraude e à segurança do titular** em identificação e autenticação.

### Direitos dos titulares (LGPD)

Confirmar a existência de tratamento · acessar os dados · **corrigir** incompletos/incorretos/
desatualizados · solicitar **anonimização, bloqueio ou eliminação** de dados desnecessários,
excessivos ou tratados em desconformidade · **portabilidade** · eliminação dos dados tratados
com base no consentimento · ser informado sobre **com quem o controlador compartilhou** ·
ser informado sobre a possibilidade de **negar o consentimento e suas consequências** ·
**revogar o consentimento a qualquer momento**.

### O custo da não conformidade

| | **GDPR** | **LGPD** |
|---|---|---|
| Multa máxima | **€ 20 milhões ou 4% do faturamento global anual** (o que for maior) | **até 2% do faturamento, limitada a R$ 50 milhões por infração** |

**Outras sanções administrativas da LGPD:** advertência com prazo de adequação · **multa
diária** por descumprimento de ordem da ANPD · **publicização da infração** · bloqueio dos
dados · eliminação dos dados · **suspensão parcial do funcionamento do banco de dados** ·
**proibição parcial ou total do exercício de atividades de tratamento**.

Além disso: danos de reputação e **ações judiciais individuais**.

### Cases

| Caso | Desfecho |
|---|---|
| **Telekall Infoservice (jul/2023)** | **primeira multa da ANPD** — R$ 14.400 por comercializar dados sem consentimento e **não possuir encarregado (DPO)** |
| **Enel — STJ (dez/2024)** | responsabilizada por vazamento de dados **não sensíveis** mesmo alegando ataque hacker; o STJ reforçou que **segurança de dados é obrigação legal, independentemente da origem do incidente** |
| **Meta/Facebook (mai/2023)** | **€ 1,2 bilhão** pela Comissão de Proteção de Dados da Irlanda — transferência de dados de usuários europeus para os EUA sem garantias adequadas |

---

## Aula 4 — Privacidade na Era da Ciência de Dados: Dados Sintéticos

### Big Data — os cinco Vs

**Volume** (petabytes/exabytes) · **Velocidade** (geração e processamento, crucial para tempo
real) · **Variedade** (estruturados, não estruturados, semiestruturados) · **Veracidade**
(confiabilidade e qualidade) · **Valor** (capacidade de extrair insights significativos).

### Os dilemas

**Uso indevido / vazamentos:** **Yahoo (2013)** — ~**3 bilhões** de contas comprometidas;
**Uber (2016)** — 57 milhões de motoristas e passageiros, com a empresa **tentando encobrir** e
pagando resgate aos hackers; **Equifax (2017)** — 147 milhões.

**O limite da anonimização:** métodos de anonimização **frequentemente não bastam**. Técnicas
de **reidentificação** combinam dados anônimos com outras fontes para revelar identidades
(estudo do NIST). A crescente sofisticação dos algoritmos **aumenta esse risco**.

**Falta de transparência e controle:** consumidores desconhecem como seus dados são coletados e
compartilhados, o que dificulta o consentimento informado.

### Dados sintéticos — o que são

> Dados **gerados artificialmente** que **preservam as propriedades estatísticas** dos dados
> originais, mas **não contêm informações que possam identificar indivíduos**.

Criados por modelos estatísticos ou algoritmos de ML que aprendem as características dos dados
reais e geram novos conjuntos que as imitam.

### Comparação com as outras técnicas de proteção

| Técnica | Como funciona | Vulnerabilidade |
|---|---|---|
| **Anonimização** | remove informações identificadoras | **vulnerável à reidentificação** se cruzada com outras fontes |
| **Pseudonimização** | substitui identificadores por pseudônimos | reduz identificação direta, mas **ainda vulnerável** |
| **Privacidade diferencial** | adiciona **ruído matemático** garantindo que informações individuais não possam ser inferidas com precisão | eficaz, mas **complexa de implementar** |
| **Dados sintéticos** | gera dados novos que imitam a distribuição | **elimina o risco de reidentificação** (não há dado real) |

### Vantagens

1. **Privacidade** — não contêm informações sensíveis reais; permitem compartilhamento sem
   comprometer indivíduos.
2. **Flexibilidade** — podem ser gerados para aumentar diversidade, simular cenários
   específicos ou **corrigir vieses nos dados originais** (ex.: criar datasets balanceados para
   treino).
3. **Escalabilidade** — geração em grande escala, mais rápida e eficiente que coletar dados reais.
4. **Custo** — mais baratos que coletar e rotular dados reais; **democratizam o acesso a dados**
   para pesquisa, viabilizando estudos de organizações menores e pesquisadores independentes.

### Métodos de geração

| Método | Como funciona | Quando usar |
|---|---|---|
| **GANs** (Goodfellow et al., 2014) | **gerador** cria dados sintéticos, **discriminador** avalia se são reais ou falsos; o gerador melhora até "enganar" o discriminador | dados altamente realistas — imagens, textos (ex.: rostos fictícios para treinar reconhecimento facial sem dados reais) |
| **Modelos estatísticos** | geram dados a partir das distribuições observadas nos dados originais | pesquisa acadêmica, modelagem de cenários |
| **Modelagem baseada em regras** | segue regras predefinidas (limites, padrões, relações entre variáveis) | simular cenários específicos de um setor — ex.: dados financeiros fictícios para **simulações de risco bancário** |

### Aplicações

- **Treinar modelos de IA** sem expor dados reais (ex.: reconhecimento de voz)
- **Testar software** em diferentes cenários, acelerando desenvolvimento e reduzindo custos
- **Pesquisa em áreas sensíveis** (saúde, finanças) sem comprometer participantes
- **Carros autônomos** — simular cenários de trânsito sem risco a vidas humanas
- **American Express e JP Morgan Chase** — treinar modelos de detecção de fraude
- **Departamento de Defesa dos EUA** — simulações e treinamentos

### Desafios

| Desafio | Detalhe |
|---|---|
| **Qualidade / fidelidade** | podem **não capturar todas as nuances** dos dados originais. Ex.: difícil replicar completamente padrões de comportamento humano como fraudes complexas — modelos treinados podem ser menos eficazes na detecção real |
| **Complexidade técnica** | exige alta expertise em IA/ML e **recursos computacionais significativos** (treinar GANs demanda tempo, conhecimento e infraestrutura robusta) |
| **Aceitação regulatória** | GDPR e LGPD podem exigir **auditorias detalhadas** provando que os dados sintéticos são seguros e mantêm a integridade dos originais — camada adicional de complexidade e custo. Ex.: um banco usando dados sintéticos para risco de crédito pode precisar de auditoria regulatória |

### Ética e transparência

> **Ética** = garantir que criação, compartilhamento e aplicação respeitem direitos individuais,
> evitem discriminação e promovam o bem coletivo.
> **Transparência** = informar claramente **quando** utilizam esses dados, **como** foram
> gerados e **quais** suas aplicações.

⚠️ **O lado sombrio:** a mesma tecnologia produz **deepfakes** e manipulações enganosas —
vídeos falsos de figuras públicas para influenciar eleições ou difamar, e desinformação gerada
por IA.

### Mercado

**Amazon** usa dados sintéticos para aprimorar o sistema de linguagem da **Alexa**.
**Roche** usa dados médicos sintéticos em pesquisas clínicas, assegurando conformidade com
regulamentações de privacidade. Em **janeiro de 2025**, Elon Musk afirmou que a indústria de IA
**atingiu um limite no uso de dados reais** para treinamento, apontando os dados sintéticos
como o futuro.

---

## Aula 5 — O Futuro da Privacidade: Novas Tecnologias e Desafios para o PbD

### PbD como instrumento de conformidade

| Exigência legal (GDPR/LGPD) | Como o PbD ajuda a cumprir |
|---|---|
| **Minimização de dados (necessidade)** | integra a minimização **ao design** do sistema. Ex.: app de seguro-saúde coleta apenas dados do plano, **sem exigir histórico médico detalhado** (que só seria necessário em caso de sinistro) |
| **Transparência e consentimento** | consentimento explícito e visível **embutido no design**, atendendo ao requisito de accountability. Ex.: Netflix permitindo ajuste de preferências e explicando como os dados de visualização personalizam recomendações |
| **Auditoria e documentação** | processos documentados no ciclo de vida dos dados, incluindo **avaliações de impacto à privacidade (DPIA)** — relatório compartilhável com reguladores em caso de fiscalização |

### Os sete desafios tecnológicos ao PbD

**1. Inteligência Artificial e Machine Learning**
- **Coleta massiva** — modelos exigem grandes volumes, incluindo dados sensíveis (biométricos,
  localização, histórico de navegação).
- **Perpetuação de vieses** — algoritmos treinados em dados históricos que refletem
  desigualdades reproduzem essas discriminações em decisões automatizadas.
- **Caixa-preta** — redes neurais profundas dificultam entender como as decisões são tomadas,
  impedindo que usuários **questionem decisões automatizadas que os afetam**.
  *Exemplo: IA na justiça criminal prevendo reincidência sem explicar como chegou à previsão.*

**2. Internet das Coisas**
- **Coleta excessiva** — dispositivos frequentemente coletam mais do que o necessário:
  localização, hábitos de consumo, preferências, **até conversas**.
- **Segurança frágil** — senhas padrão fracas, falta de criptografia, ausência de atualizações
  regulares.
- **Monitoramento contínuo** — câmeras conectadas, assistentes virtuais e smartphones rastreando
  localização, movimentos e interações.

**3. Blockchain e criptomoedas**
- **Imutabilidade × direito ao esquecimento** — uma vez registrados, os dados **não podem ser
  apagados ou modificados**, conflitando diretamente com o GDPR.
- **Transparência pública** — em blockchains públicas **todas as transações são visíveis**,
  revelando hábitos de compra, finanças e relações.
- **Pseudonimato não é anonimato** — técnicas avançadas de análise podem **identificar usuários
  e rastrear suas transações**.

**4. Reconhecimento facial**
- **Vigilância em massa** sem consentimento em espaços públicos, coletando dados biométricos.
- **Discriminação** — taxas de erro mais altas para mulheres e minorias raciais.
- **Opacidade** — muitas aplicações não informam como os dados são coletados e usados,
  violando o princípio da transparência.

**5. Realidade Virtual e Aumentada**
- Coletam dados **altamente sensíveis**: movimentos corporais, expressões faciais, localização.
- Permitem criar **perfis comportamentais** que identificam usuários mesmo em ambientes virtuais
  e podem ser usados para **manipular decisões e influenciar emoções**.
- Usuários **desconhecem o alcance** dessa coleta e não têm controle sobre ela.

**6. Computação quântica**
- Potencial de **quebrar os algoritmos de criptografia atuais** que protegem senhas, informações
  financeiras e dados médicos.
- Muitas organizações **ainda não estão preparadas** para migrar para **criptografia
  pós-quântica** (padronização em curso pelo NIST) — deixando dados vulneráveis a ataques
  futuros.

**7. PETs (Privacy-Enhancing Technologies)**
Permitem processar e analisar dados protegendo a privacidade, mas enfrentam barreiras:
- **Alto custo e complexidade técnica** — pequenas e médias empresas não têm recursos nem
  expertise.
- Técnicas como **differential privacy** e **aprendizado federado** exigem conhecimento avançado
  em criptografia, estatística e ML; **a falta de profissionais qualificados dificulta a adoção**.

### Síntese

> Empresas e pessoas desenvolvedoras devem **integrar práticas éticas e medidas de segurança
> desde a concepção**, garantindo que o avanço tecnológico seja compatível com os direitos
> fundamentais à privacidade.
>
> O PbD é mais do que um princípio técnico — é **um compromisso ético e estratégico que coloca a
> privacidade como diferencial competitivo** no cenário global.

### Mercado e tendências

- **Fórum Econômico Mundial:** +78 milhões de empregos líquidos até 2030 — **170 milhões
  criados** (big data, IA, desenvolvimento de software) contra **92 milhões obsoletos**
  (operadores de entrada de dados, caixas bancários).
- **Brasil:** a **Lei Geral do Esporte** torna o reconhecimento facial obrigatório em estádios
  com capacidade acima de 20 mil pessoas a partir de 2025 (controle de acesso, prevenção de
  fraude, combate à revenda ilegal) — um caso concreto de biometria em massa com implicações
  diretas de privacidade.
- **Tendências 2025:** integração entre inteligência humana e artificial, biotecnologia,
  tecnologias climáticas, cibersegurança aprimorada e **progresso na computação quântica**.

---

## Conexões com os outros módulos

| Tema desta trilha | Onde reaparece |
|---|---|
| Minimização e anonimização de dados | logging de inferência com **hash do input em vez do valor** — [`02-monitoramento-pipelines-infra.md`](02-monitoramento-pipelines-infra.md), Aula 5 |
| Auditoria, lineage e accountability | **Unity Catalog, Model Registry e Model Cards** — [`03-validacao-dados-qualidade.md`](03-validacao-dados-qualidade.md), Aula 8 |
| Representatividade e viés | *Gender Shades*, NIST AI RMF, fairness tests no CI — [`03-validacao-dados-qualidade.md`](03-validacao-dados-qualidade.md), Aulas 1 e 8 |
| Dados sintéticos / GANs | **GAIN** para imputação de valores ausentes — [`03-validacao-dados-qualidade.md`](03-validacao-dados-qualidade.md), Aulas 2 e 3 |
| Imputação federada e PPRL | preservação de privacidade em dados distribuídos — [`03-validacao-dados-qualidade.md`](03-validacao-dados-qualidade.md), Aulas 2 e 3 |
| EU AI Act e auditabilidade | gatilho regulatório para monitoramento contínuo — [`02-monitoramento-pipelines-infra.md`](02-monitoramento-pipelines-infra.md), Aulas 4 e 8 |

---

## Referências recorrentes do módulo

- **Cavoukian, A. (2009)** — *Privacy by Design: The 7 Foundational Principles*. Information and
  Privacy Commissioner of Ontario.
- **BRASIL. Lei nº 13.709/2018** — Lei Geral de Proteção de Dados Pessoais (LGPD).
- **UNIÃO EUROPEIA** — Regulamento Geral de Proteção de Dados (GDPR), Regulamento 2016/679.
- **Zuboff, S. (2019)** — *The Age of Surveillance Capitalism*. PublicAffairs.
- **Solove, D. (2008)** — *The Digital Person: Technology and Privacy in the Information Age*.
  NYU Press.
- **Westin, A. (1967)** — *Privacy and Freedom*. Atheneum.
- **Schneier, B. (2015)** — *Data and Goliath*. W.W. Norton.
- **Dwork, C. (2006)** — *Differential Privacy*. ICALP 2006.
- **Goodfellow, I. et al. (2014)** — *Generative Adversarial Nets*. NeurIPS.
- **Narayanan, A. et al. (2016)** — *Bitcoin and Cryptocurrency Technologies*. Princeton.
- **Sicari, S. et al. (2015)** — *Security, Privacy and Trust in Internet of Things*. Computer Networks 76.
- **NIST** — *Post-Quantum Cryptography Standardization*.
- **Cadwalladr, C.; Graham-Harrison, E. (2018)** — reportagem do Guardian sobre a Cambridge Analytica.
- **ANPD** — Autoridade Nacional de Proteção de Dados (gov.br/anpd).
