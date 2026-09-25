# Datathon Fase 5 — plataforma de experimentação adaptativa para contato bancário

Mariana de Freitas, RM363866

## O problema

Bancos e fintechs que fazem campanhas de contato ativo (ligação, SMS, push) costumam escolher um canal e um período fixos para todo mundo, ou testam duas opções num A/B que só conclui depois que a campanha inteira já rodou. As duas abordagens desperdiçam tráfego: a regra fixa nunca aprende com o que já aconteceu, e o A/B gasta metade dos contatos numa opção pior até ter significância estatística suficiente para decidir.

Um multi-armed bandit trata cada decisão de contato como uma escolha entre "braços" (aqui, canal e dia da semana) que vai sendo ajustada em tempo real conforme a campanha avança, sem precisar esperar o fim de um teste estático para aproveitar o que já foi aprendido. Foi essa adaptação contínua que me levou a usar Thompson Sampling em vez de regra fixa ou A/B tradicional.

## A base

Uso a base *bank-marketing*, publicada no Kaggle por henriqueyamahata (https://www.kaggle.com/datasets/henriqueyamahata/bank-marketing), arquivo `bank-additional-full.csv`. São 41.188 linhas, cada uma um contato de telemarketing de um banco, com o target `y` indicando se o cliente fez ou não um depósito a prazo (~11,3% de conversão). Versão: 6.47.

Removi a coluna `duration` logo no início do tratamento: ela guarda a duração da ligação, e só se sabe esse valor depois que a ligação termina. Usá-la seria vazamento temporal, um dado que não existe no momento em que a decisão de contato precisa ser tomada (o próprio enunciado do desafio cita esse cuidado).

## Como rodar local

1. Baixe o `bank-additional-full.csv` na página do Kaggle acima e salve em `data/bank-additional-full.csv` (a pasta `data/` está no `.gitignore`, o CSV não vem versionado no repositório).
2. Crie e ative um ambiente virtual: `python -m venv .venv && source .venv/bin/activate`
3. Instale as dependências: `pip install -r requirements.txt`
4. Rode os notebooks em ordem: primeiro `notebooks/01_eda.ipynb`, depois `notebooks/02_bandit.ipynb` (Restart & Run All em cada um).
5. Suba a API: `uvicorn api.main:app --reload`, depois teste com:
   curl -X POST http://localhost:8000/recomendar
     -H "Content-Type: application/json"
     -d '{"age": 24, "job": "management", "education": "university.degree", "previous": 0}'
6. Veja os experimentos: `mlflow ui --backend-store-uri sqlite:///mlflow.db`, depois abra `http://localhost:5000`.

## Modelagem

**Braços.** A base tem um único tratamento real (a ligação) e nenhum catálogo de ofertas para escolher, então defini o braço como a combinação de `contact` (canal: cellular ou telephone) e `day_of_week`, totalizando 10 braços. A "oferta" desse projeto é a abordagem de contato: por qual canal e em que dia abordar o cliente.

Preciso ser honesta sobre uma limitação de `contact`: não é uma alavanca livre do banco nessa base. A campanha migrou de telephone para cellular ao longo do tempo, e esse período coincide com uma mudança no cenário macroeconômico (`euribor3m` alto no período telephone-pesado, justamente quando a conversão cai). Ou seja, `contact` é em parte um proxy de calendário e de conjuntura econômica, não só de canal. O bandit aprende a preferir cellular, e está em parte capturando essa diferença de período, não só de canal puro.

**Segmento.** Uso tercil de idade (cortes em 34 e 44 anos, calculados por quantil sobre a própria base), não `poutcome`. A categoria `poutcome=success` tem só 1.373 linhas, que viram poucas dezenas de eventos aceitos por braço no replay, ruído demais para confiar. Segmentar por idade é uma decisão de personalização que documento aqui de propósito: o enunciado proíbe usar gênero, raça, renda ou patrimônio para personalizar, e embora idade não esteja nessa lista, trato como uma decisão que merece justificativa por escrito, não silêncio. Também não uso `marital`, `housing`, `loan` nem `default` como segmento ou entrada da API.

**Baseline.** Uso a política histórica, que sorteia o braço segundo a distribuição empírica de braços no log (não "sempre o braço mais frequente"). Testei isso na EDA antes de decidir: o braço mais frequente da base (`cellular_thu`, 15,37% de conversão histórica) não é estatisticamente diferente do braço de maior conversão (`cellular_tue`, 15,76%; diferença de 0,4pp contra um erro padrão combinado de ~0,7pp). Se eu tivesse usado "sempre o mais frequente" como baseline, ele já converteria perto do teto, sobrando quase nenhuma margem para qualquer política vencer fora do ruído. A política histórica converge para ~11,3% (a conversão média real da base), abrindo uma margem de fato mensurável para o Thompson Sampling fechar.

**Thompson Sampling.** Mantenho um par (alpha, beta) por combinação de segmento e braço, começando em Beta(1,1) (prior uniforme, sem viés inicial declarado). A cada decisão, amostro uma Beta de cada braço do segmento e escolho o argmax; ao final de cada evento aceito, reforço alpha (sucesso) ou beta (fracasso) do braço escolhido.

**Avaliação.** Como não dá para saber o que teria acontecido se o banco tivesse escolhido outro braço, avalio por replay (rejection sampling): percorro o log embaralhado, e só aceito e aprendo com uma linha quando o braço escolhido pela política bate com o braço que o banco realmente usou naquela linha; senão, descarto. Rodo 20 seeds por política e trunco todas as séries no menor número de eventos aceitos entre políticas e seeds, para não favorecer quem aceita mais eventos no gráfico. Uma ressalva importante: esse replay é não enviesado sob aleatorização uniforme de tratamento, o que não é o caso aqui. O banco escolheu canal em função do cliente e do período, então o subconjunto aceito de cada braço é uma subpopulação diferente da base inteira. O que meço é associação observada no log, não o efeito causal de aplicar essa política a todo mundo.

## Resultados


O Thompson Sampling termina 1,4pp acima do baseline, sem sobreposição de banda no trecho final, e captura cerca de 41% da distância entre o baseline e o teto in-sample. Em termos de negócio: 1,4pp sobre os 41.188 contatos dessa base equivalem a aproximadamente 588 depósitos adicionais, se esse ganho se sustentasse ao aplicar a política em toda a campanha (extrapolação otimista, ver Limitações).

A tabela de pulls por (segmento, braço) mostra um padrão consistente: em todo segmento, telephone recebe muito menos exploração que cellular, porque o modelo separa canal com confiança. O braço "vencedor" dentro do cellular, porém, muda entre segmentos, o que interpreto como ruído estatístico e não como preferência real de dia por faixa etária: a amostra por (segmento, braço) é grande o suficiente para distinguir a diferença entre cellular e telephone (~9-13pp), mas não para distinguir 1-2pp entre dias específicos.

## Os 5 casos de teste


Em todos os cinco perfis, o canal recomendado é cellular, nunca telephone, o que se sustenta no que a base mostra. O dia recomendado varia entre chamadas porque a API amostra da Beta a cada requisição (é a parte adaptativa do bandit em runtime), então não interpreto o dia específico como um padrão de comportamento daquele perfil — inclusive `job` e `education` não entram em nenhum momento no cálculo da recomendação, só `age`.

- Clientes 0 e 1 (jovem, prob ~0,151): o canal (cellular) faz sentido — jovens convertem mais por ali em todos os tercis. Os dois caírem no mesmo dia (segunda) reforça que a recomendação é por segmento, não por perfil individual; não atribuiria isso a hábito de trabalho, porque job não é usado no cálculo e a diferença entre dias dentro do cellular é pequena diante do tamanho da amostra.
- Cliente 2 (meio, prob 0,136): o canal faz sentido, mas repare que a probabilidade é a mais baixa dos cinco — bate com o achado da EDA de que o tercil "meio" (35-44 anos) converte sistematicamente menos que jovem e senior, em qualquer braço.
- Clientes 3 e 4 (senior, prob 0,162 e 0,141): mesmo raciocínio dos jovens — canal correto, dia não é um padrão de comportamento sênior, é a amostra da Beta daquela chamada específica. Os dois ficarem em dias diferentes é esperado, não uma falha.

## Governança de dados

**Base legal:** uso de dados históricos e públicos de uma campanha de contato já encerrada, sem identificação de clientes reais — a base do Kaggle já vem anonimizada. **Finalidade:** personalizar canal e dia de contato para melhorar a taxa de conversão de uma campanha de captação, não para decisões de crédito, preço ou qualquer avaliação sobre o cliente. **Minimização:** removo `duration` e `default` do tratamento, e não uso `marital`, `housing`, `loan` nem `poutcome` como segmento ou entrada da API, além de nunca usar variáveis sensíveis (gênero, raça, renda, patrimônio) para personalizar, como o enunciado exige. **Retenção:** a base usada é um arquivo público estático, não há coleta contínua de novos dados pessoais nesse projeto. **Humano no loop:** a API devolve uma recomendação de canal e dia, não uma ação automática — quem decide se liga, quando e o que fala continua sendo um operador humano.

## Arquitetura AWS

Para rodar isso em produção, guardaria a base tratada em S3 e publicaria a API como um serviço em Fargate atrás de um Application Load Balancer, com a imagem da API construída e versionada no ECR. O treino do bandit (o notebook `02_bandit.ipynb`) rodaria como um job agendado, também containerizado, disparado por um evento no EventBridge, com o resultado (`politica_ts.json`) publicado de volta num bucket S3 que a API lê na inicialização.

Para o tracking de experimentos, MLflow rodando numa instância EC2 pequena (ou o equivalente gerenciado, SageMaker Experiments, para evitar operar a instância) cobre o "uso básico" que esse projeto pede. CloudWatch cobre métricas de infraestrutura (latência e erro da API) e alarmes simples — por exemplo, se a taxa de conversão observada em produção cair abaixo do baseline por um período, o que sinalizaria um problema de dado ou de modelo antes que vire prejuízo real.

## Limitações

- O replay mede associação observada no log, não o efeito causal de aplicar a política a todos os contatos; o banco escolheu canal em função do cliente e do período, não aleatoriamente.
- `contact` está confundido com o período da campanha e com o cenário macroeconômico (`euribor3m`); o ganho medido é em parte um efeito de tempo, não só de canal.
- O volume de eventos aceitos por (segmento, braço) é suficiente para separar cellular de telephone (diferença grande, ~9-13pp), mas não para separar dias específicos dentro do cellular (diferença pequena, dentro do ruído amostral).
- O teto usado como referência é in-sample (escolhido nos mesmos dados em que o regret é medido), então é uma estimativa otimista, não um limite real de melhoria possível.
- A política não usa contexto contínuo; ela reage por segmento fixo, não por cliente individual além da faixa etária.