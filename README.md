📊 Risco e Retorno

Um dashboard interativo em Dash (Plotly) para simulação e análise de estratégias de alocação de ativos.
Permite comparar diferentes abordagens de investimento com foco em Carteira Eficiente, Paridade de Risco e Benchmark (CDI).

🚀 Funcionalidades

Seleção livre de ativos (via dropdown) para compor a carteira.

Configuração de parâmetros:
-Aporte inicial (R$)
-Aporte mensal (R$)
-Período de simulação (datas inicial e final)

Gráfico de desempenho: evolução histórica das três estratégias.

Alocação de ativos (Pizza): distribuição percentual dos pesos da Paridade de Risco, exibindo também o valor em R$ no hover.

Contribuição de risco (Barras): participação percentual de risco de cada ativo na Paridade de Risco.

Métricas resumidas:
-Total investido
-Retorno acumulado (%)
-Retorno anualizado (%)
-Valor final (R$) de cada estratégia

📂 Estrutura do Projeto


🛠️ Instalação

1 - Clone o repositório

    git clone https://github.com/<seu-usuario>/risco-e-retorno.git
    cd risco-e-retorno

2 - Crie e ative um ambiente virtual 

    python -m venv env
    source env/bin/activate   # Linux/Mac
    env\Scripts\activate      # Windows

3 - Instale as dependências

    pip install -r requirements.txt

▶️ Como Rodar

Execute no terminal:

    python dashboard_integrado.py
    
O Dash iniciará o servidor local.
Abra no navegador: 👉 http://127.0.0.1:8050
