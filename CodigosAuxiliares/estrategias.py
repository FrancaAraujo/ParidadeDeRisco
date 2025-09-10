import pandas as pd
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import warnings
import plotly.graph_objects as go

warnings.filterwarnings("ignore", category=RuntimeWarning, module="scipy.optimize._slsqp_py")

def selecionar_ativos(data, limite=100):

    """
    Exibe os ativos disponíveis e permite que o usuário selecione até um número limitado de ativos.
    
    Parâmetros:
    - data: DataFrame com os dados dos ativos, deve conter uma coluna 'Data'.
    - limite: número máximo de ativos a serem selecionados (padrão: 10).
    
    Retorna:
    - Lista com os nomes dos ativos selecionados.
    """

    all_assets = [col for col in data.columns if col != 'Data']
    print("Ativos disponíveis:")
    for i, asset in enumerate(all_assets):
        print(f"{i + 1}: {asset}")

    while True:
        selected_indices = input(f"Digite os números dos ativos que deseja usar (máx {limite}), separados por vírgula: ")
        selected_indices = selected_indices.split(',')

        try:
            selected_indices = [int(i.strip()) - 1 for i in selected_indices]
            if len(selected_indices) > limite:
                print(f"Você selecionou mais de {limite} ativos. Por favor, selecione no máximo {limite}.")
                continue
            if any(i < 0 or i >= len(all_assets) for i in selected_indices):
                print("Algum índice está fora do intervalo. Tente novamente.")
                continue
            break
        except ValueError:
            print("Entrada inválida. Digite os números separados por vírgula.")

    asset_columns = [all_assets[i] for i in selected_indices]
    print(f"\nAtivos selecionados: {asset_columns}")
    return asset_columns

# Definir o caminho para os arquivos CSV
#file_path = 'arquivos/Dados_Ativos_B32.csv'
file_path = 'arquivos/Dados_Ativos_B3_close.csv'
#file_path = 'arquivos/Dados_Ativos_B3_AdjClose.csv'
#file_path = 'arquivos/IvvbBova.csv'
cdi_file_path = 'arquivos/cdi_data_total.csv'

# Carregar os dados do CDI e ATIVOS do arquivo CSV completo
cdi_data_total = pd.read_csv(cdi_file_path)
print(cdi_data_total.columns)

data = pd.read_csv(file_path)

# Carregar os dados históricos dos ativos
data['Data'] = pd.to_datetime(data['Data'], format='%d/%m/%Y %H:%M:%S')

# Converter a coluna 'data' para datetime e adicionar o horário "16:56:00"
cdi_data_total['data'] = pd.to_datetime(cdi_data_total['data'], dayfirst=True) + pd.Timedelta(hours=16, minutes=56)

# Remove o caractere de separador de milhar e converte as colunas de ativos para float
asset_columns = selecionar_ativos(data)

cols_validas = []
for col in asset_columns:
    print(col)
    try:
        data[col] = data[col].str.replace(',', '.').astype(float)
        cols_validas.append(col)
    except:
        data.drop(columns=col, inplace=True)

asset_columns = cols_validas



# Inicializa variáveis
aporte_inicial = 1000
aporte_mensal = 400
total_investido = 0
quantities = {asset: 0 for asset in asset_columns}

# ----------------------------
# Inicialização de variáveis para o portfólio otimizado
# ----------------------------

n_assets = len(asset_columns)
x_initial = np.ones(n_assets) / n_assets
b = np.ones(n_assets) / n_assets
total_invested_portfolio = 0
quantities_portfolio = {asset: 0 for asset in asset_columns}
# Para cada ativo, manter quantidades separadas
quantities_individual = {asset: 0 for asset in asset_columns}
total_invested_individual = {asset: 0 for asset in asset_columns}

# Listas para armazenar os valores dos portfólios de cada estratégia
portfolio_values_eficiente = []
portfolio_values_paridade = []
individual_portfolio_values = {asset: [] for asset in asset_columns}
monthly_risk_parity_values = []  # Lista para armazenar os valores mensais da Paridade de Risco
cdi_values = []
# ----------------------------
# Funções Auxiliares
# ----------------------------

def filter_cdi_data(month, df_cdi_total, df_assets):
    one_year_before = month - pd.DateOffset(years=1)

    # Usar apenas df_cdi_total corretamente (sem cdi_data_total)
    cdi_filtered = df_cdi_total[(df_cdi_total['data'] >= one_year_before) & (df_cdi_total['data'] < month)].copy()

    # Normalizar datas (evita SettingWithCopyWarning)
    cdi_filtered['data'] = cdi_filtered['data'].dt.normalize()
    # Filtrar pelas datas dos ativos

    cdi_filtered = cdi_filtered[cdi_filtered['data'].isin(df_assets['Data'].dt.normalize())]

    return cdi_filtered

def get_yearly_data(month, data):
    one_year_before = month - pd.DateOffset(years=1)
    yearly_data = data[(data['Data'] >= one_year_before) & (data['Data'] < month)]
    return yearly_data

def calculate_investment(total_portfolio_value, weights, current_prices, asset_columns):
    quantities2 = {}
    for i, asset in enumerate(asset_columns):
        current_price = current_prices[asset]
        amount_to_invest_in_asset = total_portfolio_value * weights[i]
        if current_price > 0:
            quantities2[asset] = amount_to_invest_in_asset / current_price
        else:
            quantities2[asset] = 0
    return quantities2


# ------------------------
# Funções de Verificação
# ------------------------

def calculate_risk_contributions(x, cov_matrix):
    sigma_x = np.dot(cov_matrix, x)
    total_risk = np.sqrt(np.dot(x.T, sigma_x))
    marginal_risk = sigma_x / total_risk
    risk_contributions = x * marginal_risk
    return risk_contributions, total_risk

def check_equal_risk_contributions(risk_contributions):
    total_risk = np.sum(risk_contributions)
    n_assets = len(risk_contributions)
    target_risk = total_risk / n_assets
    return np.all(np.isclose(risk_contributions, target_risk, atol=1e-3))

# ----------------------------
# Funções de otimização
# ----------------------------
volatilidades_mensais = {}

def calculate_variance(x, cov_matrix):
    return np.dot(x.T, np.dot(cov_matrix, x))

def objective(x, cov_matrix, b):
    variance = calculate_variance(x, cov_matrix)
    sqrt_variance = np.sqrt(variance)
    w = x / sqrt_variance
    quadratic_term = 0.5 * np.dot(w.T, b / w)
    log_term = np.dot(b.T, np.log(w))
    return quadratic_term - log_term

def weight_sum_constraint(x):
    return np.sum(x) - 1.0

def weight_bounds(n_assets):
    return [(1e-19, 1) for _ in range(n_assets)]

def solve_system_8(cov_matrix, b, initial_x):
    n_assets = len(b)
    constraints = {'type': 'eq', 'fun': weight_sum_constraint}
    bounds = weight_bounds(n_assets)
    solution = minimize(objective, initial_x, args=(cov_matrix, b),
                        method='SLSQP', bounds=bounds, constraints=constraints, options={'maxiter': 1000, 'ftol': 1e-9})
    if not solution.success:
        raise ValueError(f"Falha na otimização: {solution.message}")
    return solution.x, solution.fun

# ----------------------------
# Estratégias de Investimento
# ----------------------------

def EstrategiaEficiente(aporte_inicial, aporte_mensal, total_investido, portfolio_values_eficiente):
    
    global quantities  # Para manter as quantidades atualizadas fora da função

    for month in pd.date_range(start=start_2023, end=end_2023, freq='MS'):
        # Definir a data de investimento como o primeiro dia útil do mês em um horário específico
        investment_date = month + pd.offsets.BMonthBegin(0)
        investment_datetime = investment_date.replace(hour=16, minute=56, second=0)
        one_year_before = investment_date - pd.DateOffset(years=1)

        yearly_data = data[(data['Data'] >= one_year_before) & (data['Data'] < investment_date)]
        yearly_dataaux = data[(data['Data'] >= one_year_before) & (data['Data'] <= investment_datetime)]
        # Filtrar os dados de CDI para o período relevante

        cdidiario_filtered = filter_cdi_data(month, cdi_data_total, yearly_data)
        # Cálculo de retornos diários para todos os ativos
        returns = yearly_data[asset_columns].pct_change().dropna()

        if returns.empty:
            print(f"Erro: Não há retornos suficientes para calcular a matriz de covariância em {month.strftime('%m/%Y')}. Pulando este mês.")
            continue

        # Tentativa inicial com todos os ativos
        current_assets = asset_columns.copy()
        success = False
        max_attempts = len(asset_columns)  # Número máximo de tentativas (uma para cada ativo que pode ser removido)

        min_vol_threshold = 13 # 5%

        for attempt in range(max_attempts):
            current_returns = returns[current_assets]
            
            # Calcular volatilidades
            volatilities = current_returns.std() * (252 ** 0.5) * 100
            volatilidades_mensais[pd.to_datetime(month).normalize()] = volatilities.to_dict()
            # Print da volatilidade em percentual
            print(f"Volatilidade dos ativos em {month.strftime('%m/%Y')}:")
            for asset, vol in volatilities.items():
                print(f" - {asset}: {vol:.2f}%")

            # Eliminar ativos com volatilidade abaixo do limite
            low_vol_assets = volatilities[volatilities < min_vol_threshold]
            if not low_vol_assets.empty:
                for asset in low_vol_assets.index:
                    print(f"Removendo ativo {asset} por baixa volatilidade ({volatilities[asset]:.2f}%) em {month.strftime('%m/%Y')}")
                    current_assets.remove(asset)

                # Se sobrar 1 ou nenhum ativo, não faz sentido continuar
                if len(current_assets) <= 1:
                    success = False
                    break

            current_returns = returns[current_assets]
            cov_matrix = current_returns.cov()

            try:
                L = np.linalg.cholesky(cov_matrix)
                L_inv = np.linalg.inv(L)
                cov_matrix_inv = L_inv.T @ L_inv
                
                success = True
                break
            except np.linalg.LinAlgError:
                # Recalcular volatilidades após possível remoção
                volatilities = current_returns.std()

                # Encontrar ativo com menor volatilidade
                min_vol_asset = volatilities.idxmin()
                print(f"Removendo ativo {min_vol_asset} por problema na matriz (vol {volatilities[min_vol_asset]:.2f}%) em {month.strftime('%m/%Y')}")
                current_assets.remove(min_vol_asset)

                if len(current_assets) <= 1:
                    print(f"Todos os ativos foram removidos por baixa volatilidade em {month.strftime('%m/%Y')}. Pulando este mês.")
                    success = False
                    break

                if len(current_assets) == 2:
                    success = True
                    break

        if not success:
            if len(current_assets) == 1:
                only_asset = current_assets[0]
                print(f"Aportando 100% em {only_asset} por ser o único ativo restante em {month.strftime('%m/%Y')}.")

                if month == pd.to_datetime(start_2023):
                    amount_to_invest = aporte_inicial
                else:
                    amount_to_invest = aporte_mensal
                total_investido += amount_to_invest
                print(f"Total investido {total_investido}")

                # Calcular o preço atual do ativo na data de investimento
                current_price = yearly_dataaux.iloc[-1][only_asset]
                new_quantity = amount_to_invest / current_price if current_price > 0 else 0
                quantities[only_asset] += new_quantity

                # Definir o início e fim do mês atual
                start_of_month = month
                end_of_month = month + pd.offsets.MonthEnd(0)

                # Filtrar dados diários para o mês atual
                daily_data = data[(data['Data'] >= start_of_month) & (data['Data'] <= end_of_month)].reset_index(drop=True)
                
                if not daily_data.empty:
                    # Substituir preços faltantes com o último preço disponível para evitar NaNs
                    daily_prices = daily_data[asset_columns].ffill()
                    
                    # --- Avaliação diária do portfólio ---
                    portfolio_daily_values = daily_prices.multiply(list(quantities.values()), axis=1).sum(axis=1)
                    portfolio_values_eficiente.extend(zip(daily_data['Data'], portfolio_daily_values))

                continue  # Pula o restante da lógica para esse mês

            else:
                print(f"Não foi possível encontrar uma combinação de ativos com matriz inversível em {month.strftime('%m/%Y')}. Pulando este mês.")
                continue
        

        # Se chegamos aqui, temos uma matriz inversível com os ativos em current_assets
        # Cálculo das médias dos retornos para os ativos atuais
        expected_returns = current_returns.mean().values

        # Vetor elementar e transposto para os ativos atuais
        e_transposto = np.ones(len(current_assets)).reshape(1, -1)

        # Calcular a média do retorno do CDI filtrado
        CDIRet = cdidiario_filtered['valor'].mean()
        # Cálculo de Rf_e para os ativos atuais (usando o retorno do CDI)
        Rf_e = np.full((len(current_assets), 1), CDIRet / 100)

        # Cálculo de M - Rf_e
        M_Rfe = expected_returns.reshape(-1, 1) - Rf_e

        # Multiplicações necessárias para obter as porcentagens
        V_M_Rfe = np.matmul(cov_matrix_inv, M_Rfe).flatten()
        V_eT = np.matmul(e_transposto, cov_matrix_inv)

        # Certifique-se de que Div seja um escalar
        Div = np.matmul(V_eT, M_Rfe)[0, 0]

        if Div == 0:
            print(f"Erro: Divisor zero em {month.strftime('%m/%Y')}. Pulando este mês.")
            continue
        
        # Cálculo das porcentagens para os ativos atuais
        percentages = V_M_Rfe / Div
        # Criar um dicionário de porcentagens para todos os ativos originais
        full_percentages = {asset: 0.0 for asset in asset_columns}
        for asset, pct in zip(current_assets, percentages):
            full_percentages[asset] = pct
        # Determinar o valor de investimento para este mês
        if month == pd.to_datetime(start_2023):
            amount_to_invest = aporte_inicial
        else:
            amount_to_invest = aporte_mensal
        total_investido += amount_to_invest
        print(f"Total investido {total_investido}")
        
        # Calcular os preços atuais na data de investimento
        current_prices = yearly_dataaux.iloc[-1][asset_columns].to_dict()

        # Calcular as quantidades a serem investidas em cada ativo com base nas porcentagens otimizadas
        new_quantities = calculate_investment(amount_to_invest, list(full_percentages.values()), current_prices, asset_columns)

        # Atualizar as quantidades do portfólio
        for asset in asset_columns:
            print(f'{asset} {quantities[asset]} {new_quantities[asset]}')
            quantities[asset] += new_quantities[asset]

        # Definir o início e fim do mês atual
        start_of_month = month
        end_of_month = month + pd.offsets.MonthEnd(0)

        # Filtrar dados diários para o mês atual
        daily_data = data[(data['Data'] >= start_of_month) & (data['Data'] <= end_of_month)].reset_index(drop=True)
        
        if not daily_data.empty:
            # Substituir preços faltantes com o último preço disponível para evitar NaNs
            daily_prices = daily_data[asset_columns].ffill()
            
            # --- Avaliação diária do portfólio otimizado ---
            portfolio_daily_values = daily_prices.multiply(list(quantities.values()), axis=1).sum(axis=1)
            portfolio_values_eficiente.extend(zip(daily_data['Data'], portfolio_daily_values))
            #print(f"total investido {total_investido}")

def ParidadeDeRisco(aporte_inicial, aporte_mensal, n_assets, x_initial, b, portfolio_values_paridade, individual_portfolio_values):
    global quantities_portfolio, quantities_individual, total_invested_portfolio, monthly_risk_parity_values
    global asset_columns

    for month in pd.date_range(start=start_2023, end=end_2023, freq='MS'):
        #print(f"Paridade {month.strftime('%m/%Y')}")

        investment_date = month + pd.offsets.BMonthBegin(0)
        investment_datetime = investment_date.replace(hour=16, minute=56, second=0)
        one_year_before = investment_date - pd.DateOffset(years=1)

        yearly_data = data[(data['Data'] >= one_year_before) & (data['Data'] < investment_date)]
        yearly_dataaux = data[(data['Data'] >= one_year_before) & (data['Data'] <= investment_datetime)]

        if len(yearly_data) < 2:
            print(f"Erro: Não há dados suficientes para {month.strftime('%m/%Y')}. Pulando.")
            continue

        returns = yearly_data[asset_columns].pct_change().dropna()
        current_assets = asset_columns.copy()
        success = False
        max_attempts = len(asset_columns)
        min_vol_threshold = 13

        for attempt in range(max_attempts):
            current_returns = returns[current_assets]
            volatilities = current_returns.std() * (252 ** 0.5) * 100

            #print(f"Volatilidade dos ativos em {month.strftime('%m/%Y')}:")
            #for asset, vol in volatilities.items():
            #    print(f" - {asset}: {vol:.2f}%")

            low_vol_assets = volatilities[volatilities < min_vol_threshold]
            if not low_vol_assets.empty:
                for asset in low_vol_assets.index:
                    #print(f"Removendo ativo {asset} por baixa volatilidade ({volatilities[asset]:.2f}%)")
                    current_assets.remove(asset)
                if len(current_assets) <= 1:
                    success = False
                    break

            current_returns = returns[current_assets]
            cov_matrix = current_returns.cov() * 1e-2

            try:
                L = np.linalg.cholesky(cov_matrix)
                L_inv = np.linalg.inv(L)
                cov_matrix_inv = L_inv.T @ L_inv
                success = True
                break
            except np.linalg.LinAlgError:
                min_vol_asset = volatilities.idxmin()
                #print(f"Removendo ativo {min_vol_asset} por problema na matriz.")
                current_assets.remove(min_vol_asset)
                if len(current_assets) <= 1:
                    success = False
                    break

        if not success:
            if len(current_assets) == 1:
                only_asset = current_assets[0]
                #print(f"Aportando 100% em {only_asset} por ser o único ativo restante em {month.strftime('%m/%Y')}.")

                amount_to_invest = aporte_inicial if month == pd.to_datetime(start_2023) else aporte_mensal
                total_invested_portfolio += amount_to_invest

                current_price = yearly_dataaux.iloc[-1][only_asset]
                new_quantity = amount_to_invest / current_price if current_price > 0 else 0
                quantities_portfolio[only_asset] += new_quantity

                start_of_month = month
                end_of_month = month + pd.offsets.MonthEnd(0)
                daily_data = data[(data['Data'] >= start_of_month) & (data['Data'] <= end_of_month)].reset_index(drop=True)

                if not daily_data.empty:
                    daily_prices = daily_data[asset_columns].ffill()
                    quantities_for_current_assets = {asset: quantities_portfolio[asset] for asset in daily_prices.columns}
                    portfolio_daily_values = daily_prices.multiply(pd.Series(quantities_for_current_assets), axis=1).sum(axis=1)
                    portfolio_values_paridade.extend(zip(daily_data['Data'], portfolio_daily_values))

                    for asset in asset_columns:
                        individual_daily_values = daily_prices[asset] * quantities_individual[asset]
                        individual_portfolio_values[asset].extend(zip(daily_data['Data'], individual_daily_values))

                    end_of_month_date = daily_data['Data'].iloc[-1]
                    end_of_month_prices = daily_data.iloc[-1][asset_columns]
                    asset_values = end_of_month_prices * pd.Series(quantities_portfolio)
                    total_portfolio_value = asset_values.sum()
                    asset_percentages = asset_values / total_portfolio_value

                    monthly_risk_parity_values.append({
                        'Date': end_of_month_date,
                        'TotalValue': total_portfolio_value,
                        'AssetValues': asset_values,
                        'AssetPercentages': asset_percentages
                    })

                continue
            else:
                print(f"Não foi possível encontrar uma combinação viável em {month.strftime('%m/%Y')}. Pulando.")
                continue

        # --- Otimização com os ativos válidos restantes ---
        cov_matrix = returns[current_assets].cov() * 1e-2
        cov_matrix_np = cov_matrix.values
        b_current = b[:len(current_assets)]
        x_initial_current = x_initial[:len(current_assets)]

        try:
            optimal_x, _ = solve_system_8(cov_matrix_np, b_current, x_initial_current)
        except ValueError as e:
            print(f"Falha na otimização após filtragem em {month.strftime('%m/%Y')}. Pulando.")
            continue

        x_initial[:len(current_assets)] = optimal_x

        amount_to_invest = aporte_inicial if month == pd.to_datetime(start_2023) else aporte_mensal
        total_invested_portfolio += amount_to_invest

        current_prices = yearly_dataaux.iloc[-1][current_assets].to_dict()
        new_quantities = calculate_investment(amount_to_invest, optimal_x, current_prices, current_assets)

        for asset in current_assets:
            quantities_portfolio[asset] += new_quantities[asset]

        # Risco
        risk_contributions, total_risk = calculate_risk_contributions(optimal_x, cov_matrix_np)
        if not check_equal_risk_contributions(risk_contributions):
            pass

        # Investimentos individuais
        for asset in asset_columns:
            total_invested_individual[asset] += amount_to_invest
            current_price_asset = yearly_dataaux.iloc[-1][asset]
            if current_price_asset > 0:
                quantities_individual[asset] += amount_to_invest / current_price_asset

        start_of_month = month
        end_of_month = month + pd.offsets.MonthEnd(0)
        daily_data = data[(data['Data'] >= start_of_month) & (data['Data'] <= end_of_month)].reset_index(drop=True)

        if not daily_data.empty:
            daily_prices = daily_data[asset_columns].ffill()
            quantities_for_current_assets = {asset: quantities_portfolio[asset] for asset in daily_prices.columns}
            portfolio_daily_values = daily_prices.multiply(pd.Series(quantities_for_current_assets), axis=1).sum(axis=1)
            portfolio_values_paridade.extend(zip(daily_data['Data'], portfolio_daily_values))

            for asset in asset_columns:
                individual_daily_values = daily_prices[asset] * quantities_individual[asset]
                individual_portfolio_values[asset].extend(zip(daily_data['Data'], individual_daily_values))

            end_of_month_date = daily_data['Data'].iloc[-1]
            end_of_month_prices = daily_data.iloc[-1][asset_columns]
            asset_values = end_of_month_prices * pd.Series(quantities_portfolio)
            total_portfolio_value = asset_values.sum()
            asset_percentages = asset_values / total_portfolio_value

            monthly_risk_parity_values.append({
                'Date': end_of_month_date,
                'TotalValue': total_portfolio_value,
                'AssetValues': asset_values,
                'AssetPercentages': asset_percentages
            })
# ----------------------------
# Estratégia CDI
# ----------------------------

cdi_values = []

def EstrategiaCDI(aporte_inicial, aporte_mensal, portfolio_values_cdi):
    saldo = 0

    for month in pd.date_range(start=start_2023, end=end_2023, freq='MS'):
        investment_date = month + pd.offsets.BMonthBegin(0)
        end_of_month = month + pd.offsets.MonthEnd(0)

        # Definir o aporte do mês
        if month == pd.to_datetime(start_2023):
            aporte = aporte_inicial
        else:
            aporte = aporte_mensal

        # Atualiza saldo e total investido
        saldo += aporte

        # Filtrar CDI diário do mês
        cdi_mes = cdi_data_total[
            (cdi_data_total['data'] >= investment_date) &
            (cdi_data_total['data'] <= end_of_month)
        ]

        # Aplicar rendimento diário sobre o saldo acumulado
        for _, row in cdi_mes.iterrows():
            rendimento_dia = row['valor'] / 100
            saldo *= (1 + rendimento_dia)
            portfolio_values_cdi.append((row['data'], saldo))

    return saldo


# ----------------------------
# Definir Período de Análise
# ----------------------------

start_2023 = '2019-01-02'
end_2023 = '2025-03-01'

# ----------------------------
# Executar as Estratégias
# ----------------------------

# Executar estratégias
EstrategiaEficiente(aporte_inicial, aporte_mensal, total_investido, portfolio_values_eficiente)
ParidadeDeRisco(aporte_inicial, aporte_mensal, n_assets, x_initial, b, portfolio_values_paridade, individual_portfolio_values)
EstrategiaCDI(aporte_inicial, aporte_mensal, cdi_values)

# ----------------------------
# Comparar Resultados
# ----------------------------

# Converte listas em DataFrame para visualização
df_eficiente = pd.DataFrame(portfolio_values_eficiente, columns=['Data', 'Eficiente'])
df_paridade = pd.DataFrame(portfolio_values_paridade, columns=['Data', 'Paridade'])
df_cdi = pd.DataFrame(cdi_values, columns=['Data', 'CDI'])

# Junta os DataFrames usando merge outer para não perder datas
df_comparacao = pd.merge(df_eficiente, df_paridade, on='Data', how='outer')
df_comparacao = pd.merge(df_comparacao, df_cdi, on='Data', how='outer')

# Ordenar pelas datas e preencher valores ausentes com o último disponível
df_comparacao.sort_values('Data', inplace=True)
df_comparacao.ffill(inplace=True)

hover_texts = []
for date in df_comparacao['Data']:
    date_month = pd.to_datetime(date).replace(day=1).normalize()
    vols = volatilidades_mensais.get(date_month, {})
    
    tooltip = f"<b>Data:</b> {date.strftime('%Y-%m-%d')}<br>"
    if vols:
        tooltip += "<b>Volatilidade:</b><br>"
        for asset, vol in vols.items():
            tooltip += f"{asset}: {vol:.2f}%<br>"
    else:
        tooltip += "Sem dados de volatilidade"
    hover_texts.append(tooltip)

# Cria o gráfico com Plotly
fig = go.Figure()

fig.add_trace(go.Scatter(
    x=df_comparacao['Data'],
    y=df_comparacao['Eficiente'],
    mode='lines',
    name='Carteira Eficiente',
    line=dict(color='blue'),
    hoverinfo='text',
    text=hover_texts
))

fig.add_trace(go.Scatter(
    x=df_comparacao['Data'],
    y=df_comparacao['Paridade'],
    mode='lines',
    name='Paridade de Risco',
    line=dict(color='green'),
    hoverinfo='skip',
    #text=hover_texts
))

fig.add_trace(go.Scatter(
    x=df_comparacao['Data'],
    y=df_comparacao['CDI'],
    mode='lines',
    name='CDI',
    line=dict(color='orange'),
    hoverinfo='skip'
))

fig.update_layout(
    title='Comparação: Carteira Eficiente vs Paridade de Risco vs CDI',
    xaxis_title='Data',
    yaxis_title='Valor da Carteira',
    hovermode='x unified',
    template='plotly_white',
    autosize=False,
    width=1200,
    height=600
)

fig.show()
