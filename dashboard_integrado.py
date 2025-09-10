# =======================
# Dashboard único (Dash)
# Layout do dashboard_v4 + Lógica completa do estrategias.py
# =======================

import dash
from dash import dcc, html, Input, Output, State
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime
import dash_bootstrap_components as dbc
from dash.exceptions import PreventUpdate

# ====== Imports usados na lógica (mantidos do estrategias.py) ======
import warnings
from scipy.optimize import minimize
import matplotlib.pyplot as plt  # (não usado no Dash, mas mantido para compatibilidade)
import matplotlib.dates as mdates  # (não usado no Dash, mas mantido para compatibilidade)

warnings.filterwarnings("ignore", category=RuntimeWarning, module="scipy.optimize._slsqp_py")

# =============================================
# CARREGAMENTO DE DADOS (mesma estrutura do estrategias.py)
# =============================================

# Caminhos dos arquivos (mantenha a pasta 'arquivos' ao lado do script)
# file_path = 'arquivos/Dados_Ativos_B32.csv'
file_path = 'arquivos/Dados_Ativos_B3_close.csv'
# file_path = 'arquivos/Dados_Ativos_B3_AdjClose.csv'
# file_path = 'arquivos/IvvbBova.csv'
cdi_file_path = 'arquivos/cdi_data_total.csv'

# Carrega CDI e dados de ativos
cdi_data_total = pd.read_csv(cdi_file_path)
data = pd.read_csv(file_path)

# Ajuste de datas exatamente como no estrategias.py
data['Data'] = pd.to_datetime(data['Data'], format='%d/%m/%Y %H:%M:%S')
cdi_data_total['data'] = pd.to_datetime(cdi_data_total['data'], dayfirst=True) + pd.Timedelta(hours=16, minutes=56)

# =============================================
# VARIÁVEIS GLOBAIS USADAS PELAS ESTRATÉGIAS (mesmo padrão)
# =============================================

# Estas variáveis serão redefinidas a cada simulação, conforme a seleção do usuário
asset_columns = []                   # lista de ativos selecionados no dropdown
aporte_inicial = 1000
aporte_mensal = 400
total_investido = 0
quantities = {}                      # quantidades por ativo (Estratégia Eficiente)

# Paridade de Risco
n_assets = 0
x_initial = np.array([])             # pesos iniciais
b = np.array([])                     # vetor b
total_invested_portfolio = 0
quantities_portfolio = {}
quantities_individual = {}
total_invested_individual = {}

# Acumuladores de séries (valores diários)
portfolio_values_eficiente = []
portfolio_values_paridade = []
individual_portfolio_values = {}
monthly_risk_parity_values = []      # lista de dicts com 'Date', 'TotalValue', 'AssetValues', 'AssetPercentages', ...
cdi_values = []

# Período
start_2023 = '2019-01-02'
end_2023 = '2025-03-01'

# Outras estruturas
volatilidades_mensais = {}

# =============================================
# FUNÇÕES (cópia fiel da lógica de estrategias.py — sem alterar cálculo)
# =============================================

def filter_cdi_data(month, df_cdi_total, df_assets):
    one_year_before = month - pd.DateOffset(years=1)
    cdi_filtered = df_cdi_total[(df_cdi_total['data'] >= one_year_before) & (df_cdi_total['data'] < month)].copy()
    cdi_filtered['data'] = cdi_filtered['data'].dt.normalize()
    cdi_filtered = cdi_filtered[cdi_filtered['data'].isin(df_assets['Data'].dt.normalize())]
    return cdi_filtered

def get_yearly_data(month, data_):
    one_year_before = month - pd.DateOffset(years=1)
    yearly_data = data_[(data_['Data'] >= one_year_before) & (data_['Data'] < month)]
    return yearly_data

def calculate_investment(total_portfolio_value, weights, current_prices, asset_cols):
    quantities2 = {}
    for i, asset in enumerate(asset_cols):
        current_price = current_prices[asset]
        amount_to_invest_in_asset = total_portfolio_value * weights[i]
        if current_price > 0:
            quantities2[asset] = amount_to_invest_in_asset / current_price
        else:
            quantities2[asset] = 0
    return quantities2

def calculate_risk_contributions(x, cov_matrix):
    sigma_x = np.dot(cov_matrix, x)
    total_risk = np.sqrt(np.dot(x.T, sigma_x))
    marginal_risk = sigma_x / total_risk if total_risk != 0 else np.zeros_like(sigma_x)
    risk_contributions = x * marginal_risk
    return risk_contributions, total_risk

def check_equal_risk_contributions(risk_contributions):
    total_risk = np.sum(risk_contributions)
    n_assets_ = len(risk_contributions)
    target_risk = total_risk / n_assets_ if n_assets_ != 0 else 0
    return np.all(np.isclose(risk_contributions, target_risk, atol=1e-3))

def calculate_variance(x, cov_matrix):
    return np.dot(x.T, np.dot(cov_matrix, x))

def objective(x, cov_matrix, b_):
    variance = calculate_variance(x, cov_matrix)
    sqrt_variance = np.sqrt(variance) if variance > 0 else 1.0
    w = x / sqrt_variance
    quadratic_term = 0.5 * np.dot(w.T, b_ / (w + 1e-18))
    log_term = np.dot(b_.T, np.log(w + 1e-18))
    return quadratic_term - log_term

def weight_sum_constraint(x):
    return np.sum(x) - 1.0

def weight_bounds(n_assets_):
    return [(1e-19, 1) for _ in range(n_assets_)]

def solve_system_8(cov_matrix, b_, initial_x):
    n_assets_ = len(b_)
    constraints = {'type': 'eq', 'fun': weight_sum_constraint}
    bounds = weight_bounds(n_assets_)
    solution = minimize(
        objective, initial_x, args=(cov_matrix, b_),
        method='SLSQP', bounds=bounds, constraints=constraints,
        options={'maxiter': 1000, 'ftol': 1e-9}
    )
    if not solution.success:
        raise ValueError(f"Falha na otimização: {solution.message}")
    return solution.x, solution.fun

def EstrategiaEficiente(aporte_inicial_, aporte_mensal_, total_investido_, portfolio_values_eficiente_):
    global quantities, asset_columns, volatilidades_mensais
    for month in pd.date_range(start=start_2023, end=end_2023, freq='MS'):
        investment_date = month + pd.offsets.BMonthBegin(0)
        investment_datetime = investment_date.replace(hour=16, minute=56, second=0)
        one_year_before = investment_date - pd.DateOffset(years=1)

        yearly_data = data[(data['Data'] >= one_year_before) & (data['Data'] < investment_date)]
        yearly_dataaux = data[(data['Data'] >= one_year_before) & (data['Data'] <= investment_datetime)]

        cdidiario_filtered = filter_cdi_data(month, cdi_data_total, yearly_data)
        returns = yearly_data[asset_columns].pct_change().dropna()

        if returns.empty:
            continue

        current_assets = asset_columns.copy()
        success = False
        max_attempts = len(asset_columns)
        min_vol_threshold = 13

        for attempt in range(max_attempts):
            current_returns = returns[current_assets]

            volatilities = current_returns.std() * (252 ** 0.5) * 100
            volatilidades_mensais[pd.to_datetime(month).normalize()] = volatilities.to_dict()

            low_vol_assets = volatilities[volatilities < min_vol_threshold]
            if not low_vol_assets.empty:
                for asset in low_vol_assets.index:
                    if asset in current_assets:
                        current_assets.remove(asset)
                if len(current_assets) <= 1:
                    success = False
                    break

            current_returns = returns[current_assets]
            cov_matrix = current_returns.cov()

            try:
                _ = np.linalg.cholesky(cov_matrix)  # valida positividade
                success = True
                break
            except np.linalg.LinAlgError:
                volatilities = current_returns.std()
                min_vol_asset = volatilities.idxmin()
                if min_vol_asset in current_assets:
                    current_assets.remove(min_vol_asset)
                if len(current_assets) <= 1:
                    success = False
                    break
                if len(current_assets) == 2:
                    success = True
                    break

        if not success:
            if len(current_assets) == 1:
                only_asset = current_assets[0]
                amount_to_invest = aporte_inicial_ if month == pd.to_datetime(start_2023) else aporte_mensal_
                total_investido_ += amount_to_invest
                current_price = yearly_dataaux.iloc[-1][only_asset]
                new_quantity = amount_to_invest / current_price if current_price > 0 else 0
                quantities[only_asset] += new_quantity

                start_of_month = month
                end_of_month = month + pd.offsets.MonthEnd(0)
                daily_data = data[(data['Data'] >= start_of_month) & (data['Data'] <= end_of_month)].reset_index(drop=True)
                if not daily_data.empty:
                    daily_prices = daily_data[asset_columns].ffill()
                    portfolio_daily_values = daily_prices.multiply(list(quantities.values()), axis=1).sum(axis=1)
                    portfolio_values_eficiente_.extend(zip(daily_data['Data'], portfolio_daily_values))
                continue
            else:
                continue

        expected_returns = current_returns.mean().values
        e_transposto = np.ones(len(current_assets)).reshape(1, -1)
        CDIRet = cdidiario_filtered['valor'].mean()
        Rf_e = np.full((len(current_assets), 1), CDIRet / 100)
        M_Rfe = expected_returns.reshape(-1, 1) - Rf_e

        # inversa de cov (via Cholesky)
        L = np.linalg.cholesky(cov_matrix)
        L_inv = np.linalg.inv(L)
        cov_matrix_inv = L_inv.T @ L_inv

        V_M_Rfe = np.matmul(cov_matrix_inv, M_Rfe).flatten()
        V_eT = np.matmul(e_transposto, cov_matrix_inv)
        Div = np.matmul(V_eT, M_Rfe)[0, 0]
        if Div == 0:
            continue

        percentages = V_M_Rfe / Div
        full_percentages = {asset: 0.0 for asset in asset_columns}
        for asset, pct in zip(current_assets, percentages):
            full_percentages[asset] = pct

        amount_to_invest = aporte_inicial_ if month == pd.to_datetime(start_2023) else aporte_mensal_
        total_investido_ += amount_to_invest

        current_prices = yearly_dataaux.iloc[-1][asset_columns].to_dict()
        new_quantities = calculate_investment(amount_to_invest, list(full_percentages.values()), current_prices, asset_columns)
        for asset in asset_columns:
            quantities[asset] += new_quantities[asset]

        start_of_month = month
        end_of_month = month + pd.offsets.MonthEnd(0)
        daily_data = data[(data['Data'] >= start_of_month) & (data['Data'] <= end_of_month)].reset_index(drop=True)
        if not daily_data.empty:
            daily_prices = daily_data[asset_columns].ffill()
            portfolio_daily_values = daily_prices.multiply(list(quantities.values()), axis=1).sum(axis=1)
            portfolio_values_eficiente_.extend(zip(daily_data['Data'], portfolio_daily_values))

def ParidadeDeRisco(aporte_inicial_, aporte_mensal_, n_assets_, x_initial_, b_, portfolio_values_paridade_, individual_portfolio_values_):
    global quantities_portfolio, quantities_individual, total_invested_portfolio, monthly_risk_parity_values, asset_columns
    for month in pd.date_range(start=start_2023, end=end_2023, freq='MS'):
        investment_date = month + pd.offsets.BMonthBegin(0)
        investment_datetime = investment_date.replace(hour=16, minute=56, second=0)
        one_year_before = investment_date - pd.DateOffset(years=1)

        yearly_data = data[(data['Data'] >= one_year_before) & (data['Data'] < investment_date)]
        yearly_dataaux = data[(data['Data'] >= one_year_before) & (data['Data'] <= investment_datetime)]

        if len(yearly_data) < 2:
            continue

        returns = yearly_data[asset_columns].pct_change().dropna()
        current_assets = asset_columns.copy()
        success = False
        max_attempts = len(asset_columns)
        min_vol_threshold = 13

        for attempt in range(max_attempts):
            current_returns = returns[current_assets]
            volatilities = current_returns.std() * (252 ** 0.5) * 100

            low_vol_assets = volatilities[volatilities < min_vol_threshold]
            if not low_vol_assets.empty:
                for asset in low_vol_assets.index:
                    if asset in current_assets:
                        current_assets.remove(asset)
                if len(current_assets) <= 1:
                    success = False
                    break

            current_returns = returns[current_assets]
            cov_matrix = current_returns.cov() * 1e-2

            try:
                _ = np.linalg.cholesky(cov_matrix)
                success = True
                break
            except np.linalg.LinAlgError:
                min_vol_asset = volatilities.idxmin()
                if min_vol_asset in current_assets:
                    current_assets.remove(min_vol_asset)
                if len(current_assets) <= 1:
                    success = False
                    break

        if not success:
            if len(current_assets) == 1:
                only_asset = current_assets[0]
                amount_to_invest = aporte_inicial_ if month == pd.to_datetime(start_2023) else aporte_mensal_
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
                    portfolio_values_paridade_.extend(zip(daily_data['Data'], portfolio_daily_values))

                    for asset in asset_columns:
                        individual_daily_values = daily_prices[asset] * quantities_individual[asset]
                        individual_portfolio_values_[asset].extend(zip(daily_data['Data'], individual_daily_values))

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
                continue

        cov_matrix = returns[current_assets].cov() * 1e-2
        cov_matrix_np = cov_matrix.values
        b_current = b_[:len(current_assets)]
        x_initial_current = x_initial_[:len(current_assets)]

        try:
            optimal_x, _ = solve_system_8(cov_matrix_np, b_current, x_initial_current)
        except ValueError:
            continue

        x_initial_[:len(current_assets)] = optimal_x

        amount_to_invest = aporte_inicial_ if month == pd.to_datetime(start_2023) else aporte_mensal_
        total_invested_portfolio += amount_to_invest

        current_prices = yearly_dataaux.iloc[-1][current_assets].to_dict()
        new_quantities = calculate_investment(amount_to_invest, optimal_x, current_prices, current_assets)
        for asset in current_assets:
            quantities_portfolio[asset] += new_quantities[asset]

        # ===== Pesos e RC do solver (alvo)
        risk_contributions, total_risk = calculate_risk_contributions(optimal_x, cov_matrix_np)

        for asset in asset_columns:
            total_invested_individual[asset] += amount_to_invest
            current_price_asset = yearly_dataaux.iloc[-1][asset]
            if current_price_asset > 0:
                quantities_individual[asset] += amount_to_invest / current_price_asset

        start_of_month = month
        end_of_month = month + pd.offsets.MonthEnd(0)
        daily_data = data[(data['Data'] >= start_of_month) & (data['Data'] <= end_of_month)].reset_index(drop=True)

        # ----- NOVO: pesos e contrib. de risco (mês corrente) -----
        weights_series = pd.Series(0.0, index=asset_columns, dtype=float)
        weights_series[current_assets] = optimal_x

        rc, total_risk = calculate_risk_contributions(optimal_x, cov_matrix_np)
        rc_pct = rc / rc.sum() if rc.sum() != 0 else np.zeros_like(rc)
        rc_pct_series = pd.Series(0.0, index=asset_columns, dtype=float)
        rc_pct_series[current_assets] = rc_pct

        if not daily_data.empty:
            daily_prices = daily_data[asset_columns].ffill()
            quantities_for_current_assets = {asset: quantities_portfolio[asset] for asset in daily_prices.columns}
            portfolio_daily_values = daily_prices.multiply(pd.Series(quantities_for_current_assets), axis=1).sum(axis=1)
            portfolio_values_paridade_.extend(zip(daily_data['Data'], portfolio_daily_values))

            for asset in asset_columns:
                individual_daily_values = daily_prices[asset] * quantities_individual[asset]
                individual_portfolio_values_[asset].extend(zip(daily_data['Data'], individual_daily_values))

            end_of_month_date = daily_data['Data'].iloc[-1]
            end_of_month_prices = daily_data.iloc[-1][asset_columns]
            asset_values = end_of_month_prices * pd.Series(quantities_portfolio)
            total_portfolio_value = asset_values.sum()
            asset_percentages = asset_values / total_portfolio_value

            # ===== Paridade de Risco: alvo e realizado
            weights_target = pd.Series(0.0, index=asset_columns, dtype=float)
            weights_target[current_assets] = optimal_x

            rc_target, _ = calculate_risk_contributions(optimal_x, cov_matrix_np)
            rc_target_pct = rc_target / rc_target.sum() if rc_target.sum() != 0 else np.zeros_like(rc_target)
            rc_target_series = pd.Series(0.0, index=asset_columns, dtype=float)
            rc_target_series[current_assets] = rc_target_pct

            weights_realized = (asset_values / total_portfolio_value).astype(float)
            w_real_cur = weights_realized[current_assets].values
            rc_realized, _ = calculate_risk_contributions(w_real_cur, cov_matrix_np)
            rc_realized_pct = rc_realized / rc_realized.sum() if rc_realized.sum() != 0 else np.zeros_like(rc_realized)
            rc_realized_series = pd.Series(0.0, index=asset_columns, dtype=float)
            rc_realized_series[current_assets] = rc_realized_pct

            monthly_risk_parity_values.append({
                'Date': end_of_month_date,
                'TotalValue': total_portfolio_value,
                'AssetValues': asset_values,              # R$ por ativo (carteira realizada)
                'AssetPercentages': asset_percentages,    # % por valor (carteira realizada)
                'WeightsTarget': weights_target,          # pesos-alvo do solver (Paridade)
                'RiskContribTarget': rc_target_series,    # RC% alvo (tende a ser igualitária)
                'WeightsRealized': weights_realized,      # pesos efetivos na carteira
                'RiskContribRealized': rc_realized_series # RC% realizada na carteira
            })

def EstrategiaCDI(aporte_inicial_, aporte_mensal_, portfolio_values_cdi_):
    saldo = 0
    for month in pd.date_range(start=start_2023, end=end_2023, freq='MS'):
        investment_date = month + pd.offsets.BMonthBegin(0)
        end_of_month = month + pd.offsets.MonthEnd(0)
        aporte = aporte_inicial_ if month == pd.to_datetime(start_2023) else aporte_mensal_
        saldo += aporte

        cdi_mes = cdi_data_total[
            (cdi_data_total['data'] >= investment_date) &
            (cdi_data_total['data'] <= end_of_month)
        ]
        for _, row in cdi_mes.iterrows():
            rendimento_dia = row['valor'] / 100
            saldo *= (1 + rendimento_dia)
            portfolio_values_cdi_.append((row['data'], saldo))
    return saldo

# =============================================
# DASHBOARD (layout moderno)
# =============================================

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.LUX])
server = app.server

COLORS = {
    'dark': {
        'bg': '#0F172A',
        'card_bg': '#1E293B',
        'text': '#F8FAFC',
        'primary': '#3B82F6',
        'secondary': '#10B981',
        'accent': '#8B5CF6',
        'success': '#10B981',
        'warning': '#F59E0B',
        'danger': '#EF4444',
        'navbar': 'dark',
        'toggle_label': '☀️ Modo Claro'
    },
    'light': {
        'bg': '#F8FAFC',
        'card_bg': '#FFFFFF',
        'text': '#1E293B',
        'primary': '#2563EB',
        'secondary': '#059669',
        'accent': '#7C3AED',
        'success': '#059669',
        'warning': '#D97706',
        'danger': '#DC2626',
        'navbar': 'light',
        'toggle_label': '🌙 Modo Escuro'
    }
}

def hex_to_rgba(hex_color: str, alpha: float) -> str:
    """Converte '#RRGGBB' para 'rgba(r,g,b,a)'."""
    h = hex_color.lstrip('#')
    r = int(h[0:2], 16)
    g = int(h[2:4], 16)
    b = int(h[4:6], 16)
    return f'rgba({r},{g},{b},{alpha})'

CUSTOM_STYLES = {
    'card': {
        'borderRadius': '12px',
        'boxShadow': '0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06)',
        'border': 'none',
        'transition': 'background-color .25s ease, color .25s ease'
    },
    'navbar': {
        'boxShadow': '0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06)',
        'padding': '0.75rem 1rem',
        'transition': 'background-color .25s ease, color .25s ease'
    },
    'metric_card': {
        'borderLeft': f'4px solid {COLORS["dark"]["primary"]}'
    },
    'button': {
        'borderRadius': '8px',
        'fontWeight': '500',
        'transition': 'all 0.2s ease'
    }
}

app.layout = html.Div(
    id='main-container',
    children=[
        dcc.Store(id='theme-store', data='dark'),

        # NAVBAR (removidos Relatório e Exportar)
        dbc.Navbar(
            dbc.Container(
                [
                    html.A(
                        dbc.Row(
                            [
                                dbc.Col(html.Img(src="https://cdn-icons-png.flaticon.com/512/2583/2583344.png", height="28px")),
                                dbc.Col(dbc.NavbarBrand("Risco & Retorno", className="ms-2 fw-bold")),
                            ],
                            align="center",
                            className="g-0",
                        ),
                        href="#",
                        style={"textDecoration": "none"},
                    ),
                    dbc.Row(
                        [
                            dbc.Col(
                                dbc.ButtonGroup(
                                    [
                                        dbc.Switch(
                                            id="theme-toggle",
                                            label=COLORS['dark']['toggle_label'],
                                            value=False,
                                            className="d-flex align-items-center ms-3",
                                            style={"marginLeft": "10px"}
                                        ),
                                    ],
                                    className="ms-auto",
                                ),
                                width="auto",
                            ),
                        ],
                        align="center",
                    ),
                ],
                fluid=True,
            ),
            color="primary",
            dark=True,
            sticky="top",
            style=CUSTOM_STYLES['navbar'],
            className="mb-4"
        ),

        # CONTEÚDO
        dbc.Container(
            fluid=True,
            className="py-3",
            children=[
                # Métricas
                dbc.Row(
                    [
                        dbc.Col(
                            dbc.Card(
                                [
                                    dbc.CardHeader("Valor Investido", className="small text-muted fw-bold"),
                                    dbc.CardBody(
                                        [
                                            html.H4("R$ 0,00", id="total-investido", className="mb-1 fw-bold"),
                                            html.Small([html.I(className="fas fa-arrow-up me-1"), "+0% em 12M"], className="text-success"),
                                        ],
                                        className="pt-2 pb-3"
                                    ),
                                ],
                                className="h-100",
                                style={**CUSTOM_STYLES['card'], **CUSTOM_STYLES['metric_card']}
                            ),
                            md=3, className="mb-4",
                        ),
                        dbc.Col(
                            dbc.Card(
                                [
                                    dbc.CardHeader("Carteira Eficiente", className="small text-muted fw-bold"),
                                    dbc.CardBody([
                                        html.H4("0%", id="retorno-eficiente", className="mb-1 fw-bold"),
                                        html.Small("R$ 0,00", id="valor-eficiente-final", className="text-muted")
                                    ], className="pt-2 pb-3"),
                                ],
                                className="h-100",
                                style={**CUSTOM_STYLES['card'], 'borderLeft': f'4px solid {COLORS["dark"]["secondary"]}'}
                            ),
                            md=3, className="mb-4",
                        ),
                        dbc.Col(
                            dbc.Card(
                                [
                                    dbc.CardHeader("Paridade de Risco", className="small text-muted fw-bold"),
                                    dbc.CardBody([
                                        html.H4("0%", id="retorno-paridade", className="mb-1 fw-bold"),
                                        html.Small("R$ 0,00", id="valor-paridade-final", className="text-muted")
                                    ], className="pt-2 pb-3"),
                                ],
                                className="h-100",
                                style={**CUSTOM_STYLES['card'], 'borderLeft': f'4px solid {COLORS["dark"]["accent"]}'}
                            ),
                            md=3, className="mb-4",
                        ),
                        dbc.Col(
                            dbc.Card(
                                [
                                    dbc.CardHeader("Benchmark (CDI)", className="small text-muted fw-bold"),
                                    dbc.CardBody([
                                        html.H4("0%", id="retorno-cdi", className="mb-1 fw-bold"),
                                        html.Small("R$ 0,00", id="valor-cdi-final", className="text-muted")
                                    ], className="pt-2 pb-3"),
                                ],
                                className="h-100",
                                style={**CUSTOM_STYLES['card'], 'borderLeft': '4px solid #94A3B8'}
                            ),
                            md=3, className="mb-4",
                        ),
                    ],
                    className="g-4",
                ),

                # Parâmetros + Gráfico principal
                dbc.Row(
                    [
                        dbc.Col(
                            dbc.Card(
                                [
                                    dbc.CardHeader("Parâmetros da Simulação", className="fw-bold"),
                                    dbc.CardBody(
                                        [
                                            dbc.Form(
                                                [
                                                    dbc.Row(
                                                        [
                                                            dbc.Col(
                                                                [
                                                                    dbc.Label("Aporte Inicial (R$)", className="small fw-bold mb-2"),
                                                                    dbc.Input(id="aporte-inicial", type="number", value=1000, min=0,
                                                                              className="mb-3", style={'borderRadius': '8px'}),
                                                                ], md=6,
                                                            ),
                                                            dbc.Col(
                                                                [
                                                                    dbc.Label("Aporte Mensal (R$)", className="small fw-bold mb-2"),
                                                                    dbc.Input(id="aporte-mensal", type="number", value=400, min=0,
                                                                              className="mb-3", style={'borderRadius': '8px'}),
                                                                ], md=6,
                                                            ),
                                                        ], className="mb-3",
                                                    ),
                                                    dbc.Row(
                                                        [
                                                            dbc.Col(
                                                                [
                                                                    dbc.Label("Data Inicial", className="small fw-bold mb-2"),
                                                                    dcc.DatePickerSingle(
                                                                        id='start-date',
                                                                        min_date_allowed=data['Data'].min().date(),
                                                                        max_date_allowed=data['Data'].max().date(),
                                                                        initial_visible_month=data['Data'].min().date(),
                                                                        date=data['Data'].min().date(),
                                                                        className="w-100 mb-3",
                                                                        display_format='DD/MM/YYYY',
                                                                    ),
                                                                ], md=6,
                                                            ),
                                                            dbc.Col(
                                                                [
                                                                    dbc.Label("Data Final", className="small fw-bold mb-2"),
                                                                    dcc.DatePickerSingle(
                                                                        id='end-date',
                                                                        min_date_allowed=data['Data'].min().date(),
                                                                        max_date_allowed=data['Data'].max().date(),
                                                                        initial_visible_month=data['Data'].max().date(),
                                                                        date=data['Data'].max().date(),
                                                                        className="w-100 mb-3",
                                                                        display_format='DD/MM/YYYY',
                                                                    ),
                                                                ], md=6,
                                                            ),
                                                        ], className="mb-3",
                                                    ),
                                                    dbc.Row(
                                                        [
                                                            dbc.Col(
                                                                [
                                                                    dbc.Label("Selecione os Ativos", className="small fw-bold mb-2"),
                                                                    dcc.Dropdown(
                                                                        id="asset-selector",
                                                                        options=[{"label": a, "value": a} for a in data.columns if a != "Data"],
                                                                        value=[c for c in list(data.columns) if c != "Data"][:5],
                                                                        multi=True,
                                                                        placeholder="Escolha qualquer quantidade de ativos"
                                                                    )
                                                                ], md=12
                                                            )
                                                        ], className="mb-4"
                                                    ),
                                                    dbc.Button(
                                                        [html.I(className="fas fa-play me-2"), "Executar Simulação"],
                                                        id="run-simulation",
                                                        color="primary",
                                                        className="w-100 fw-bold",
                                                        size="lg",
                                                        style={**CUSTOM_STYLES['button'], 'padding': '0.75rem'}
                                                    ),
                                                ]
                                            )
                                        ]
                                    ),
                                ],
                                className="h-100",
                                style=CUSTOM_STYLES['card']
                            ),
                            md=4, className="mb-4",
                        ),
                        dbc.Col(
                            dbc.Card(
                                [
                                    dbc.CardHeader(
                                        html.Div(["Desempenho das Estratégias"]), className="fw-bold"
                                    ),
                                    dbc.CardBody(
                                        dcc.Graph(
                                            id='main-graph',
                                            config={
                                                'displayModeBar': True,
                                                'displaylogo': False,
                                                'modeBarButtonsToRemove': ['lasso2d', 'select2d'],
                                                'modeBarButtonsToAdd': ['resetScale2d']
                                            },
                                            style={'height': '460px', 'width': '100%', 'transition': 'background-color .25s ease, color .25s ease'},
                                            className="w-100"
                                        )

                                    ),
                                ],
                                style=CUSTOM_STYLES['card']
                            ),
                            md=8, className="mb-4",
                        ),
                    ],
                    className="g-4",
                ),

                # Gráficos secundários
                dbc.Row(
                    [
                        dbc.Col(
                            dbc.Card(
                                [
                                    dbc.CardHeader("Alocação de Ativos - Paridade de Risco", className="fw-bold"),
                                    dbc.CardBody(dcc.Graph(id='asset-pie-chart', style={'height': '300px', 'transition': 'background-color .25s ease, color .25s ease'},
                                                           config={'displayModeBar': False}))
                                ],
                                className="h-100", style=CUSTOM_STYLES['card']
                            ),
                            md=6, className="mb-4",
                        ),
                        dbc.Col(
                            dbc.Card(
                                [
                                    dbc.CardHeader("Contribuição de Risco - Paridade de Risco", className="fw-bold"),
                                    dbc.CardBody(dcc.Graph(id='risk-bar-chart', style={'height': '300px', 'transition': 'background-color .25s ease, color .25s ease'},
                                                           config={'displayModeBar': False}))
                                ],
                                className="h-100", style=CUSTOM_STYLES['card']
                            ),
                            md=6, className="mb-4",
                        ),
                    ],
                    className="g-4",
                ),

                # Tabela
                dbc.Row(
                    dbc.Col(
                        dbc.Card(
                            [
                                dbc.CardHeader("Métricas de Desempenho", className="fw-bold"),
                                dbc.CardBody(html.Div(id='performance-table', className="table-responsive"))
                            ],
                            style=CUSTOM_STYLES['card']
                        ),
                        className="mb-4",
                    )
                ),
            ],
        ),
    ],
    style={'minHeight': '100vh', 'fontFamily': "'Inter', sans-serif", 'transition': 'background-color .25s ease, color .25s ease'},
)

# Incluir fontes
app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
        <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css">
        <style>
            body { font-family: 'Inter', sans-serif; }
            .metric-card:hover { transform: translateY(-2px); transition: all 0.2s ease; }
            .dropdown-menu { font-size: 0.85rem; }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''

# =============================================
# CALLBACKS
# =============================================

@app.callback(
    Output('theme-store', 'data'),
    Output('theme-toggle', 'label'),
    Input('theme-toggle', 'value'),
    State('theme-store', 'data'),
)
def toggle_theme(toggle_value, current_theme):
    new_theme = 'light' if toggle_value else 'dark'
    return new_theme, COLORS[new_theme]['toggle_label']

@app.callback(
    Output('main-container', 'style'),
    Input('theme-store', 'data'),
)
def update_theme_style(theme):
    return {
        'backgroundColor': COLORS[theme]['bg'],
        'color': COLORS[theme]['text'],
        'transition': 'background-color .25s ease, color .25s ease'
    }

@app.callback(
    Output('main-graph', 'figure'),
    Output('total-investido', 'children'),
    Output('retorno-eficiente', 'children'),
    Output('retorno-paridade', 'children'),
    Output('retorno-cdi', 'children'),
    Output('valor-eficiente-final', 'children'),
    Output('valor-paridade-final', 'children'),
    Output('valor-cdi-final', 'children'),
    Output('asset-pie-chart', 'figure'),
    Output('risk-bar-chart', 'figure'),
    Output('performance-table', 'children'),
    Input('run-simulation', 'n_clicks'),
    Input('theme-store', 'data'),
    State('aporte-inicial', 'value'),
    State('aporte-mensal', 'value'),
    State('start-date', 'date'),
    State('end-date', 'date'),
    State('asset-selector', 'value'),
)
def update_dashboard(n_clicks, theme, aporte_ini, aporte_mes, start_date, end_date, selected_assets):
    if n_clicks is None:
        raise PreventUpdate
    if not selected_assets:
        raise PreventUpdate

    # ===============================
    # PREPARAÇÃO (mesma limpeza do estrategias.py)
    # ===============================
    global asset_columns, aporte_inicial, aporte_mensal, total_investido
    global n_assets, x_initial, b, total_invested_portfolio
    global quantities, quantities_portfolio, quantities_individual
    global total_invested_individual, portfolio_values_eficiente
    global portfolio_values_paridade, individual_portfolio_values
    global monthly_risk_parity_values, cdi_values, start_2023, end_2023
    global volatilidades_mensais

    # Seleção de ativos vinda do dropdown
    asset_columns = list(selected_assets)

    # Converte colunas escolhidas para float (como no estrategias.py)
    cols_validas = []
    for col in asset_columns:
        try:
            data[col] = data[col].astype(str).str.replace(',', '.').astype(float)
            cols_validas.append(col)
        except Exception:
            if col in data.columns:
                data.drop(columns=col, inplace=True)
    asset_columns = cols_validas

    # (Re)inicialização das variáveis exatamente como no estrategias.py
    aporte_inicial = float(aporte_ini or 0)
    aporte_mensal = float(aporte_mes or 0)
    total_investido = 0.0

    quantities = {asset: 0 for asset in asset_columns}

    n_assets = len(asset_columns)
    x_initial = np.ones(n_assets) / n_assets if n_assets > 0 else np.array([])
    b = np.ones(n_assets) / n_assets if n_assets > 0 else np.array([])

    total_invested_portfolio = 0.0
    quantities_portfolio = {asset: 0 for asset in asset_columns}
    quantities_individual = {asset: 0 for asset in asset_columns}
    total_invested_individual = {asset: 0 for asset in asset_columns}

    portfolio_values_eficiente = []
    portfolio_values_paridade = []
    individual_portfolio_values = {asset: [] for asset in asset_columns}
    monthly_risk_parity_values = []
    cdi_values = []
    volatilidades_mensais = {}

    # Datas
    start_2023 = pd.to_datetime(start_date).strftime('%Y-%m-%d')
    end_2023 = pd.to_datetime(end_date).strftime('%Y-%m-%d')

    # ===============================
    # EXECUÇÃO DAS ESTRATÉGIAS (sem alterar a lógica)
    # ===============================
    EstrategiaEficiente(aporte_inicial, aporte_mensal, total_investido, portfolio_values_eficiente)
    ParidadeDeRisco(aporte_inicial, aporte_mensal, n_assets, x_initial, b, portfolio_values_paridade, individual_portfolio_values)
    EstrategiaCDI(aporte_inicial, aporte_mensal, cdi_values)

    # ===============================
    # CONSTRUÇÃO DOS DATAFRAMES (mesmo procedimento do estrategias.py)
    # ===============================
    df_eficiente = pd.DataFrame(portfolio_values_eficiente, columns=['Data', 'Eficiente']) if portfolio_values_eficiente else pd.DataFrame(columns=['Data','Eficiente'])
    df_paridade = pd.DataFrame(portfolio_values_paridade, columns=['Data', 'Paridade']) if portfolio_values_paridade else pd.DataFrame(columns=['Data','Paridade'])
    df_cdi = pd.DataFrame(cdi_values, columns=['Data', 'CDI']) if cdi_values else pd.DataFrame(columns=['Data','CDI'])

    df_comparacao = pd.merge(df_eficiente, df_paridade, on='Data', how='outer')
    df_comparacao = pd.merge(df_comparacao, df_cdi, on='Data', how='outer')
    if not df_comparacao.empty:
        df_comparacao.sort_values('Data', inplace=True)
        df_comparacao.ffill(inplace=True)

    # Hover-texts (como no estrategias.py, para a série Eficiente)
    hover_texts = []
    if not df_comparacao.empty:
        for date in df_comparacao['Data']:
            date_month = pd.to_datetime(date).replace(day=1).normalize()
            vols = volatilidades_mensais.get(date_month, {})
            tooltip = f"<b>Data:</b> {pd.to_datetime(date).strftime('%Y-%m-%d')}<br>"
            if vols:
                tooltip += "<b>Volatilidade:</b><br>"
                for asset, vol in vols.items():
                    tooltip += f"{asset}: {vol:.2f}%<br>"
            else:
                tooltip += "Sem dados de volatilidade"
            hover_texts.append(tooltip)

    # ===============================
    # GRÁFICO PRINCIPAL (exatamente como estrategias.py)
    # ===============================
    main_fig = go.Figure()

    if not df_comparacao.empty:
        main_fig.add_trace(go.Scatter(
            x=df_comparacao['Data'],
            y=df_comparacao['Eficiente'],
            mode='lines',
            name='Carteira Eficiente',
            line=dict(color='blue'),
            hoverinfo='text',
            text=hover_texts
        ))
        main_fig.add_trace(go.Scatter(
            x=df_comparacao['Data'],
            y=df_comparacao['Paridade'],
            mode='lines',
            name='Paridade de Risco',
            line=dict(color='green'),
            hoverinfo='skip'
        ))
        main_fig.add_trace(go.Scatter(
            x=df_comparacao['Data'],
            y=df_comparacao['CDI'],
            mode='lines',
            name='CDI',
            line=dict(color='orange'),
            hoverinfo='skip'
        ))

    main_fig.update_layout(
        title='Comparação: Carteira Eficiente vs Paridade de Risco vs CDI',
        xaxis_title='Data',
        yaxis_title='Valor da Carteira',
        hovermode='x unified',
        template='plotly_white',
        height=460,
        margin=dict(l=40, r=20, t=60, b=40),
        legend=dict(orientation="v", y=1, x=1)
    )

    # ===============================
    # MÉTRICAS (baseadas no gráfico principal e total investido)
    # ===============================
    months = len(pd.date_range(start=pd.to_datetime(start_date), end=pd.to_datetime(end_date), freq='MS'))
    total_investido_val = float(aporte_ini or 0) + float(aporte_mes or 0) * max(0, months - 1)

    def _last_safe(df, col):
        return float(df[col].iloc[-1]) if (not df.empty and col in df.columns and not df[col].dropna().empty) else 0.0

    last_ef  = _last_safe(df_comparacao, 'Eficiente')
    last_pa  = _last_safe(df_comparacao, 'Paridade')
    last_cdi = _last_safe(df_comparacao, 'CDI')

    def pct_return(last_value, invested):
        if invested == 0 or pd.isna(last_value):
            return 0.0
        return (last_value / invested - 1.0) * 100.0

    ret_ef  = pct_return(last_ef,  total_investido_val)
    ret_pa  = pct_return(last_pa,  total_investido_val)
    ret_cdi = pct_return(last_cdi, total_investido_val)

    # Valores finais em R$ para os cards
    valor_ef_str  = f"R$ {last_ef:,.2f}"
    valor_pa_str  = f"R$ {last_pa:,.2f}"
    valor_cdi_str = f"R$ {last_cdi:,.2f}"

    # ===============================
    # ALOCAÇÃO (pizza) e CONTRIBUIÇÃO (barras) — Paridade
    # ===============================
    tmpl = 'plotly_white' if theme == 'light' else 'plotly_dark'
    pie_fig = go.Figure()
    risk_fig = go.Figure()

    if monthly_risk_parity_values:
        last = monthly_risk_parity_values[-1]

        # --- Pizza: pesos-alvo do solver (em %) e R$ no hover
        if 'WeightsTarget' in last:
            w = last['WeightsTarget']
            total_val = float(last.get('TotalValue', 0) or 0)
            alloc_amounts = (w * total_val) if total_val > 0 else last.get('AssetValues', w * 0)

            labels = list(w.index)
            values = (w * 100).values
            customdata = alloc_amounts.reindex(w.index).values

            pie_fig.add_trace(go.Pie(
                labels=labels,
                values=values,
                hole=0.5,
                textinfo='percent',
                insidetextorientation='radial',
                customdata=customdata,
                hovertemplate="<b>%{label}</b><br>%{percent}<br>R$ %{customdata:,.2f}<extra></extra>",
                name="Pesos alvo"
            ))
        else:
            labels = list(last['AssetPercentages'].index)
            values = (last['AssetPercentages'] * 100).values
            customdata = last['AssetValues'].reindex(labels).values if 'AssetValues' in last else values*0
            pie_fig.add_trace(go.Pie(
                labels=labels, values=values, hole=0.5, textinfo='percent',
                insidetextorientation='radial',
                customdata=customdata,
                hovertemplate="<b>%{label}</b><br>%{percent}<br>R$ %{customdata:,.2f}<extra></extra>",
                name="Pesos alvo"
            ))

        pie_fig.update_layout(
            template=tmpl,
            margin=dict(l=20, r=20, t=30, b=20),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            legend=dict(font=dict(color='black'))  # legenda PRETA
        )

        # --- Contribuição de Risco (RC) alvo em %
        rc_key = 'RiskContribTarget'   # (ou 'RiskContribRealized' se quiser a realizada)
        if rc_key in last:
            labels_rc = list(last[rc_key].index)
            values_rc = (last[rc_key] * 100).values
            risk_fig.add_trace(go.Bar(
                x=labels_rc, y=values_rc,
                text=[f"{v:.1f}%" for v in values_rc],
                textposition='auto',
                name="RC (alvo)",                     # garante legenda
                showlegend=True
            ))
            ymax = max(35, float(np.nanmax(values_rc)) + 5) if len(values_rc) else 35
        else:
            labels_rc, values_rc, ymax = [], [], 35

        risk_fig.update_layout(
            template=tmpl,
            xaxis_title=None,
            yaxis_title="% (contribuição de risco)",
            margin=dict(l=20, r=20, t=30, b=20),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(tickfont=dict(color='black')),
            yaxis=dict(tickfont=dict(color='black')),
            legend=dict(font=dict(color='black')),
            font=dict(color=None)  # mantemos eixos/ticks conforme template
        )

    # ===============================
    # TABELA DE DESEMPENHO (métricas simples)
    # ===============================
    table_header = [
        html.Thead(html.Tr([
            html.Th("Métrica", className="fw-bold"),
            html.Th("Eficiente", className="text-end fw-bold"),
            html.Th("Paridade", className="text-end fw-bold"),
            html.Th("CDI", className="text-end fw-bold")
        ], className="border-top-0"))
    ]
    anual_ef = (ret_ef / max(1, months)) * 12
    anual_pa = (ret_pa / max(1, months)) * 12
    anual_cdi = (ret_cdi / max(1, months)) * 12

    table_body = [
        html.Tbody([
            html.Tr([html.Td("Retorno Total"), html.Td(f"{ret_ef:.1f}%", className="text-end"),
                     html.Td(f"{ret_pa:.1f}%", className="text-end"), html.Td(f"{ret_cdi:.1f}%", className="text-end")],
                    className="border-bottom"),
            html.Tr([html.Td("Retorno Anualizado"), html.Td(f"{anual_ef:.1f}%", className="text-end"),
                     html.Td(f"{anual_pa:.1f}%", className="text-end"), html.Td(f"{anual_cdi:.1f}%", className="text-end")],
                    className="border-bottom"),

        ])
    ]
    table = dbc.Table(table_header + table_body, bordered=False, hover=True, responsive=True, striped=True, className="mb-0")

    # ===============================
    # RETORNO DOS COMPONENTES
    # ===============================
    return (
        main_fig,
        f"R$ {total_investido_val:,.2f}",
        f"{ret_ef:.1f}%",
        f"{ret_pa:.1f}%",
        f"{ret_cdi:.1f}%",
        valor_ef_str,
        valor_pa_str,
        valor_cdi_str,
        pie_fig,
        risk_fig,
        table
    )

# =============================================
# RUN
# =============================================
if __name__ == '__main__':
    app.run(debug=True)
