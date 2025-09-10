import dash
from dash import dcc, html, Input, Output, State
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime
import dash_bootstrap_components as dbc
from dash.exceptions import PreventUpdate

# =============================================
# CONFIGURAÇÃO DO APP
# =============================================

# Inicializar o app com tema moderno
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.LUX])
server = app.server

# Paleta de cores moderna
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

# Estilos personalizados
CUSTOM_STYLES = {
    'card': {
        'borderRadius': '12px',
        'boxShadow': '0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06)',
        'border': 'none'
    },
    'navbar': {
        'boxShadow': '0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06)',
        'padding': '0.75rem 1rem'
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

# =============================================
# LAYOUT DO DASHBOARD
# =============================================

app.layout = html.Div(
    id='main-container',
    children=[
        dcc.Store(id='theme-store', data='dark'),
        
        # Barra de navegação superior moderna
        dbc.Navbar(
            dbc.Container(
                [
                    html.A(
                        dbc.Row(
                            [
                                dbc.Col(html.Img(src="https://cdn-icons-png.flaticon.com/512/2583/2583344.png", height="28px")),
                                dbc.Col(dbc.NavbarBrand("FinDash Pro", className="ms-2 fw-bold")),
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
                                        dbc.Button(
                                            [html.I(className="fas fa-file-export me-2"), "Exportar"],
                                            color="primary",
                                            outline=True,
                                            className="me-2",
                                            style=CUSTOM_STYLES['button']
                                        ),
                                        dbc.Button(
                                            [html.I(className="fas fa-chart-line me-2"), "Relatório"],
                                            color="primary",
                                            style=CUSTOM_STYLES['button']
                                        ),
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
        
        # Conteúdo principal
        dbc.Container(
            fluid=True,
            className="py-3",
            children=[
                # Linha de métricas modernas
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
                            md=3,
                            className="mb-4",
                        ),
                        dbc.Col(
                            dbc.Card(
                                [
                                    dbc.CardHeader("Retorno Eficiente", className="small text-muted fw-bold"),
                                    dbc.CardBody(
                                        [
                                            html.H4("0%", id="retorno-eficiente", className="mb-1 fw-bold"),
                                            html.Small("Melhor desempenho", className="text-muted"),
                                        ],
                                        className="pt-2 pb-3"
                                    ),
                                ],
                                className="h-100",
                                style={**CUSTOM_STYLES['card'], 'borderLeft': f'4px solid {COLORS["dark"]["secondary"]}'}
                            ),
                            md=3,
                            className="mb-4",
                        ),
                        dbc.Col(
                            dbc.Card(
                                [
                                    dbc.CardHeader("Paridade de Risco", className="small text-muted fw-bold"),
                                    dbc.CardBody(
                                        [
                                            html.H4("0%", id="retorno-paridade", className="mb-1 fw-bold"),
                                            html.Small("Risco balanceado", className="text-muted"),
                                        ],
                                        className="pt-2 pb-3"
                                    ),
                                ],
                                className="h-100",
                                style={**CUSTOM_STYLES['card'], 'borderLeft': f'4px solid {COLORS["dark"]["accent"]}'}
                            ),
                            md=3,
                            className="mb-4",
                        ),
                        dbc.Col(
                            dbc.Card(
                                [
                                    dbc.CardHeader("Benchmark (CDI)", className="small text-muted fw-bold"),
                                    dbc.CardBody(
                                        [
                                            html.H4("0%", id="retorno-cdi", className="mb-1 fw-bold"),
                                            html.Small("Referência", className="text-muted"),
                                        ],
                                        className="pt-2 pb-3"
                                    ),
                                ],
                                className="h-100",
                                style={**CUSTOM_STYLES['card'], 'borderLeft': '4px solid #94A3B8'}
                            ),
                            md=3,
                            className="mb-4",
                        ),
                    ],
                    className="g-4",
                ),
                
                # Linha de gráfico e controles
                dbc.Row(
                    [
                        dbc.Col(
                            dbc.Card(
                                [
                                    dbc.CardHeader(
                                        "Parâmetros da Simulação",
                                        className="fw-bold"
                                    ),
                                    dbc.CardBody(
                                        [
                                            dbc.Form(
                                                [
                                                    dbc.Row(
                                                        [
                                                            dbc.Col(
                                                                [
                                                                    dbc.Label("Aporte Inicial (R$)", className="small fw-bold mb-2"),
                                                                    dbc.Input(
                                                                        id="aporte-inicial",
                                                                        type="number",
                                                                        value=10000,
                                                                        min=0,
                                                                        className="mb-3",
                                                                        style={'borderRadius': '8px'}
                                                                    ),
                                                                ],
                                                                md=6,
                                                            ),
                                                            dbc.Col(
                                                                [
                                                                    dbc.Label("Aporte Mensal (R$)", className="small fw-bold mb-2"),
                                                                    dbc.Input(
                                                                        id="aporte-mensal",
                                                                        type="number",
                                                                        value=2000,
                                                                        min=0,
                                                                        className="mb-3",
                                                                        style={'borderRadius': '8px'}
                                                                    ),
                                                                ],
                                                                md=6,
                                                            ),
                                                        ],
                                                        className="mb-3",
                                                    ),
                                                    dbc.Row(
                                                        [
                                                            dbc.Col(
                                                                [
                                                                    dbc.Label("Data Inicial", className="small fw-bold mb-2"),
                                                                    dcc.DatePickerSingle(
                                                                        id='start-date',
                                                                        min_date_allowed=datetime(2000, 1, 1),
                                                                        max_date_allowed=datetime.today(),
                                                                        initial_visible_month=datetime(2015, 1, 1),
                                                                        date=datetime(2015, 1, 1).date(),
                                                                        className="w-100 mb-3",
                                                                        display_format='DD/MM/YYYY',
                                                                        style={'borderRadius': '8px'}
                                                                    ),
                                                                ],
                                                                md=6,
                                                            ),
                                                            dbc.Col(
                                                                [
                                                                    dbc.Label("Data Final", className="small fw-bold mb-2"),
                                                                    dcc.DatePickerSingle(
                                                                        id='end-date',
                                                                        min_date_allowed=datetime(2000, 1, 1),
                                                                        max_date_allowed=datetime.today(),
                                                                        initial_visible_month=datetime.today(),
                                                                        date=datetime.today().date(),
                                                                        className="w-100 mb-3",
                                                                        display_format='DD/MM/YYYY'
                                                                    ),
                                                                ],
                                                                md=6,
                                                            ),
                                                        ],
                                                        className="mb-4",
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
                            md=4,
                            className="mb-4",
                        ),
                        dbc.Col(
                            dbc.Card(
                                [
                                    dbc.CardHeader(
                                        html.Div([
                                            "Desempenho das Estratégias",
                                            dbc.DropdownMenu(
                                                [
                                                    dbc.DropdownMenuItem("1 Ano", id="1y"),
                                                    dbc.DropdownMenuItem("3 Anos", id="3y"),
                                                    dbc.DropdownMenuItem("5 Anos", id="5y"),
                                                    dbc.DropdownMenuItem("Todo Período", id="all"),
                                                ],
                                                label="Período",
                                                color="link",
                                                className="float-end",
                                                style={'fontSize': '0.85rem'}
                                            ),
                                        ]),
                                        className="fw-bold"
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
                                            style={'height': '400px'},
                                        )
                                    ),
                                ],
                                style=CUSTOM_STYLES['card']
                            ),
                            md=8,
                            className="mb-4",
                        ),
                    ],
                    className="g-4",
                ),
                
                # Linha de gráficos secundários
                dbc.Row(
                    [
                        dbc.Col(
                            dbc.Card(
                                [
                                    dbc.CardHeader(
                                        "Alocação de Ativos",
                                        className="fw-bold"
                                    ),
                                    dbc.CardBody(
                                        dcc.Graph(
                                            id='asset-pie-chart',
                                            style={'height': '300px'},
                                            config={'displayModeBar': False}
                                        )
                                    ),
                                ],
                                className="h-100",
                                style=CUSTOM_STYLES['card']
                            ),
                            md=6,
                            className="mb-4",
                        ),
                        dbc.Col(
                            dbc.Card(
                                [
                                    dbc.CardHeader(
                                        "Contribuição de Risco",
                                        className="fw-bold"
                                    ),
                                    dbc.CardBody(
                                        dcc.Graph(
                                            id='risk-bar-chart',
                                            style={'height': '300px'},
                                            config={'displayModeBar': False}
                                        )
                                    ),
                                ],
                                className="h-100",
                                style=CUSTOM_STYLES['card']
                            ),
                            md=6,
                            className="mb-4",
                        ),
                    ],
                    className="g-4",
                ),
                
                # Tabela de desempenho
                dbc.Row(
                    dbc.Col(
                        dbc.Card(
                            [
                                dbc.CardHeader(
                                    "Métricas de Desempenho",
                                    className="fw-bold"
                                ),
                                dbc.CardBody(
                                    html.Div(
                                        id='performance-table',
                                        className="table-responsive",
                                    )
                                ),
                            ],
                            style=CUSTOM_STYLES['card']
                        ),
                        className="mb-4",
                    )
                ),
                
                # Rodapé moderno
                dbc.Row(
                    dbc.Col(
                        html.Div(
                            [
                                html.Hr(className="my-2"),
                                html.P(
                                    [
                                        html.Small(
                                            [
                                                "© 2023 FinDash Pro | ",
                                                html.A("Termos", href="#", className="text-muted me-2"),
                                                html.A("Privacidade", href="#", className="text-muted me-2"),
                                                html.A("Contato", href="#", className="text-muted me-2"),
                                                html.A("Documentação", href="#", className="text-muted"),
                                            ],
                                            className="text-center d-block mb-0"
                                        )
                                    ],
                                    className="mt-3"
                                ),
                            ],
                            className="mt-4",
                        )
                    )
                ),
            ],
        ),
    ],
    style={'minHeight': '100vh', 'fontFamily': "'Inter', sans-serif"},
)

# Adicionando fontes modernas
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
            body {
                font-family: 'Inter', sans-serif;
            }
            .metric-card:hover {
                transform: translateY(-2px);
                transition: all 0.2s ease;
            }
            .dropdown-menu {
                font-size: 0.85rem;
            }
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
    if toggle_value:
        new_theme = 'light'
    else:
        new_theme = 'dark'
    return new_theme, COLORS[new_theme]['toggle_label']

@app.callback(
    Output('main-container', 'style'),
    Input('theme-store', 'data'),
)
def update_theme_style(theme):
    return {
        'backgroundColor': COLORS[theme]['bg'],
        'color': COLORS[theme]['text'],
    }

@app.callback(
    Output('main-graph', 'figure'),
    Output('total-investido', 'children'),
    Output('retorno-eficiente', 'children'),
    Output('retorno-paridade', 'children'),
    Output('retorno-cdi', 'children'),
    Output('asset-pie-chart', 'figure'),
    Output('risk-bar-chart', 'figure'),
    Output('performance-table', 'children'),
    Input('run-simulation', 'n_clicks'),
    Input('theme-store', 'data'),
    State('aporte-inicial', 'value'),
    State('aporte-mensal', 'value'),
    State('start-date', 'date'),
    State('end-date', 'date'),
)
def update_dashboard(n_clicks, theme, aporte_inicial, aporte_mensal, start_date, end_date):
    if n_clicks is None:
        raise PreventUpdate
    
    # Simulação de dados (substituir por suas funções reais)
    dates = pd.date_range(start=start_date, end=end_date, freq='M')
    eficiente_values = np.cumprod(1 + np.random.normal(0.008, 0.05, len(dates))) * aporte_inicial
    paridade_values = np.cumprod(1 + np.random.normal(0.006, 0.03, len(dates))) * aporte_inicial
    cdi_values = np.cumprod(1 + np.random.normal(0.004, 0.01, len(dates))) * aporte_inicial
    
    # Adicionar aportes mensais
    for i in range(1, len(dates)):
        eficiente_values[i:] += aporte_mensal
        paridade_values[i:] += aporte_mensal
        cdi_values[i:] += aporte_mensal
    
    # Calcular métricas
    total_investido = aporte_inicial + aporte_mensal * (len(dates) - 1)
    retorno_eficiente = (eficiente_values[-1] / total_investido - 1) * 100
    retorno_paridade = (paridade_values[-1] / total_investido - 1) * 100
    retorno_cdi = (cdi_values[-1] / total_investido - 1) * 100
    
    # Gráfico principal moderno
    main_fig = go.Figure()
    
    main_fig.add_trace(go.Scatter(
        x=dates,
        y=eficiente_values,
        name='Estratégia Eficiente',
        line=dict(color=COLORS[theme]['primary'], width=3),
        hovertemplate='<b>%{x|%b %Y}</b><br>R$ %{y:,.2f}<extra></extra>',
        fill='tozeroy',
        fillcolor=f'{COLORS[theme]["primary"]}20'
    ))
    
    main_fig.add_trace(go.Scatter(
        x=dates,
        y=paridade_values,
        name='Paridade de Risco',
        line=dict(color=COLORS[theme]['secondary'], width=3),
        hovertemplate='<b>%{x|%b %Y}</b><br>R$ %{y:,.2f}<extra></extra>',
        fill='tozeroy',
        fillcolor=f'{COLORS[theme]["secondary"]}20'
    ))
    
    main_fig.add_trace(go.Scatter(
        x=dates,
        y=cdi_values,
        name='CDI',
        line=dict(color=COLORS[theme]['accent'], width=2, dash='dot'),
        hovertemplate='<b>%{x|%b %Y}</b><br>R$ %{y:,.2f}<extra></extra>'
    ))
    
    main_fig.update_layout(
        template='plotly_white' if theme == 'light' else 'plotly_dark',
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(size=12)
        ),
        margin=dict(l=20, r=20, t=40, b=20),
        hovermode="x unified",
        xaxis_title=None,
        yaxis_title="Valor do Portfólio (R$)",
        yaxis=dict(
            gridcolor='rgba(200, 200, 200, 0.2)' if theme == 'dark' else 'rgba(0, 0, 0, 0.1)',
            tickformat=",.0f"
        ),
        xaxis=dict(
            gridcolor='rgba(200, 200, 200, 0.2)' if theme == 'dark' else 'rgba(0, 0, 0, 0.1)',
            showgrid=False
        )
    )
    
    # Gráfico de pizza de alocação moderno
    assets = ['Ações', 'Renda Fixa', 'Fundos', 'ETFs', 'Internacional']
    weights = np.random.dirichlet(np.ones(5), size=1)[0]
    
    pie_fig = go.Figure()
    pie_fig.add_trace(go.Pie(
        labels=assets,
        values=weights*100,
        marker_colors=[COLORS[theme]['primary'], COLORS[theme]['secondary'], 
                      COLORS[theme]['accent'], '#7C4DFF', '#FF4081'],
        hole=0.5,
        textinfo='percent',
        textposition='inside',
        insidetextorientation='radial',
        hoverinfo='label+percent+value',
        textfont=dict(size=12)
    ))
    
    pie_fig.update_layout(
        showlegend=True,
        legend=dict(
            orientation="v",
            yanchor="middle",
            xanchor="right",
            x=1.2,
            font=dict(size=11)
        ),
        margin=dict(l=20, r=20, t=30, b=20),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    
    # Gráfico de barras de risco moderno
    risk_fig = go.Figure()
    risk_fig.add_trace(go.Bar(
        x=assets,
        y=np.random.randint(10, 30, size=5),
        marker_color=COLORS[theme]['secondary'],
        name='Contribuição',
        text=np.random.randint(10, 30, size=5),
        texttemplate='%{text}%',
        textposition='auto'
    ))
    
    risk_fig.update_layout(
        xaxis_title=None,
        yaxis_title="% de Risco",
        margin=dict(l=20, r=20, t=30, b=20),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        yaxis=dict(
            gridcolor='rgba(200, 200, 200, 0.2)' if theme == 'dark' else 'rgba(0, 0, 0, 0.1)',
            range=[0, 35]
        ),
        xaxis=dict(
            gridcolor='rgba(200, 200, 200, 0.2)' if theme == 'dark' else 'rgba(0, 0, 0, 0.1)'
        )
    )
    
    # Tabela de desempenho moderna
    table_header = [
        html.Thead(html.Tr([
            html.Th("Métrica", className="fw-bold"),
            html.Th("Eficiente", className="text-end fw-bold"),
            html.Th("Paridade", className="text-end fw-bold"),
            html.Th("CDI", className="text-end fw-bold")
        ], className="border-top-0"))
    ]
    
    table_body = [
        html.Tbody([
            html.Tr([
                html.Td("Retorno Total"),
                html.Td(f"{retorno_eficiente:.1f}%", className="text-end"),
                html.Td(f"{retorno_paridade:.1f}%", className="text-end"),
                html.Td(f"{retorno_cdi:.1f}%", className="text-end")
            ], className="border-bottom"),
            html.Tr([
                html.Td("Retorno Anualizado"),
                html.Td(f"{retorno_eficiente/len(dates)*12:.1f}%", className="text-end"),
                html.Td(f"{retorno_paridade/len(dates)*12:.1f}%", className="text-end"),
                html.Td(f"{retorno_cdi/len(dates)*12:.1f}%", className="text-end")
            ], className="border-bottom"),
            html.Tr([
                html.Td("Volatilidade"),
                html.Td("15.2%", className="text-end"),
                html.Td("10.5%", className="text-end"),
                html.Td("3.8%", className="text-end")
            ], className="border-bottom"),
            html.Tr([
                html.Td("Índice Sharpe"),
                html.Td("1.25", className="text-end text-success fw-bold"),
                html.Td("1.08", className="text-end text-success fw-bold"),
                html.Td("0.45", className="text-end")
            ])
        ])
    ]
    
    table = dbc.Table(
        table_header + table_body,
        bordered=False,
        hover=True,
        responsive=True,
        striped=True,
        className="mb-0"
    )
    
    return (
        main_fig,
        f"R$ {total_investido:,.2f}",
        f"{retorno_eficiente:.1f}%",
        f"{retorno_paridade:.1f}%",
        f"{retorno_cdi:.1f}%",
        pie_fig,
        risk_fig,
        table
    )

if __name__ == '__main__':
    app.run(debug=True)