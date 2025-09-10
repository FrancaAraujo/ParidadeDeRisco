import pandas as pd
from datetime import datetime, timedelta
import yfinance as yf

# Lista completa de ativos brasileiros (precisa de .SA)
BR_TICKERS = [
    'ALOS3', 'ABEV3', 'ASAI3', 'AURE3', 'AMOB3', 'AZUL4', 'AZZA3', 'B3SA3', 'BBSE3', 
    'BBDC3', 'BBDC4', 'BRAP4', 'BBAS3', 'BRKM5', 'BRAV3', 'BRFS3', 'BPAC11', 'CXSE3', 
    'CRFB3', 'CCRO3', 'CMIG4', 'COGN3', 'CPLE6', 'CSAN3', 'CPFE3', 'CMIN3', 'CVCB3', 
    'CYRE3', 'ELET3', 'ELET6', 'EMBR3', 'ENGI11', 'ENEV3', 'EGIE3', 'EQTL3', 'FLRY3', 
    'GGBR4', 'GOAU4', 'NTCO3', 'HAPV3', 'HYPE3', 'IGTI11', 'IRBR3', 'ISAE4', 'ITSA4', 
    'ITUB4', 'JBSS3', 'KLBN11', 'RENT3', 'LREN3', 'LWSA3', 'MGLU3', 'POMO4', 'MRFG3', 
    'BEEF3', 'MRVE3', 'MULT3', 'PCAR3', 'PETR3', 'PETR4', 'RECV3', 'PRIO3', 'PETZ3', 
    'PSSA3', 'RADL3', 'RAIZ4', 'RDOR3', 'RAIL3', 'SBSP3', 'SANB11', 'STBP3', 'SMTO3', 
    'CSNA3', 'SLCE3', 'SUZB3', 'TAEE11', 'VIVT3', 'TIMS3', 'TOTS3', 'UGPA3', 'USIM5', 
    'VALE3', 'VAMO3', 'VBBR3', 'VIVA3', 'WEGE3', 'YDUQ3'
]

def atualizar_cotacoes(file_path):
    # Carregar o arquivo CSV
    tabela = pd.read_csv(file_path)
    
    # Verificar coluna Data
    if 'Data' not in tabela.columns:
        raise ValueError("O arquivo CSV não possui uma coluna chamada 'Data'")
    
    # Extrair dados
    ultima_linha = tabela.iloc[-1]
    data_ultima = "02/01/2018 16:56:00"
    colunas_ativos = tabela.columns[1:]  # Excluir a coluna Data
    
    # Preparar tickers formatados
    ativos_formatados = []
    for ativo in colunas_ativos:
        if ativo in BR_TICKERS:
            ativos_formatados.append(ativo + '.SA')  # Adiciona .SA para brasileiros
        else:
            ativos_formatados.append(ativo)  # Mantém original para S&P 500
    
    # Converter data
    try:
        data_datetime = datetime.strptime(data_ultima.split()[0], '%d/%m/%Y')
        data_inicio = (data_datetime + timedelta(days=1)).strftime('%Y-%m-%d')
    except:
        # Caso a data já esteja em formato diferente
        data_inicio = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
    
    data_fim = datetime.now().strftime('%Y-%m-%d')
    
    # Baixar cotações em lotes
    batch_size = 50  # Yahoo Finance tem limite de tickers por requisição
    dados_novos = pd.DataFrame()
    
    for i in range(0, len(ativos_formatados), batch_size):
        batch = ativos_formatados[i:i + batch_size]
        try:
            dados_batch = yf.download(
                batch,
                start=data_inicio,
                end=data_fim,
                interval="1d",
            )['Close']
            
            # Corrigir formato quando há apenas 1 ativo no batch
            if len(batch) == 1:
                dados_batch = pd.DataFrame(dados_batch)
                dados_batch.columns = [batch[0]]
            
            dados_novos = pd.concat([dados_novos, dados_batch], axis=1)
        except Exception as e:
            print(f"Erro ao baixar lote {i//batch_size + 1}: {e}")
            continue
    
    # Processar os dados
    if not dados_novos.empty:
        # Remover sufixos .SA dos nomes das colunas
        dados_novos.columns = [col.replace('.SA', '') for col in dados_novos.columns]
        
        # Formatar datas
        dados_novos.index = dados_novos.index.strftime('%d/%m/%Y %H:%M:%S')
        
        # Preencher valores ausentes
        dados_novos = dados_novos.ffill().bfill()
        
        # Formatar números com 2 casas decimais e vírgula
        for col in dados_novos.columns:
            dados_novos[col] = dados_novos[col].apply(
                lambda x: f"{x:.2f}".replace('.', ',') if pd.notnull(x) else ""
            )
        
        # Resetar índice e renomear coluna de data
        dados_novos = dados_novos.reset_index().rename(columns={'Date': 'Data'})
        
        # Concatenar com dados originais
        tabela_atualizada = pd.concat([tabela, dados_novos], ignore_index=True)
        
        # Salvar arquivo
        tabela_atualizada.to_csv(file_path, index=False, sep=',', header=True, quoting=0)
        print("Cotações atualizadas com sucesso!")
        return True
    else:
        print("Nenhum dado novo foi encontrado.")
        return False

# Uso do código
if __name__ == "__main__":
    file_path = 'arquivos/Dados_Ativos_B3.csv'
    sucesso = atualizar_cotacoes(file_path)
    
    if sucesso:
        print("Arquivo atualizado com sucesso!")
    else:
        print("Houve um problema na atualização.")