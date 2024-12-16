O arquivo transformers_modified_art2.py é um script Python que implementa um modelo de Transformer para previsão de séries temporais e detecção de anomalias. Abaixo está uma descrição detalhada do conteúdo e das funcionalidades do arquivo:

    Importações:
        Importa bibliotecas necessárias como torch, pandas, matplotlib.pyplot, sklearn.preprocessing, plotly.graph_objects, entre outras.

    Definição da Função de Perda SMAPE:
        Define a classe SMAPELoss que herda de nn.Module e implementa a função de perda SMAPE (Symmetric Mean Absolute Percentage Error).

    Definição do Dataset de Séries Temporais:
        Define a classe TimeSeriesDataset que herda de Dataset e é usada para preparar os dados de séries temporais com uma janela deslizante.

    Definição do Modelo Transformer:
        Define a classe TransformerModel que herda de nn.Module e implementa o modelo Transformer para previsão de séries temporais. Inclui camadas de codificação e decodificação.

    Configuração do Dispositivo:
        Configura o dispositivo para usar GPU se disponível, caso contrário, usa CPU.

    Carregamento e Pré-processamento dos Dados:
        Carrega dados de arquivos CSV, realiza pré-processamento como normalização e conversão para tensores.

    Criação de DataLoaders:
        Cria DataLoader para dividir os dados em lotes e embaralhá-los para treinamento e validação.

    Definição da Função de Perda e Otimizador:
        Define a função de perda como SMAPELoss e o otimizador como Adam.

    Loop de Treinamento:
        Implementa o loop de treinamento do modelo, incluindo a passagem para frente, cálculo da perda, retropropagação e atualização dos pesos.

    Validação do Modelo:
        Implementa a fase de validação para calcular a perda em dados de validação.

    Plotagem das Perdas de Treinamento e Validação:
        Plota as perdas de treinamento e validação ao longo das épocas.

    Salvamento e Carregamento do Modelo Treinado:
        Salva o modelo treinado em um arquivo e fornece código para carregar o modelo salvo.

    Previsão e Detecção de Anomalias:
        Carrega novos dados para previsão, realiza pré-processamento, faz previsões usando o modelo treinado e detecta anomalias com base nas diferenças de erro.

    Plotagem dos Resultados:
        Plota os resultados das previsões e anomalias detectadas.

    Cálculo de Métricas de Desempenho:
        Calcula várias métricas de desempenho como acurácia, precisão, recall, especificidade, F1 score, e AUC.
