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




Modelo Transformer

    Definição do Modelo:
        O modelo é definido pela classe TransformerModel, que herda de nn.Module.
        Ele possui camadas de embedding para o codificador (encoder_embedding) e o decodificador (decoder_embedding).
        O codificador é composto por uma camada de Transformer (TransformerEncoderLayer) e um TransformerEncoder com múltiplas camadas.
        O decodificador é uma camada linear que ajusta a saída do codificador para o tamanho de saída desejado.

    Arquitetura:
        input_size: Número de características de entrada.
        output_size: Número de características de saída.
        window_size: Tamanho da janela deslizante.
        d_model: Dimensão do modelo.
        nhead: Número de cabeças de atenção.
        num_layers: Número de camadas do Transformer.

    Forward Pass:
        O método forward recebe duas janelas de dados (prev_window e current_window).
        As janelas são passadas pelas camadas de embedding e codificadas pelo Transformer.
        A saída do codificador é decodificada para prever a próxima janela de dados.

Treinamento do Modelo

    Preparação dos Dados:
        Os dados são lidos de arquivos CSV e pré-processados (normalização, preenchimento de valores nulos, etc.).
        Os dados são convertidos em tensores e organizados em janelas deslizantes usando a classe TimeSeriesDataset.

    DataLoader:
        Os dados são divididos em conjuntos de treinamento e validação usando train_test_split.
        DataLoader é utilizado para criar batches de dados para treinamento e validação.

    Função de Perda e Otimizador:
        A função de perda utilizada é a SMAPELoss, que calcula o erro percentual absoluto médio simétrico.
        O otimizador utilizado é o Adam com uma taxa de aprendizado definida.

    Loop de Treinamento:
        O modelo é treinado por um número definido de épocas (num_epochs).
        Em cada época, o modelo é colocado em modo de treinamento (model.train()).
        Para cada batch de dados, o modelo faz a previsão, calcula a perda, realiza a retropropagação e atualiza os pesos.
        Após o treinamento, o modelo é avaliado no conjunto de validação (model.eval()), e a perda de validação é calculada.
        As perdas de treinamento e validação são armazenadas e plotadas ao final do treinamento.

    Salvamento e Carregamento do Modelo:
        O modelo treinado é salvo em um arquivo (torch.save).
        O modelo pode ser carregado posteriormente para fazer previsões em novos dados.

Avaliação e Detecção de Anomalias

    Previsão em Novos Dados:
        Novos dados são lidos, pré-processados e convertidos em tensores.
        O modelo faz previsões sobre os novos dados.
        As previsões são denormalizadas para comparar com os dados reais.

    Cálculo de Erros e Anomalias:
        A diferença entre as previsões e os dados reais é calculada.
        A média e o desvio padrão dos erros são calculados em uma janela deslizante.
        Um limiar é definido para detectar anomalias (e.g., três vezes o desvio padrão).
        Anomalias são identificadas onde os erros excedem o limiar.

    Visualização:
        Os resultados, incluindo as previsões, erros e anomalias, são plotados usando plotly.

Essa implementação permite a previsão de séries temporais e a detecção de anomalias nos dados, utilizando a arquitetura Transformer.
