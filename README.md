#EN
The transformers_modified_art2.py script is a Python implementation of a Transformer model for time series forecasting and anomaly detection. Here's a detailed description of the content and functionalities of the script:
Import Statements

    The script imports essential libraries such as torch, pandas, matplotlib.pyplot, sklearn.preprocessing, plotly.graph_objects, and others.

SMAPE Loss Function

    The SMAPELoss class inherits from nn.Module and implements the Symmetric Mean Absolute Percentage Error (SMAPE) loss function.
    This loss function is particularly suited for time series forecasting, as it accounts for both overestimation and underestimation errors.

Time Series Dataset Class

    The TimeSeriesDataset class inherits from Dataset and prepares time series data using a sliding window approach.
    It divides the data into smaller windows for training and prediction purposes, enabling the Transformer model to process sequential data effectively.

Transformer Model Definition

    The TransformerModel class inherits from nn.Module and implements a Transformer architecture for time series forecasting.
    Key Components:
        Embedding Layers: For both the encoder (encoder_embedding) and decoder (decoder_embedding).
        Encoder: Uses TransformerEncoderLayer and TransformerEncoder for multi-layer encoding.
        Decoder: A linear layer maps the encoder's output to the desired prediction size.
    Architecture Parameters:
        input_size: Number of input features.
        output_size: Number of output features.
        window_size: Sliding window size.
        d_model: Model dimension for embeddings.
        nhead: Number of attention heads.
        num_layers: Number of Transformer layers.
    Forward Pass:
        The method receives two data windows (prev_window and current_window).
        These windows are embedded, passed through the encoder, and decoded to predict the next data window.

Device Configuration

    The script automatically configures the device to use a GPU if available, otherwise defaults to CPU.

Data Loading and Preprocessing

    Loading Data: Data is read from CSV files.
    Preprocessing:
        Normalization of data for better model performance.
        Handling missing values (e.g., filling nulls).
        Conversion of data into PyTorch tensors.

DataLoader Creation

    Data is split into training and validation sets using train_test_split.
    DataLoader is used to generate mini-batches of data for training and validation.

Loss Function and Optimizer

    Loss Function: The script uses SMAPELoss to calculate forecasting errors.
    Optimizer: The Adam optimizer is employed with a defined learning rate to update model weights.

Training Loop

    The training process involves:
        Setting the model to training mode (model.train()).
        Iterating through the training data in batches.
        Performing forward passes, computing the loss, backpropagating errors, and updating weights.
    Validation Phase:
        After each training epoch, the model is evaluated on validation data (model.eval()).
        Validation loss is computed to monitor overfitting.
    Loss Tracking:
        Training and validation losses are plotted across epochs for performance analysis.

Model Saving and Loading

    The trained model is saved to a file using torch.save.
    This allows the model to be reloaded later for making predictions on unseen data.

Evaluation and Anomaly Detection
Forecasting on New Data

    Data Preprocessing:
        New data is read, normalized, and converted to tensors.
        Predictions are made using the trained model.
        Predictions are denormalized for comparison with actual values.
    Error Calculation:
        The difference between predictions and actual values is computed.
        Mean and standard deviation of errors are calculated over a sliding window.
    Anomaly Detection:
        A threshold is defined (e.g., three times the standard deviation).
        Errors exceeding the threshold are flagged as anomalies.

Visualization

    The script uses plotly to create interactive plots for:
        Predictions vs. actual values.
        Errors and anomaly detection thresholds.
        Highlighting detected anomalies on the time series data.

Performance Metrics

    The script calculates various metrics to evaluate model performance:
        Accuracy (ACC): Measures the percentage of correct predictions.
        Precision (PR): Evaluates how many of the predicted anomalies are true anomalies.
        Recall (REC): Measures the proportion of actual anomalies that were detected.
        Specificity (SP): Indicates how well the model avoids false alarms.
        F1 Score: Harmonic mean of precision and recall.
        AUC: Area Under the Curve, summarizing the performance across different thresholds.

Summary of Capabilities

    Time Series Forecasting:
        Predicts future values based on historical data.
        Uses Transformer architecture for sequential pattern learning.
    Anomaly Detection:
        Detects deviations in the data using error thresholds.
        Classifies anomalies with high accuracy and precision.
    Visualization:
        Provides clear, interactive visual insights into predictions and anomalies.
    Scalability:
        Supports large datasets with GPU acceleration for faster training.

This implementation showcases a robust solution for time series forecasting and anomaly detection using the powerful Transformer model architecture.

#PTBR
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
