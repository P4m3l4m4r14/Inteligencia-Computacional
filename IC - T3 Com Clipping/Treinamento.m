clc; clear; close all;

T = 1000;
time = 1:T;

sinal_limpo = sin(0.01 * time);
ruido = 0.05 * randn(1, T);
x = sinal_limpo + ruido;

figure;
plot(time, x);
title('Série Temporal: Seno com Ruído');
xlabel('Tempo (time)');
ylabel('Valor (x)');
grid on;
xlim([1 T]);

tau = 4;
n_amostras = T-tau;
features = zeros(n_amostras, tau);

for i = 1:tau

    features(:, i) = x(i : n_amostras + i - 1)';
end

labels = x(tau+1:end)';

n_train = 600;

features_train = features(1:n_train, :);
labels_train = labels(1:n_train, :);


% --- 1. Definição da Arquitetura (O equivalente ao nn.Sequential) ---
input_size = tau;    % 4 (definido no código anterior)
hidden_size = 10;    % Primeira camada (nn.Dense(10))
output_size = 1;     % Segunda camada (nn.Dense(1))

% --- 2. Inicialização Xavier (net.initialize(init.Xavier())) ---
% A inicialização Xavier sorteia pesos de uma distribuição uniforme
% ajustada pelo tamanho das entradas e saídas para evitar gradientes explosivos.

% Limite para W1: sqrt(6 / (entrada + saida))
limit1 = sqrt(6 / (input_size + hidden_size));
W1 = (rand(input_size, hidden_size) * 2 * limit1) - limit1;
b1 = zeros(1, hidden_size); % Bias inicia com 0

% Limite para W2: sqrt(6 / (entrada + saida))
limit2 = sqrt(6 / (hidden_size + output_size));
W2 = (rand(hidden_size, output_size) * 2 * limit2) - limit2;
b2 = zeros(1, output_size);

% Empacotando tudo numa estrutura 'net' para ficar organizado
net.W1 = W1;
net.b1 = b1;
net.W2 = W2;
net.b2 = b2;

% --- 3. Definindo a Função Forward (Como a rede processa os dados) ---
function y_pred = forward_pass(X, net)
    % Camada 1: Entrada -> Oculta
    z1 = X * net.W1 + net.b1; % Multiplicação de matriz + bias

    % Ativação ReLU (activation='relu')
    a1 = max(0, z1);          % Zera tudo que for negativo

    % Camada 2: Oculta -> Saída (nn.Dense(1))
    z2 = a1 * net.W2 + net.b2;

    % Saída final (sem ativação para regressão)
    y_pred = z2;
end

% --- 4. Definindo a Perda (gluon.loss.L2Loss) ---
% L2Loss é basicamente o Erro Quadrático Médio (MSE) multiplicado por 0.5
function loss_val = l2_loss(y_pred, y_real)
    diff = y_pred - y_real;
    loss_val = 0.5 * mean(diff .^ 2);
end

% --- Parâmetros de Treinamento ---
epochs = 5;
lr = 0.01;  % Taxa de aprendizado (Learning Rate)
batch_size = 16;
n_train = 600;

% Loop das Épocas (Quantas vezes a rede vê o dataset inteiro)
for epoch = 1:epochs

    % Calcular quantos batches temos
    num_batches = floor(n_train / batch_size);
    epoch_loss = 0; % Acumulador de erro para mostrar no print

    % Loop dos Batches (Iterar sobre os dados)
    for b = 1:num_batches
        % 1. Pegar o batch atual
        idx_inicio = (b-1) * batch_size + 1;
        idx_fim = b * batch_size;

        X = features_train(idx_inicio:idx_fim, :);
        y = labels_train(idx_inicio:idx_fim, :);

        % --- PASSO 1: FORWARD PASS (Calcular a previsão) ---
        % Precisamos guardar os valores intermediários (z1, a1) para o backprop
        z1 = X * net.W1 + net.b1;     % Entrada -> Oculta
        a1 = max(0, z1);              % ReLU
        z2 = a1 * net.W2 + net.b2;    % Oculta -> Saída
        y_pred = z2;

        % Acumular erro para visualização (L2 Loss)
        loss_val = 0.5 * mean((y_pred - y).^2);
        epoch_loss = epoch_loss + loss_val;

        % --- PASSO 2: BACKPROPAGATION (Calcular os gradientes) ---
        % Aqui calculamos a derivada do erro em relação a cada peso.
        % Regra da Cadeia: dL/dW = dL/dy_pred * dy_pred/dW

        % A. Gradiente na Saída (Derivada do MSE: y_pred - y)
        delta2 = (y_pred - y);  % (batch_size x 1)

        % Gradientes para W2 e b2
        dW2 = a1' * delta2;             % (hidden x 1)
        db2 = sum(delta2, 1);           % Soma os erros do batch para o bias

        % B. Gradiente na Camada Oculta (Voltando para trás)
        % Propaga o erro através de W2
        delta1_pre = delta2 * net.W2';  % (batch_size x hidden)

        % Derivada da ReLU: 1 se x > 0, 0 se x <= 0
        delta1 = delta1_pre .* (z1 > 0);

        % Gradientes para W1 e b1
        dW1 = X' * delta1;              % (input x hidden)
        db1 = sum(delta1, 1);

        % --- PASSO 3: OTIMIZAÇÃO (Atualizar os pesos) ---
        % SGD: Peso_novo = Peso_velho - lr * (gradiente / tamanho_batch)
        % Dividimos pelo batch_size para tirar a média do gradiente

        net.W2 = net.W2 - lr * (dW2 / batch_size);
        net.b2 = net.b2 - lr * (db2 / batch_size);
        net.W1 = net.W1 - lr * (dW1 / batch_size);
        net.b1 = net.b1 - lr * (db1 / batch_size);
    end

    % Mostrar resultado da época
    avg_loss = epoch_loss / num_batches;
    fprintf('Epoch %d, Loss: %f\n', epoch, avg_loss);
end

% --- Visualizar o Resultado Final ---
figure;
% Previsão em todo o conjunto (incluindo o que não foi treinado)
z1 = features * net.W1 + net.b1;
a1 = max(0, z1);
y_final = a1 * net.W2 + net.b2;

plot(1:T, x, 'b', 'DisplayName', 'Original (Com Ruído)'); hold on;
% Ajuste do índice: features começam em 'tau', então plotamos deslocado
plot((tau+1):T, y_final, 'r', 'LineWidth', 2, 'DisplayName', 'Previsão da MLP');
legend;
title('Resultado do Treinamento da MLP');

