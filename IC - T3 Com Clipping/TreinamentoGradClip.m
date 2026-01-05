clc; clear; close all;

T = 1000;
time = 1:T;

sinal_limpo = sin(0.01 * time);
ruido = 0.05 * randn(1, T);
x = sinal_limpo + ruido;

figure;
plot(time, x);
title('Sˆmrie Temporal: Seno com Ruˆqdo');
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


% --- 1. Definic~ao da Arquitetura (O equivalente ao nn.Sequential) ---
input_size = tau;    % 4 (definido no cˆudigo anterior)
hidden_size = 10;    % Primeira camada (nn.Dense(10))
output_size = 1;     % Segunda camada (nn.Dense(1))

% --- 2. Inicializac~ao Xavier (net.initialize(init.Xavier())) ---
% A inicializac~ao Xavier sorteia pesos de uma distribuic~ao uniforme
% ajustada pelo tamanho das entradas e saˆqdas para evitar gradientes explosivos.

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

% --- 3. Definindo a Func~ao Forward (Como a rede processa os dados) ---
function y_pred = forward_pass(X, net)
    % Camada 1: Entrada -> Oculta
    z1 = X * net.W1 + net.b1; % Multiplicac~ao de matriz + bias

    % Ativac~ao ReLU (activation='relu')
    a1 = max(0, z1);          % Zera tudo que for negativo

    % Camada 2: Oculta -> Saˆqda (nn.Dense(1))
    z2 = a1 * net.W2 + net.b2;

    % Saˆqda final (sem ativac~ao para regress~ao)
    y_pred = z2;
end

% --- 4. Definindo a Perda (gluon.loss.L2Loss) ---
% L2Loss ˆm basicamente o Erro Quadrˆhtico Mˆmdio (MSE) multiplicado por 0.5
function loss_val = l2_loss(y_pred, y_real)
    diff = y_pred - y_real;
    loss_val = 0.5 * mean(diff .^ 2);
end

% --- Par^ametros de Treinamento ---
epochs = 5;
lr = 0.01;  % Taxa de aprendizado (Learning Rate)
batch_size = 16;
n_train = 600;

% Loop das ˆ[pocas (Quantas vezes a rede vˆ§ o dataset inteiro)
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

        % --- PASSO 1: FORWARD PASS (Calcular a previs~ao) ---
        % Precisamos guardar os valores intermediˆhrios (z1, a1) para o backprop
        z1 = X * net.W1 + net.b1;     % Entrada -> Oculta
        a1 = max(0, z1);              % ReLU
        z2 = a1 * net.W2 + net.b2;    % Oculta -> Saˆqda
        y_pred = z2;

        % Acumular erro para visualizac~ao (L2 Loss)
        loss_val = 0.5 * mean((y_pred - y).^2);
        epoch_loss = epoch_loss + loss_val;

        % --- PASSO 2: BACKPROPAGATION (Calcular os gradientes) ---
        % Aqui calculamos a derivada do erro em relac~ao a cada peso.
        % Regra da Cadeia: dL/dW = dL/dy_pred * dy_pred/dW

        % A. Gradiente na Saˆqda (Derivada do MSE: y_pred - y)
        delta2 = (y_pred - y);  % (batch_size x 1)

        % Gradientes para W2 e b2
        dW2 = a1' * delta2;             % (hidden x 1)
        db2 = sum(delta2, 1);           % Soma os erros do batch para o bias

        % B. Gradiente na Camada Oculta (Voltando para trˆhs)
        % Propaga o erro atravˆms de W2
        delta1_pre = delta2 * net.W2';  % (batch_size x hidden)

        % Derivada da ReLU: 1 se x > 0, 0 se x <= 0
        delta1 = delta1_pre .* (z1 > 0);

        % Gradientes para W1 e b1
        dW1 = X' * delta1;              % (input x hidden)
        db1 = sum(delta1, 1);

        % --- Cˆ_DIGO DE GRADIENT CLIPPING (Inserir apˆus o Backprop) ---

        % 1. Definir o limite (theta)
         theta = 1.0; % Valor comum, pode variar entre 1 e 5

        % 2. Calcular a "Norma Global" (A magnitude total do vetor de gradientes)
        % Soma dos quadrados de TODOS os gradientes da rede
        soma_quadrados = sum(sum(dW1.^2)) + sum(sum(db1.^2)) + ...
                 sum(sum(dW2.^2)) + sum(sum(db2.^2));

        norma_global = sqrt(soma_quadrados);

        if mod(b, 10) == 0 % Mostra a cada 10 batches para n~ao poluir
          fprintf('Norma atual: %.2f | Theta: %.2f\n', norma_global, theta);
        end

        % 3. Verificar se explodiu e cortar
        if norma_global > theta
          % Fator de escala: quanto precisamos diminuir?
          scale = theta / norma_global;

          % Aplicar a reduc~ao em todos os gradientes
          dW1 = dW1 * scale;
          db1 = db1 * scale;
          dW2 = dW2 * scale;
          db2 = db2 * scale;

          % Opcional: Mostrar aviso se quiser ver acontecendo
          % fprintf('Gradiente explodiu (Norma: %.2f). Cortado!\n', norma_global);
        end

        % --- PASSO 3: OTIMIZAC~AO (Atualizar os pesos) ---
        % SGD: Peso_novo = Peso_velho - lr * (gradiente / tamanho_batch)
        % Dividimos pelo batch_size para tirar a mˆmdia do gradiente

        net.W2 = net.W2 - lr * (dW2 / batch_size);
        net.b2 = net.b2 - lr * (db2 / batch_size);
        net.W1 = net.W1 - lr * (dW1 / batch_size);
        net.b1 = net.b1 - lr * (db1 / batch_size);

    end

    % Mostrar resultado da ˆmpoca
    avg_loss = epoch_loss / num_batches;
    fprintf('Epoch %d, Loss: %f\n', epoch, avg_loss);
end

% --- Visualizar o Resultado Final ---
figure;
% Previs~ao em todo o conjunto (incluindo o que n~ao foi treinado)
z1 = features * net.W1 + net.b1;
a1 = max(0, z1);
y_final = a1 * net.W2 + net.b2;

plot(1:T, x, 'b', 'DisplayName', 'Original (Com Ruˆqdo)'); hold on;
% Ajuste do ˆqndice: features comecam em 'tau', ent~ao plotamos deslocado
plot((tau+1):T, y_final, 'r', 'LineWidth', 2, 'DisplayName', 'Previs~ao da MLP');
legend;
title('Resultado do Treinamento da MLP');

