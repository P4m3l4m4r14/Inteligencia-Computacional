% +=======================================================================================
% TRABALHO 2
% Aluna: Pâmela Maria Pontes Frota
% Matrícula: 554556
% Professor: Jarbas Joaci
% Disciplina: Inteligência Computacional
% =======================================================================================
% Questão 01: Classifique o conjunto de dados Vertebral Column Data set (disponível em
%(https://archive.ics.uci.edu/ml/datasets/Vertebral+Column) em três classes (normal, disk
%hernia e spondilolysthesis) usando uma rede neural RBF. Utilize a estratégia de
%validação hold-out (70% das amostras para treino e o restante para teste) e efetue 10
%xecuções (permutar as amostras do conjunto de dados em cada execução). O resultado
%deve ser a acurácia média.
%OBS: Deixei comentários para facilitar a orientação ao longo do código.
% ========================================================================================

clear; clc; close all; pkg load statistics;

% ===== FUNÇÕES ================================================================
function C = kmeans(X, k)
    n = size(X, 1);
    rand_idx = randperm(n);
    C = X(rand_idx(1:k), :);

    for iter = 1:50
        D = pdist2(X, C);
        [~, idx] = min(D, [], 2);

        C_old = C;

        for i = 1:k
            pontos_no_cluster = X(idx == i, :);
            if ~isempty(pontos_no_cluster)
                C(i, :) = mean(pontos_no_cluster, 1);
            end
        end

        diferenca = abs(C - C_old);
        if max(diferenca(:)) < 1e-4
         break;
        end
    end
end

function [C, sigma, W] = treino_rede(X_train, y_train, num_neuronios)
    %3.1. CONFIGURAÇÃO DA REDE
   k = num_neuronios;
   % --- Escolha dos Centros ---
   C = kmeans(X_train, num_neuronios);

   % --- Definir Sigma ---
   if k > 1
    dist_center = pdist(C);
    sigma = mean(dist_center) * 1.5;
   else
    sigma = 1;
   end

   % --- Calcular a Matriz de Ativação ---
   r_train = pdist2(X_train, C); % r é o raio definido pela distância entre X e C

   phi = exp(-(r_train.^2)/(2 * sigma.^2));

   % Adicionar Bias
   phi = [phi, ones(size(phi, 1), 1)];

   % 3.2. TREINAMENTO DA CAMADA DE SAÍDA
   n_classes = 3; %número de classificações exigido na questão

   % Codificação One-Hot
   Y_target = zeros(length(y_train), n_classes);
   idx_linear = sub2ind(size(Y_target), (1:length(y_train))', y_train);
   Y_target(idx_linear) = 1;

   % --- Calcular os Pesos ---
   W = pinv(phi) * Y_target;
end

function acuracia = avaliar_rbf(X_test, y_test, C, sigma, W)
    r_test = pdist2(X_test, C);

    phi_test = exp(-(r_test.^2)/(2 * sigma.^2));
    phi_test = [phi_test, ones(size(phi_test, 1),1)];

    % Calcular a saída da rede
    Y_scores = phi_test * W;

    % Decodificar
    [~, y_pred] = max(Y_scores, [], 2);

    %calcular acuracia
    acertos = sum(y_pred == y_test);
    acuracia = acertos/length(y_test);
end
% ==============================================================================

% 1. CARREGAMENTO E PREPARAÇÃO DOS DADOS
caminho_script = fileparts(mfilename('fullpath'));
cd(caminho_script);

nome_arquivo = 'column_3C.dat';
data = importdata(nome_arquivo);

if isstruct(data)
    X = data.data;
    labels = data.textdata;
    if length(labels) > size(X, 1)
       labels = labels(end-size(X,1)+1:end, :);
    end
else
    error('Não foi possível separar números de texto automaticamente.');
end

y = zeros(size(labels));

y(strcmp(labels, 'DH')) = 1;  % DH vira 1
y(strcmp(labels, 'SL')) = 2;  % SL vira 2
y(strcmp(labels, 'NO')) = 3;  % NO vira 3

% --- Normalização---
mu = mean(X);
sigma_data = std(X);
sigma_data(sigma_data == 0) = 1;

X = (X - mu) ./ sigma_data;

% Definir tamanho do treino (70%)
num_execucoes = 10;
num_neuronios = 20;
num_train = round(0.70 * size(X, 1));

acuracias = zeros(10, 1); % Vetor para guardar os resultados

fprintf('Iniciando as %d execuções...\n', num_execucoes);

for i = 1: num_execucoes
    % 2 HOLD-OUT
    ind_rand = randperm(size(X, 1));
    idx_train = ind_rand(i:num_train);
    idx_test = ind_rand(num_train+1:end);

    X_train = X(idx_train, :);
    y_train = y(idx_train, :);

    X_test  = X(idx_test, :);
    y_test  = y(idx_test, :);

    % 3. CONFIGURAÇÃO DA REDE RBF E TREINAMENTO
    [C, sigma, W] = treino_rede(X_train, y_train, num_neuronios);

   disp('Treinamento concluído.');

    % 4. TESTE E AVALIAÇÃO
    acc = avaliar_rbf(X_test, y_test, C, sigma, W);

    % --- Armazenamento ---
    accuracies(i) = acc;
    fprintf('Execução %d: Acurácia = %.2f%%\n', i, acc*100);
end

% 5. VISUALIZAÇÃO
fprintf('-----------------------------\n');
fprintf('Acurácia Média: %.4f%%\n', mean(accuracies)*100);
fprintf('Desvio Padrão:  %.4f\n', std(accuracies));

