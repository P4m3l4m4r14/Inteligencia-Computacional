clc, clear;
% ===== FUNÇÕES ================================================================
function state =  init_rnn_state(batch_size, num_hiddens)
    H = zeros(batch_size, num_hiddens);
    state = {H};
endfunction

function [outputs, state] = rnn(inputs, state, params)
  % Desempacota os parâmetros (assumindo que params é um Cell Array)
  W_xh = params{1};
  W_hh = params{2};
  b_h  = params{3};
  W_hq = params{4};
  b_q  = params{5};

  % Desempacota o estado (assumindo Cell Array)
  H = state{1};

  % Descobrir as dimensões REAIS da entrada
  [num_steps, batch_size, vocab_size] = size(inputs);

  outputs = [];

  % Loop através dos passos de tempo (Time Steps)
  for t = 1:num_steps
      % Pega a fatia de dados do tempo 't'
      X_slice = inputs(t, :, :);
      X = reshape(X_slice, batch_size, vocab_size);

      % Equação da Camada Oculta: tanh(X*W_xh + H*W_hh + b)
      % No Octave, * é multiplicação matricial (igual a np.dot)
      H = tanh((X * W_xh) + (H * W_hh) + b_h);

      % Equação da Saída: Y = H*W_hq + b
      Y = (H * W_hq) + b_q;

      % Concatena o resultado verticalmente (axis=0 no Python)
      outputs = [outputs; Y];
  endfor

  % Retorna outputs e o novo estado (encapsulado em Cell Array)
  state = {H};
endfunction

function grads = grad_clipping(grads, theta)
  % grads: Cell Array contendo as matrizes de gradiente {dW_xh, dW_hh, ...}
  % theta: O valor limite (threshold) para o corte

  % 1. Calcular a Norma L2 Global
  soma_quadrados = 0;

  for i = 1:length(grads)
    % Pega cada matriz de gradiente, eleva ao quadrado e soma tudo
    % O (:) transforma a matriz em um vetor para somar fácil
    soma_quadrados = soma_quadrados + sum(grads{i}(:) .^ 2);
  endfor

  norma_global = sqrt(soma_quadrados);

  % 2. Aplicar o Clipping se necessário
  if norma_global > theta
    % Fator de correção
    scale = theta / norma_global;

    % Multiplica todos os gradientes pelo fator de redução
    for i = 1:length(grads)
      grads{i} = grads{i} * scale;
    endfor

    % Debug para ver se está cortando
    % printf("Gradiente explodiu (Norma: %.2f). Reduzido por %.4f\n", norma_global, scale);
  endif
endfunction

% --- Funções Matemáticas ---
function params = sgd(params, grads, lr, batch_size)
  % Stochastic Gradient Descent
  for i = 1:length(params)
      params{i} = params{i} - (lr * grads{i});
  endfor
endfunction



function [params, perplexity, speed] = train_epoch_ch8(params, X_train, Y_train, lr, batch_size, num_steps, vocab_size)
  % X_train e Y_train: As matrizes gigantes que sairam do carregar_dataset_txt
  % lr: Learning Rate (Taxa de aprendizado)

  % 1. Configurações
  tic(); % Inicia o cronômetro (substitui d2l.Timer)
  num_tokens = size(X_train, 2); % Total de colunas (steps * batches)
  num_batches = floor(num_tokens / num_steps);

  % Descobre num_hiddens olhando para os parametros atuais
  W_xh = params{1};
  num_hiddens = size(W_xh, 2);

  % Inicializa o estado (H) com zeros
  state = init_rnn_state(batch_size, num_hiddens);

  total_loss = 0;
  total_tokens_count = 0;

  % 2. Loop pelos Batches (substitui 'for X, Y in train_iter')
  for i = 1:num_batches
      % --- Fatiamento dos Dados (Slice) ---
      start_col = (i-1) * num_steps + 1;
      end_col   = i * num_steps;

      X_batch = X_train(:, start_col:end_col);
      Y_batch = Y_train(:, start_col:end_col);

      % Transforma X em One-Hot
      X_onehot = to_onehot(X_batch, vocab_size);

      % --- O PASSO MAIS IMPORTANTE (BPTT) ---
      % No Python, isso era dividido em: forward(), loss(), backward().
      % No Octave, faremos tudo numa função matemática manual.
      % Essa função calcula a Loss e os Gradientes para esse batch.
      [grads, loss_val, state] = rnn_bptt(params, X_onehot, Y_batch, state, vocab_size);

      % --- Gradient Clipping (que já criamos) ---
      grads = grad_clipping(grads, 1.0);

      % --- Atualização dos Pesos (SGD) ---
      % Substitui o 'updater'
      params = sgd(params, grads, lr, 1);

      % --- Métricas ---
      % loss_val é a média do erro. Multiplicamos pelo numero de itens para somar o total.
      num_items = numel(Y_batch);
      total_loss = total_loss + (loss_val * num_items);
      total_tokens_count = total_tokens_count + num_items;

      % Opcional: Mostrar progresso a cada 10%
      if mod(i, floor(num_batches/10)) == 0
          printf("Batch %d/%d - Loss atual: %.4f\n", i, num_batches, loss_val);
      endif
  endfor

  tempo_total = toc();
  perplexity = exp(total_loss / total_tokens_count);
  speed = total_tokens_count / tempo_total;

  printf("Epoch finalizada. Perplexidade: %.2f. Velocidade: %.1f tokens/seg\n", perplexity, speed);
endfunction

function params = train_ch8(params, X_train, Y_train, vocab_size, idx_to_char, lr, num_epochs, batch_size, num_steps)
  % params: Cell array com pesos iniciais
  % idx_to_char: Vetor para converter indices de volta para texto (usado na predição)

  % 1. Configurar o Gráfico em Tempo Real
  figure(1);
  clf; % Limpa figura anterior
  set(gcf, 'UserData', 0);

  % Cria um botão que, ao ser clicado, muda 'StopNow' para 1 (Verdadeiro)
  uicontrol('Style', 'pushbutton', 'String', 'PARAR TREINO', ...
            'Position', [10 10 100 30], ...
            'Callback', 'set(gcbf, "UserData", 1)');
  xlabel('Epochs');
  ylabel('Perplexity');
  title('Progresso do Treinamento');
  grid on;
  hold on;

  % Listas para guardar o histórico e plotar
  x_data = [];
  y_data = [];

  % Configuração de Predição
  % Vamos testar a rede gerando texto a partir deste prefixo
  prefixo_teste = 'acampamento meio sangue';
  num_preds = 50;

  printf("Iniciando treinamento por %d épocas...\n", num_epochs);

  % 2. Loop Principal de Treinamento
  for epoch = 1:num_epochs

      % Roda uma época inteira (treina com todos os dados)
      % IMPORTANTE: params é atualizado e retornado aqui
      [params, ppl, speed] = train_epoch_ch8(params, X_train, Y_train, lr, batch_size, num_steps, vocab_size);

      % 3. Atualização Visual e Logs (A cada 10 épocas ou na primeira)
      if mod(epoch, 10) == 0 || epoch == 1

          save("checkpoint_params2.mat", "params");
          % Atualiza gráfico
          x_data = [x_data, epoch];
          y_data = [y_data, ppl];
          plot(x_data, y_data, 'b-o', 'LineWidth', 2);
          drawnow; % Força o Octave a desenhar o gráfico agora

          if get(gcf, 'UserData') == 1
              printf("\n!!! PARADA SOLICITADA PELO USUÁRIO !!!\n");
              printf("Salvando checkpoint antes de sair...\n");
              save("checkpoint_interrompido.mat", "params");
              break; % Quebra o loop for
          endif

          % Mostra status no terminal
          printf("\n--- Epoch %d ---\n", epoch);
          printf("Perplexidade: %.2f | Velocidade: %.1f tokens/sec\n", ppl, speed);

          % Gera um texto de exemplo para vermos a evolução
          texto_gerado = predict_ch8(prefixo_teste, num_preds, params, idx_to_char, vocab_size);
          printf("Teste ('%s'): %s\n", prefixo_teste, texto_gerado);
      endif

  endfor

  printf("\nTreinamento Finalizado!\n");

  texto_final_1 = predict_ch8('acampamento meio sangue', 50, params, idx_to_char, vocab_size);
  texto_final_2 = predict_ch8('meio sangue', 50, params, idx_to_char, vocab_size);

  printf("Final 1: %s\n", texto_final_1);
  printf("Final 2: %s\n", texto_final_2);

endfunction
% ==============================================================================
clear; clc; close all;

% 1. Hiperparâmetros
batch_size = 128;
num_steps = 35;
num_hiddens = 512;
learning_rate = 100; % Taxa alta é comum quando implementamos do zero sem otimizadores complexos (Adam)
num_epochs = 100;

arquivo = 'PJO_e_o_ladrao_de_raios.txt';

% 2. Carregar Dados
printf("Carregando dataset...\n");
[X, Y, vocab_size, idx_to_char] = carregar_dataset_txt(arquivo, batch_size, num_steps);

% 3. Inicializar Modelo
printf("Inicializando pesos...\n");
if exist("checkpoint_params2.mat", "file")
    load("checkpoint_params2.mat"); % Carrega a variável 'params' treinada
    printf("Continuando treinamento de onde parou...\n");
else
    params = get_params(vocab_size, num_hiddens); % Começa do zero
endif

% 4. Iniciar Treinamento
% A função train_ch8 vai abrir o gráfico e mostrar o progresso
params = train_ch8(params, X, Y, vocab_size, idx_to_char, learning_rate, num_epochs, batch_size, num_steps);

