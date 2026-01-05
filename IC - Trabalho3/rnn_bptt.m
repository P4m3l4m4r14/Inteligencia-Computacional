function [grads, loss_mean, last_state] = rnn_bptt(params, X_onehot, Y_batch, state, vocab_size)
  % Desempacotar pesos
  W_xh = params{1}; W_hh = params{2}; b_h = params{3};
  W_hq = params{4}; b_q = params{5};

  H = state{1};
  [num_steps, batch_size, ~] = size(X_onehot);

  % =========================================================
  % 1. FORWARD PASS (Guardando histórico)
  % =========================================================
  % Precisamos salvar os H de cada passo para usar na derivada depois.
  % h_history{1} é o H inicial. h_history{t+1} é o H do tempo t.
  h_history = cell(num_steps + 1, 1);
  h_history{1} = H;

  % Também guardamos as previsões (y_hat) para calcular o erro
  y_hat_history = cell(num_steps, 1);

  loss_total = 0;

  for t = 1:num_steps
      % Pega input atual: reshape de (1, batch, vocab) para (batch, vocab)
      X_t = reshape(X_onehot(t, :, :), batch_size, vocab_size);

      % RNN Fórmula
      H = tanh((X_t * W_xh) + (H * W_hh) + b_h);
      O = (H * W_hq) + b_q;

      % Softmax (estável)
      % Subtraímos o max para evitar overflow exponencial
      O_safe = O - max(O, [], 2);
      y_prob = exp(O_safe) ./ sum(exp(O_safe), 2);

      % Salvar no histórico
      h_history{t+1} = H;
      y_hat_history{t} = y_prob;

      % -- Cálculo da Loss (Cross Entropy) para este passo --
      % Precisamos pegar a probabilidade apenas da letra correta (Target)
      % Y_batch está transposto em relação ao tempo, vamos ajustar: Y_batch(batch, step)
      targets = Y_batch(:, t); % vetor de indices (batch_size x 1)

      % Truque de indexação linear para pegar os valores corretos da matriz y_prob
      % Isso evita criar loops for lentos
      indices_lineares = sub2ind(size(y_prob), (1:batch_size)', targets);
      prob_correta = y_prob(indices_lineares);

      % Loss = -log(probabilidade da classe certa)
      loss_total = loss_total - sum(log(prob_correta + 1e-10)); % +1e-10 evita log(0)
  endfor

  loss_mean = loss_total / (num_steps * batch_size);
  last_state = {H};

  % =========================================================
  % 2. BACKWARD PASS (Derivadas)
  % =========================================================
  % Inicializar gradientes com zeros
  dW_xh = zeros(size(W_xh));
  dW_hh = zeros(size(W_hh));
  db_h  = zeros(size(b_h));
  dW_hq = zeros(size(W_hq));
  db_q  = zeros(size(b_q));

  % Gradiente do estado oculto "próximo" (começa com zero no último passo)
  dh_next = zeros(batch_size, size(W_hh, 1));

  % Loop REVERSO (Do tempo T até 1)
  for t = num_steps:-1:1
      % Recuperar dados desse tempo
      X_t = reshape(X_onehot(t, :, :), batch_size, vocab_size);
      y_prob = y_hat_history{t};
      H_curr = h_history{t+1}; % H deste passo
      H_prev = h_history{t};   % H do passo anterior

      % -- Derivada da Saída (Softmax + CrossEntropy) --
      % O gradiente da Loss em relação à saída (O) é simplesmente (Prob - 1_no_target)
      % Começamos com as probabilidades previstas
      dy = y_prob;
      targets = Y_batch(:, t);
      % Subtraímos 1 apenas na posição da classe correta
      for b = 1:batch_size
          dy(b, targets(b)) = dy(b, targets(b)) - 1;
      endfor

      % Gradientes da camada de saída (Output Layer)
      dW_hq = dW_hq + (H_curr' * dy);
      db_q  = db_q  + sum(dy, 1);

      % -- Derivada da Camada Oculta (Hidden Layer) --
      % O erro vem de dois lugares:
      % 1. Da saída deste passo (dy * W_hq')
      % 2. Do futuro (dh_next * W_hh') -> É aqui que o tempo volta!
      dh = (dy * W_hq') + (dh_next * W_hh');

      % Derivada da ativação tanh: dtanh = dh * (1 - H^2)
      dtanh = dh .* (1 - H_curr .^ 2);

      % Gradientes da camada oculta
      dW_xh = dW_xh + (X_t' * dtanh);
      dW_hh = dW_hh + (H_prev' * dtanh);
      db_h  = db_h  + sum(dtanh, 1);

      % Atualiza dh_next para a próxima iteração (que é o passo anterior t-1)
      dh_next = dtanh;
  endfor

  % Empacotar gradientes
  grads = {dW_xh, dW_hh, db_h, dW_hq, db_q};

endfunction
