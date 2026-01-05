function generated_text = predict_ch8(prefix, num_preds, params, idx_to_char, vocab_size)
  % 1. Configurações Iniciais
  % Descobrir num_hiddens olhando para o tamanho da matriz de pesos W_xh (params{1})
  % W_xh tem tamanho (vocab_size, num_hiddens)
  W_xh = params{1};
  num_hiddens = size(W_xh, 2);

  % Inicializa estado para APENAS 1 sequência (Batch Size = 1)
  state = init_rnn_state(1, num_hiddens);

  % 2. Converter o Prefixo (Texto) para Índices (Números)
  % Usamos ismember para achar os números correspondentes às letras do prefixo
  [~, prefix_idxs] = ismember(prefix, idx_to_char);

  % A lista de saídas começa com o primeiro caractere do prefixo
  outputs = [prefix_idxs(1)];

  % 3. Fase de Warm-up (Aquecimento)
  % Passa o resto do prefixo pela rede para atualizar o 'state'
  % Começa do 2º caractere até o fim do prefixo
  for i = 2:length(prefix_idxs)
      % A entrada é o caractere ANTERIOR (que já está salvo em outputs)
      prev_idx = outputs(end);

      % Criar X one-hot manualmente para 1 passo e 1 batch
      % Shape necessário pro rnn: (num_steps, batch_size, vocab_size) -> (1, 1, vocab)
      X = zeros(1, 1, vocab_size);
      X(1, 1, prev_idx) = 1;

      % Roda a RNN. Ignoramos o Y (predição) aqui, queremos só o 'state' atualizado
      [~, state] = rnn(X, state, params);

      % Na lista de saídas, forçamos o caractere real do prefixo (Ground Truth)
      outputs = [outputs, prefix_idxs(i)];
  endfor

  % 4. Fase de Predição (Geração de texto novo)
  for i = 1:num_preds
      % A entrada é o último caractere gerado (ou o último do prefixo)
      prev_idx = outputs(end);

      % Prepara entrada One-Hot
      X = zeros(1, 1, vocab_size);
      X(1, 1, prev_idx) = 1;

      % Roda a RNN
      [Y, state] = rnn(X, state, params);

      % Y é um vetor de probabilidades (1 x vocab_size).
      % Pegamos o índice com maior valor (Argmax)
      [~, predicted_idx] = max(Y, [], 2);

      % Adiciona o índice previsto à lista
      outputs = [outputs, predicted_idx];
  endfor

  % 5. Decodificar (Números -> Texto)
  % O Octave permite indexar o vetor de caracteres direto com a lista de inteiros
  generated_text = idx_to_char(outputs);

endfunction
