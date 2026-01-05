function inputs_onehot = to_onehot(X, vocab_size)
  % X: matriz (batch_size, num_steps) vinda do dataset
  % Retorna: (num_steps, batch_size, vocab_size)

  % Transpõe X para ter (num_steps, batch_size) - padrão D2L
  X = X';
  [num_steps, batch_size] = size(X);

  % Cria tensor 3D de zeros
  inputs_onehot = zeros(num_steps, batch_size, vocab_size);

  % Preenche com 1s nas posições corretas
  for t = 1:num_steps
    for b = 1:batch_size
      idx = X(t, b);
      inputs_onehot(t, b, idx) = 1;
    endfor
  endfor
endfunction
