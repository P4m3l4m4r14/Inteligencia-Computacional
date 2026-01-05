function params = get_params(vocab_size, num_hiddens)
  % A entrada "device" foi removida pois o Octave roda na CPU por padrão.
  num_inputs = vocab_size;
  num_outputs = vocab_size;

  % --- Parâmetros da Camada Oculta ---
  % Shape: (Entradas x Ocultas)
  W_xh = 0.01 * randn(num_inputs, num_hiddens);

  % Shape: (Ocultas x Ocultas)
  W_hh = 0.01 * randn(num_hiddens, num_hiddens);

  % Bias da oculta (inicializado com zeros)
  b_h = zeros(1, num_hiddens);

  % --- Parâmetros da Camada de Saída ---
  % Shape: (Ocultas x Saídas)
  W_hq = 0.01 * randn(num_hiddens, num_outputs);

  % Bias de saída
  b_q = zeros(1, num_outputs);

  % Retornar como uma Cell Array
  % para manter a estrutura original de acesso por índice.
  params = {W_xh, W_hh, b_h, W_hq, b_q};
endfunction

