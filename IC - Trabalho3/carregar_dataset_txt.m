function [X_reshaped, Y_reshaped, vocab_len, unique_chars] = carregar_dataset_txt(caminho_arquivo, batch_size, num_steps)

  % =================================================================
  % 1. LEITURA
  % =================================================================
  fid = fopen(caminho_arquivo, 'r');
  if fid == -1
    error('ERRO: Nao foi possivel abrir o arquivo: %s', caminho_arquivo);
  endif
  text_raw = fread(fid, '*char')';
  fclose(fid);

  % =================================================================
  % 2. LIMPEZA (Regex)
  % =================================================================
  text_raw = lower(text_raw);
  % Troca quebras de linha e tabs por espaço
  text_clean = regexprep(text_raw, '[\n\r\t]', ' ');
  % Remove tudo que nao for letra ou espaço
  text_clean = regexprep(text_raw, '[^a-z áàâãéèêíóôõúç]', '');
  % Remove espaços multiplos
  text_clean = regexprep(text_clean, '\s+', ' ');
  text_clean = strtrim(text_clean);

  % =================================================================
  % 3. VOCABULÁRIO E TOKENIZAÇÃO (MÉTODO VETORIZADO)
  % =================================================================
  % Lista de caracteres únicos (o índice aqui será o ID do char)
  unique_chars = unique(text_clean);
  vocab_len = length(unique_chars);

  if vocab_len == 0
    error('Vocabulario vazio. Verifique o arquivo txt.');
  endif

  % A função 'ismember' compara o texto inteiro contra o vocabulário.
  % O segundo retorno (idx) é exatamente a posição de cada letra no vocab.
  [~, corpus] = ismember(text_clean, unique_chars);

  % =================================================================
  % 4. BATCHING
  % =================================================================
  num_tokens = length(corpus);
  num_batches = floor((num_tokens - 1) / (batch_size * num_steps));

  if num_batches <= 0
     error('Texto muito curto para gerar batches.');
  endif

  limit = num_batches * batch_size * num_steps;
  X_data = corpus(1 : limit);
  Y_data = corpus(2 : limit + 1);

  X_reshaped = reshape(X_data, batch_size, []);
  Y_reshaped = reshape(Y_data, batch_size, []);

  printf("Sucesso! Vocab: %d chars. Batches: %d. Metodo: ismember (rapido)\n", vocab_len, num_batches);

endfunction
