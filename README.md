# 🧠 Inteligência Computacional - Projetos Práticos

Este repositório contém as implementações e trabalhos práticos desenvolvidos para a disciplina de **Inteligência Computacional** do curso de Engenharia de Computação.

Os projetos focam na implementação de algoritmos de Redes Neurais "do zero" (from scratch), utilizando matemática pura e álgebra linear, sem a dependência de frameworks de alto nível (como PyTorch ou TensorFlow) para a lógica principal.

---

## 📂 Estrutura do Repositório

### 1. [IC - Trabalho 2] - (Insira o Tema, ex: MLP / Classificação)
*Pasta: `IC - Trabalho2`*

> *Implementação de uma Rede Neural RBF e Problema do caxeiro viajante*

**Principais conceitos:**
- Rede RBF
- Caxeiro viajante - ainda não concluido

---

### 2. [IC - Trabalho 3] - RNN from Scratch (Geração de Texto)
*Pasta: `IC - Trabalho3`*

Implementação completa de uma **Rede Neural Recorrente (RNN)** para modelagem de linguagem em nível de caractere (Character-Level Language Model). A rede aprende a prever o próximo caractere de uma sequência, permitindo gerar texto novo ao estilo do dataset de treinamento (livros).

**Destaques Técnicos:**
- **From Scratch:** Toda a lógica, incluindo o *Forward Pass* e o *Backward Pass*, foi implementada manualmente.
- **Backpropagation Through Time (BPTT):** Cálculo manual dos gradientes voltando no tempo.
- **Gradient Clipping:** Implementado para evitar o problema de explosão de gradientes.
- **Otimização:** Uso de SGD (Stochastic Gradient Descent).
- **Dataset:** Treinado com textos literários (Ex: *Percy Jackson* / *A Máquina do Tempo*).

---

## 🛠️ Tecnologias Utilizadas

* **Linguagem:** GNU Octave (Compatível com MATLAB).
* **Bibliotecas:** Nenhuma biblioteca de Deep Learning foi utilizada. Apenas bibliotecas padrão de álgebra linear.

## 🚀 Como Executar

Para rodar os projetos, você precisará do [GNU Octave](https://gnu.org/software/octave/) instalado.

1. Clone este repositório:
   ```bash
   git clone [https://github.com/P4m3l4m4r14/IC-Trabalho3.git](https://github.com/P4m3l4m4r14/IC-Trabalho3.git)
