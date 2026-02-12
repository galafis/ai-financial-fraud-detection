# Guia de Contribuição

Bem-vindo(a) ao projeto AI-Powered Financial Fraud Detection! Agradecemos o seu interesse em contribuir.

Para garantir um processo de colaboração eficiente e produtivo, por favor, siga as diretrizes abaixo:

## Como Contribuir

1.  **Faça um Fork do Repositório**: Clique no botão 'Fork' no canto superior direito da página do GitHub para criar uma cópia do projeto em sua conta.

2.  **Clone o seu Fork**: Clone o repositório para a sua máquina local:
    ```bash
    git clone https://github.com/SEU_USUARIO/ai-financial-fraud-detection.git
    cd ai-financial-fraud-detection
    ```

3.  **Crie uma Branch para sua Feature ou Correção**: Crie uma nova branch para suas alterações. Use nomes descritivos como `feature/nova-funcionalidade` ou `fix/correcao-bug-x`.
    ```bash
    git checkout -b feature/sua-nova-feature
    ```

4.  **Faça suas Alterações**: Implemente suas modificações. Certifique-se de:
    *   Escrever código limpo e bem documentado.
    *   Adicionar testes unitários e de integração para novas funcionalidades ou correções.
    *   Atualizar a documentação (README.md, docs/) conforme necessário.

5.  **Teste suas Alterações**: Antes de submeter, execute os testes para garantir que suas mudanças não introduziram novos problemas:
    ```bash
    pytest tests/
    ```

6.  **Commit suas Alterações**: Faça commits atômicos e use mensagens de commit claras e concisas. Siga a convenção de commits (e.g., `feat: adiciona nova funcionalidade`, `fix: corrige bug de login`, `docs: atualiza readme`).
    ```bash
    git add .
    git commit -m "feat: sua mensagem de commit"
    ```

7.  **Envie suas Alterações e Abra um Pull Request (PR)**:
    ```bash
    git push origin feature/sua-nova-feature
    ```
    Em seguida, vá para a página do seu fork no GitHub e abra um Pull Request para a branch `main` do repositório original. Descreva suas alterações detalhadamente no PR.

## O que Estamos Procurando

*   🐛 Correções de bugs.
*   ✨ Novas funcionalidades ou modelos de ML.
*   📚 Melhorias na documentação.
*   🧪 Testes adicionais.
*   🔧 Otimizações de desempenho.
*   📊 Novas capacidades de monitoramento.

## Estilo de Código

*   Siga as diretrizes do PEP 8 para Python.
*   Use type hints sempre que possível.
*   Escreva docstrings para funções, classes e métodos.
*   Adicione testes unitários para todo o código novo.

## Licença

Ao contribuir para este projeto, você concorda que suas contribuições serão licenciadas sob a licença MIT do projeto.

---

**Agradecemos sua colaboração!**
