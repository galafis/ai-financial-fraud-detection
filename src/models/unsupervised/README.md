# Modelos Não Supervisionados – AI Financial Fraud Detection

Implementações de algoritmos não supervisionados para detecção de anomalias e agrupamento no contexto de fraudes financeiras.

Algoritmos Documentados
-------------------------
- Isolation Forest: Detecção de outliers baseada em floresta randômica
- Autoencoders: Reconhecimento de padrões distantes usando redes neurais
- KMeans/Agglomerative: Clusterização para segmentação comportamental

Estrutura de Arquivos
---------------------
- `isolation_forest_model.py`: Implementação e interface de uso
- `autoencoder_model.py`: Redes neurais autoencoders e métricas
- `clustering_model.py`: Diversas técnicas de agrupamento (KMeans, Agglomerative etc.)
- `model_trainer.py`: Treinador unificado para algoritmos não supervisionados

Orientações
-----------
- Documente todas as configurações específicas por técnica.
- Inclua exemplos de uso para integração com o pipeline de features/inferência.
- Mantenha benchmarks de comparação e explique limitações dos métodos.

Ao adicionar novos algoritmos, mantenha este README atualizado.
