"""
Teste de performance e latência dos endpoints da API.
Utilize este template para medir o tempo de resposta (latência) e throughput sob carga.
Implemente casos de teste automatizados com requests sincronas e assíncronas conforme necessário.
"""
import time

def test_latency_example():
    start = time.time()
    # TODO: faça uma chamada real à API aqui, por exemplo requests.post(...)
    duration = time.time() - start
    # Defina um threshold para a latência máxima aceitável
    assert duration < 0.2  # 200ms

# Expanda este template usando pytest-benchmark, Locust ou Artillery para testes avançados de carga.
