"""
Testes de integração para endpoints da API de detecção de fraudes financeiras.

Este módulo contém testes de integração que validam o comportamento completo
dos endpoints da API, incluindo:
- Autenticação e autorização
- Validação de entrada e saída de dados
- Códigos de status HTTP
- Formato de resposta JSON
- Tratamento de erros
- Performance básica dos endpoints

Os testes utilizam requisições HTTP reais contra uma instância da API
para garantir que todos os componentes funcionem corretamente em conjunto.
"""

import pytest
import requests
import httpx
import json
from typing import Dict, Any
from unittest import TestCase


class TestAPIEndpoints(TestCase):
    """
    Classe para testes de integração dos endpoints da API.
    """
    
    @classmethod
    def setUpClass(cls):
        """Configuração inicial dos testes."""
        # URL base da API (ajustar conforme necessário)
        cls.base_url = "http://localhost:8000/api/v1"
        cls.headers = {
            "Content-Type": "application/json",
            "Accept": "application/json"
        }
        
        # Dados de exemplo para testes
        cls.sample_transaction_data = {
            "transaction_id": "txn_123456789",
            "amount": 1500.00,
            "timestamp": "2025-09-16T20:37:00Z",
            "merchant_id": "merchant_001",
            "card_number": "****1234",
            "location": "São Paulo, SP",
            "transaction_type": "purchase"
        }
    
    def test_health_check_endpoint(self):
        """
        Testa o endpoint de health check da API.
        """
        response = requests.get(f"{self.base_url}/health")
        
        self.assertEqual(response.status_code, 200)
        self.assertIn("status", response.json())
        self.assertEqual(response.json()["status"], "healthy")
    
    def test_fraud_detection_endpoint_valid_data(self):
        """
        Testa o endpoint de detecção de fraude com dados válidos.
        """
        response = requests.post(
            f"{self.base_url}/fraud/detect",
            json=self.sample_transaction_data,
            headers=self.headers
        )
        
        self.assertEqual(response.status_code, 200)
        
        # Verifica estrutura da resposta
        response_data = response.json()
        self.assertIn("is_fraud", response_data)
        self.assertIn("confidence_score", response_data)
        self.assertIn("risk_factors", response_data)
        
        # Verifica tipos de dados
        self.assertIsInstance(response_data["is_fraud"], bool)
        self.assertIsInstance(response_data["confidence_score"], (int, float))
        self.assertIsInstance(response_data["risk_factors"], list)
    
    def test_fraud_detection_endpoint_invalid_data(self):
        """
        Testa o endpoint de detecção de fraude com dados inválidos.
        """
        invalid_data = {
            "transaction_id": "",  # ID vazio
            "amount": -100,  # Valor negativo
        }
        
        response = requests.post(
            f"{self.base_url}/fraud/detect",
            json=invalid_data,
            headers=self.headers
        )
        
        self.assertEqual(response.status_code, 400)
        
        response_data = response.json()
        self.assertIn("error", response_data)
        self.assertIn("message", response_data)
    
    def test_fraud_detection_endpoint_missing_fields(self):
        """
        Testa o endpoint com campos obrigatórios ausentes.
        """
        incomplete_data = {
            "transaction_id": "txn_123"
            # Faltando outros campos obrigatórios
        }
        
        response = requests.post(
            f"{self.base_url}/fraud/detect",
            json=incomplete_data,
            headers=self.headers
        )
        
        self.assertEqual(response.status_code, 422)  # Unprocessable Entity
    
    def test_authentication_required_endpoint(self):
        """
        Testa endpoint que requer autenticação sem token.
        """
        response = requests.get(f"{self.base_url}/admin/stats")
        
        self.assertIn(response.status_code, [401, 403])  # Unauthorized ou Forbidden
    
    def test_not_found_endpoint(self):
        """
        Testa endpoint inexistente.
        """
        response = requests.get(f"{self.base_url}/nonexistent")
        
        self.assertEqual(response.status_code, 404)
    
    def test_method_not_allowed(self):
        """
        Testa método HTTP não permitido em endpoint.
        """
        response = requests.put(f"{self.base_url}/fraud/detect")
        
        self.assertEqual(response.status_code, 405)  # Method Not Allowed


class TestAPIEndpointsWithHTTPX(TestCase):
    """
    Classe para testes usando httpx (async/await).
    """
    
    def setUp(self):
        """Configuração para cada teste."""
        self.base_url = "http://localhost:8000/api/v1"
        self.headers = {
            "Content-Type": "application/json",
            "Accept": "application/json"
        }
    
    @pytest.mark.asyncio
    async def test_async_fraud_detection(self):
        """
        Teste assíncrono do endpoint de detecção de fraude.
        """
        sample_data = {
            "transaction_id": "txn_async_001",
            "amount": 2500.00,
            "timestamp": "2025-09-16T20:37:00Z",
            "merchant_id": "merchant_002",
            "card_number": "****5678",
            "location": "Rio de Janeiro, RJ",
            "transaction_type": "withdrawal"
        }
        
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.base_url}/fraud/detect",
                json=sample_data,
                headers=self.headers
            )
        
        assert response.status_code == 200
        response_data = response.json()
        assert "is_fraud" in response_data
        assert "confidence_score" in response_data
    
    @pytest.mark.asyncio
    async def test_concurrent_requests(self):
        """
        Testa múltiplas requisições concorrentes.
        """
        import asyncio
        
        async def make_request(client, transaction_id):
            data = {
                "transaction_id": f"txn_{transaction_id}",
                "amount": 1000.00,
                "timestamp": "2025-09-16T20:37:00Z",
                "merchant_id": "merchant_001",
                "card_number": "****1234",
                "location": "São Paulo, SP",
                "transaction_type": "purchase"
            }
            
            response = await client.post(
                f"{self.base_url}/fraud/detect",
                json=data,
                headers=self.headers
            )
            return response
        
        async with httpx.AsyncClient() as client:
            # Faz 5 requisições concorrentes
            tasks = [make_request(client, i) for i in range(5)]
            responses = await asyncio.gather(*tasks)
        
        # Verifica se todas as respostas foram bem-sucedidas
        for response in responses:
            assert response.status_code == 200


if __name__ == "__main__":
    # Executa os testes quando o arquivo é executado diretamente
    import unittest
    
    # Executa apenas os testes síncronos por padrão
    suite = unittest.TestLoader().loadTestsFromTestCase(TestAPIEndpoints)
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite)
    
    print("\n" + "="*50)
    print("Para executar os testes assíncronos, use:")
    print("pytest test_endpoints.py::TestAPIEndpointsWithHTTPX -v")
