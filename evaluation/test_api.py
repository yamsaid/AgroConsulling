"""
Script de test pour l'API RAG Agricole
"""

import requests
import json
import time

BASE_URL = "http://localhost:8000"

def test_health():
    """Test health endpoint"""
    print("🧪 Test health endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/health")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Health: {data['status']}")
            print(f"   Uptime: {data['uptime']:.1f}s")
            print(f"   Components: {list(data['components'].keys())}")
            return True
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Health check error: {e}")
        return False

def test_system_info():
    """Test system info endpoint"""
    print("\n🧪 Test system info...")
    try:
        response = requests.get(f"{BASE_URL}/system/info")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ System: {data['system_name']}")
            print(f"   Version: {data['version']}")
            print(f"   Requests: {data['statistics']['total_requests']}")
            return True
        else:
            print(f"❌ System info failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ System info error: {e}")
        return False

def test_ask_question(question: str):
    """Test ask endpoint avec une question"""
    print(f"\n🧪 Test question: '{question}'")
    try:
        payload = {
            "question": question,
            "max_results": 3,
            "template": "standard",
            "verbose": True
        }
        
        start_time = time.time()
        response = requests.post(f"{BASE_URL}/ask", json=payload)
        processing_time = time.time() - start_time
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Réponse reçue en {processing_time:.2f}s")
            print(f"   Succès: {data['success']}")
            print(f"   Modèle: {data['metadata']['model']}")
            print(f"   Temps génération: {data['metadata']['generation_time']:.2f}s")
            print(f"   Documents utilisés: {data['metadata']['documents_used']}")
            print(f"\n🤖 RÉPONSE:")
            print(f"{data['reponse']}")
            
            if data['sources']:
                print(f"\n📚 SOURCES:")
                for source in data['sources']:
                    print(f"   - {source['titre']} (pertinence: {source['pertinence']:.3f})")
            
            return True
        else:
            print(f"❌ Ask failed: {response.status_code}")
            print(f"   Détail: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Ask error: {e}")
        return False

def main():
    """Test complet de l'API"""
    print("=" * 60)
    print("🧪 SUITE DE TESTS API RAG AGRICOLE")
    print("=" * 60)
    
    # 1. Test santé
    if not test_health():
        print("❌ API non disponible")
        return
    
    # 2. Test info système
    if not test_system_info():
        print("❌ Info système inaccessible")
        return
    
    # 3. Test questions
    test_questions = [
        "Quel engrais pour le mil ?",
        "Techniques de conservation des sols",
        "Maladies du maïs au Burkina Faso"
    ]
    
    for question in test_questions:
        success = test_ask_question(question)
        if not success:
            print("❌ Test question échoué")
            break
        print("\n" + "-" * 50)
    
    print("\n" + "=" * 60)
    print("✅ TESTS TERMINÉS")
    print("=" * 60)

if __name__ == "__main__":
    main()