import requests

BASE_LD = "http://localhost:8000"
BASE_NEG = "http://localhost:9000"

def test_en_negotiation_flow():
    user_id = "integration-test-user-123"

    # Step 1: Detect language
    detect_resp = requests.post(f"{BASE_LD}/detect", json={"text": "How much for this?"})
    detect_data = detect_resp.json()
    assert detect_data["language"] == "en"
    assert detect_data["confidence"] > 0.9

    # Step 2: Set user preference in LD
    requests.post(f"{BASE_LD}/user/{user_id}/lang", json={"language": detect_data["language"]})

    # Step 3: Negotiation payload
    payload = {
        "offer": 5000,
        "product": {"id": "sku-lip-001", "name": "Matte Lipstick - Ruby", "base_price": 12000},
        "state": {"conversation_id": "t1", "user_id": user_id, "meta": {"buyer_text": "How much for this?"}}
    }

    # Step 4: Call negotiation endpoint
    neg_resp = requests.post(f"{BASE_NEG}/decide", json=payload)
    neg_data = neg_resp.json()

    assert neg_data["action"] == "ESCALATE"
    assert "price" in neg_data
    assert "reply" in neg_data
    print("Negotiation flow passed:", neg_data)

if __name__ == "__main__":
    test_en_negotiation_flow()
