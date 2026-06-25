import requests
import json
import time

BASE_URL = "http://127.0.0.1:8000"

def test_quarter_data():
    print("==================================================")
    print("   Testing Semantic Collision (Q1 vs Q4 Data)     ")
    print("==================================================")
    
    # Login as sid (Finance)
    print("[*] Logging in as 'sid' (Finance)...")
    res = requests.post(f"{BASE_URL}/login", json={"emp_id": "sid", "password": "sid123"})
    if res.status_code != 200:
        print(f"Login failed: {res.text}")
        return
    token = res.json()["access_token"]
    print("[+] Login successful.\n")
    
    questions = [
        "What was the total revenue in Q1 2024?",
        "What was the total revenue in Q4 2024?",
        "What were the marketing expenses in Q1 2024 vs Q4 2024?"
    ]
    
    for q in questions:
        print(f"[Q] QUESTION: {q}")
        start = time.time()
        ask_res = requests.post(f"{BASE_URL}/ask", json={"token": token, "question": q})
        end = time.time()
        
        if ask_res.status_code == 200:
            data = ask_res.json()
            print(f"[A] ANSWER: {data.get('answer')}")
            print(f"[S] SOURCES: {data.get('sources')}")
            print(f"[T] TIME: {round(end - start, 2)}s\n")
        else:
            print(f"[X] FAILED: {ask_res.text}\n")

if __name__ == "__main__":
    test_quarter_data()
