import urllib.request
import urllib.parse
import json
import os

def check_health():
    print("Checking Health...")
    try:
        with urllib.request.urlopen("http://localhost:5000/api/health") as response:
            print(f"Health Status Code: {response.getcode()}")
            print(f"Health Response: {response.read().decode()}")
    except Exception as e:
        print(f"Health check failed: {e}")

def check_predict():
    print("\nChecking Prediction...")
    url = "http://localhost:5000/api/predict"
    filepath = "Test_Input_Data/sample_input.csv"
    
    if not os.path.exists(filepath):
        print(f"File not found: {filepath}")
        return

    boundary = "---123456789"
    
    with open(filepath, "rb") as f:
        file_content = f.read()
        
    part_boundary = f"--{boundary}\r\n".encode()
    part_headers = f'Content-Disposition: form-data; name="file"; filename="sample_input.csv"\r\nContent-Type: text/csv\r\n\r\n'.encode()
    part_footer = f"\r\n--{boundary}--\r\n".encode()
    
    body = part_boundary + part_headers + file_content + part_footer
    
    req = urllib.request.Request(url, data=body, method="POST")
    req.add_header("Content-Type", f"multipart/form-data; boundary={boundary}")
    
    try:
        with urllib.request.urlopen(req) as response:
            print(f"Predict Status Code: {response.getcode()}")
            data = json.load(response)
            print(f"Success: {data.get('success')}")
            print(f"Predictions: {data.get('predictions')}")
    except Exception as e:
        print(f"Predict failed: {e}")
        if hasattr(e, 'read'):
             print(f"Error Body: {e.read().decode()}")

if __name__ == "__main__":
    check_health()
    check_predict()
