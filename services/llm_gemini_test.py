import google.generativeai as genai

genai.configure(api_key="AIzaSyDHKVauUqUrpwhnn7d-iRvVTg8AX_nSDng")
model = genai.GenerativeModel('gemini-2.5-flash')

try:
    # Using a tiny prompt to minimize usage
    response = model.generate_content("Hi", generation_config={"max_output_tokens": 5})
    
    print("✅ Connection Successful!")
    print(f"Response: {response.text}")
    print(f"Tokens used: {response.usage_metadata.total_token_count}")
except Exception as e:
    print(f"❌ Error: {e}")
