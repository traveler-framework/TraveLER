from fastapi import FastAPI
import httpx
import argparse

app = FastAPI()

async def forward_request(request_data: dict):
    async with httpx.AsyncClient(timeout=300.0) as client:
        # Forward the request to the original server and wait for the response
        response = await client.post(f"http://localhost:{sglang_port}/generate", json=request_data)
        return response.json()

@app.post("/generate")
async def api_wrapper(request_data: dict):
    # Forward the request and wait for the response
    response_data = await forward_request(request_data=request_data)
    # Return the response received from the original server
    return response_data

@app.post("/embed")
async def api_wrapper(request_data: dict):
    # Forward the request and wait for the response
    response_data = await forward_request(request_data=request_data)
    # Return the response received from the original server
    return response_data

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sglang_port", type=int, default=30000)
    parser.add_argument("--wrapper_port", type=int, default=8000)
    args = parser.parse_args()
    
    sglang_port = args.sglang_port
    wrapper_port = args.wrapper_port
    
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=wrapper_port)
