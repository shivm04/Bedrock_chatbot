import json
import boto3

bedrock = boto3.client(
    "bedrock-runtime",
    region_name="us-east-1"
)

MODEL_ID = "amazon.nova-lite-v1:0"

def lambda_handler(event, context):

    body = json.loads(event.get("body", "{}"))
    message = body.get("message", "")

    payload = {
        "messages": [
            {
                "role": "user",
                "content": [{"text": message}]
            }
        ]
    }

    response = bedrock.invoke_model(
        modelId=MODEL_ID,
        body=json.dumps(payload),
        contentType="application/json",
        accept="application/json"
    )

    result = json.loads(response["body"].read())

    reply = result["output"]["message"]["content"][0]["text"]

    return {
        "statusCode": 200,
        "headers": {
            "Content-Type": "application/json",
            "Access-Control-Allow-Origin": "*"
        },
        "body": json.dumps({"reply": reply})
    }