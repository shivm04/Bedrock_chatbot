import json
import boto3
import os

# Bedrock client
bedrock = boto3.client(
    service_name="bedrock-runtime",
    region_name=os.environ.get("AWS_REGION", "us-east-1")
)

MODEL_ID = os.environ.get(
    "MODEL_ID",
    "anthropic.claude-3-haiku-20240307-v1:0"
)


def lambda_handler(event, context):
    try:
        # -----------------------------
        # Handle CORS Preflight Request
        # -----------------------------
        method = event.get("requestContext", {}).get("http", {}).get("method")

        if method == "OPTIONS":
            return {
                "statusCode": 200,
                "headers": {
                    "Content-Type": "application/json"
                },
                "body": ""
            }

        # -----------------------------
        # Parse request body
        # -----------------------------
        body = json.loads(event.get("body", "{}"))
        user_message = body.get("message", "")

        if not user_message:
            return response_json(400, {"error": "Message is required"})

        # -----------------------------
        # Claude Request Payload
        # -----------------------------
        payload = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 500,
            "messages": [
                {
                    "role": "user",
                    "content": user_message
                }
            ]
        }

        # -----------------------------
        # Call Bedrock Claude Model
        # -----------------------------
        response = bedrock.invoke_model(
            modelId=MODEL_ID,
            body=json.dumps(payload),
            contentType="application/json",
            accept="application/json"
        )

        response_body = json.loads(response["body"].read())

        # Extract Claude reply
        reply_text = response_body["content"][0]["text"]

        # -----------------------------
        # Success Response
        # -----------------------------
        return response_json(200, {
            "reply": reply_text
        })

    except Exception as e:
        print("ERROR:", str(e))

        return response_json(500, {
            "error": "Internal Server Error",
            "details": str(e)
        })


# -----------------------------
# Common JSON Response Function
# -----------------------------
def response_json(status, body):
    return {
        "statusCode": status,
        "headers": {
            "Content-Type": "application/json"
        },
        "body": json.dumps(body)
    }