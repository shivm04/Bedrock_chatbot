import json
import boto3

bedrock = boto3.client("bedrock-runtime", region_name="us-east-1")

MODEL_ID = "anthropic.claude-haiku-4-5-20251001-v1:0"


def lambda_handler(event, context):
    try:
        # Function URL request body
        body = json.loads(event.get("body", "{}"))
        user_message = body.get("message", "Hello")

        payload = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 300,
            "messages": [
                {
                    "role": "user",
                    "content": user_message
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

        reply = result["content"][0]["text"]

        return {
            "statusCode": 200,
            "headers": {
                "Content-Type": "application/json",
                "Access-Control-Allow-Origin": "*"
            },
            "body": json.dumps({"reply": reply})
        }

    except Exception as e:
        print("ERROR:", str(e))

        return {
            "statusCode": 500,
            "headers": {
                "Content-Type": "application/json",
                "Access-Control-Allow-Origin": "*"
            },
            "body": json.dumps({"error": str(e)})
        }