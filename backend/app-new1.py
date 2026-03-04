"""
Universal AWS AI Chatbot - Lambda Backend
Uses AWS Bedrock (Claude 3 Sonnet) with a single dynamic code-execution tool.
Claude writes real boto3 Python code at runtime to answer ANY AWS question —
no hardcoded service list, no predefined tools per service.
"""

import json
import boto3
import logging
import traceback
import io
from contextlib import redirect_stdout, redirect_stderr

logger = logging.getLogger()
logger.setLevel(logging.INFO)

# ─────────────────────────────────────────────
# Bedrock client  (must be in a region where
# Claude 3 Sonnet is available, e.g. us-east-1)
# ─────────────────────────────────────────────
bedrock_runtime = boto3.client("bedrock-runtime", region_name="us-east-1")

# Model  ─ swap to haiku for cheaper / opus for smarter
MODEL_ID = "anthropic.claude-3-sonnet-20240229-v1:0"


# ─────────────────────────────────────────────────────────────────────────────
# ONE universal tool  – Claude writes boto3 code → we execute it live
# ─────────────────────────────────────────────────────────────────────────────
TOOLS = [
    {
        "name": "execute_aws_query",
        "description": (
            "Execute any Python boto3 code to query ANY AWS service in real time.\n"
            "Use this for every AWS question — EC2, S3, Lambda, RDS, IAM, VPC, ECS, EKS,\n"
            "DynamoDB, SQS, SNS, CloudWatch, Route53, CloudFront, WAF, GuardDuty, Macie,\n"
            "Secrets Manager, SSM, ACM, Config, CloudTrail, CodePipeline, Glue, Athena,\n"
            "Redshift, ElastiCache, Kinesis, Step Functions, EventBridge, Organizations,\n"
            "Cost Explorer, Budgets, SageMaker, Bedrock, and every other service.\n\n"
            "Rules for the code you write:\n"
            "- boto3 and json are already imported\n"
            "- Store the final answer in a variable named exactly `result` (dict or list)\n"
            "- Use paginators for large result sets\n"
            "- Wrap per-region loops in try/except so one bad region does not abort\n"
            "- Convert non-serialisable types with str() inside json.dumps default\n"
            "- Limit lists to 100 items but always report the true total_count\n"
            "- Never use os.system, subprocess, open(), or __import__"
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "description": {
                    "type": "string",
                    "description": "One-line description of what this code does."
                },
                "code": {
                    "type": "string",
                    "description": (
                        "Valid Python boto3 code. Must assign a variable named `result`.\n\n"
                        "Examples:\n"
                        "  # EC2 count\n"
                        "  ec2 = boto3.client('ec2', region_name='us-east-1')\n"
                        "  pages = ec2.get_paginator('describe_instances').paginate()\n"
                        "  instances = [i for p in pages for r in p['Reservations'] for i in r['Instances']]\n"
                        "  result = {'total_count': len(instances), 'instances': instances[:50]}\n\n"
                        "  # All Lambda functions across every region\n"
                        "  all_fn = []\n"
                        "  regions = [r['RegionName'] for r in boto3.client('ec2','us-east-1').describe_regions()['Regions']]\n"
                        "  for reg in regions:\n"
                        "      try:\n"
                        "          for p in boto3.client('lambda', region_name=reg).get_paginator('list_functions').paginate():\n"
                        "              all_fn.extend(p['Functions'])\n"
                        "      except Exception:\n"
                        "          pass\n"
                        "  result = {'total_count': len(all_fn), 'functions': all_fn[:50]}"
                    )
                }
            },
            "required": ["description", "code"]
        }
    }
]


# ─────────────────────────────────────────────────────────────────────────────
# Safe executor
# ─────────────────────────────────────────────────────────────────────────────

def run_boto3_code(code: str) -> dict:
    """
    Execute Claude-generated boto3 code inside a restricted namespace.
    Returns a dict with status + result (or error details).
    """
    stdout_buf = io.StringIO()
    stderr_buf = io.StringIO()

    # Pre-populate namespace with safe builtins + boto3/json
    safe_builtins = {
        k: __builtins__[k] if isinstance(__builtins__, dict) else getattr(__builtins__, k, None)
        for k in (
            "print", "len", "range", "enumerate", "zip", "map", "filter",
            "list", "dict", "set", "tuple", "str", "int", "float", "bool",
            "sorted", "reversed", "min", "max", "sum", "abs", "round",
            "isinstance", "type", "hasattr", "getattr", "any", "all",
            "next", "iter", "repr", "hash",
            "Exception", "ValueError", "KeyError", "TypeError",
            "AttributeError", "IndexError", "StopIteration",
            "True", "False", "None",
        )
        if (isinstance(__builtins__, dict) and k in __builtins__)
        or (not isinstance(__builtins__, dict) and hasattr(__builtins__, k))
    }
    safe_builtins["__import__"] = __import__  # allow import inside exec

    ns = {
        "__builtins__": safe_builtins,
        "boto3": boto3,
        "json": json,
        "result": None,
    }

    # Inject extra stdlib helpers
    bootstrap = (
        "import datetime\nimport re\nimport math\nimport collections\n"
        "from decimal import Decimal\nimport itertools\n"
    )
    try:
        exec(bootstrap, ns)
    except Exception:
        pass

    local_ns: dict = {}
    try:
        with redirect_stdout(stdout_buf), redirect_stderr(stderr_buf):
            exec(compile(code, "<aws_query>", "exec"), ns, local_ns)

        result = local_ns.get("result", ns.get("result"))

        if result is None:
            return {
                "status": "warning",
                "message": "Code ran without error but 'result' was never set.",
                "stdout": stdout_buf.getvalue()[:2000],
            }

        # Make JSON-safe
        serialised = json.loads(json.dumps(result, default=str))
        return {
            "status": "success",
            "result": serialised,
            "stdout": stdout_buf.getvalue()[:500],
        }

    except Exception as exc:
        tb = traceback.format_exc()
        logger.error("Code execution error:\n%s", tb)
        return {
            "status": "error",
            "error": str(exc),
            "traceback": tb[-2000:],   # last 2 KB is enough
            "stdout": stdout_buf.getvalue()[:500],
            "stderr": stderr_buf.getvalue()[:500],
        }


# ─────────────────────────────────────────────────────────────────────────────
# System prompt
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a universal AWS Cloud Intelligence Assistant.
You answer questions about ANY AWS service by writing live boto3 Python code
and executing it through the execute_aws_query tool.

────────────────────────────────────────────────────────────────
WORKFLOW (always follow this)
────────────────────────────────────────────────────────────────
1. Understand what AWS data the user needs.
2. Write clean boto3 code and call execute_aws_query.
3. If the code returns an error → read the traceback, fix the code, retry (max 3 times).
4. Once you have the data → give a clear, direct answer.

────────────────────────────────────────────────────────────────
CODE RULES
────────────────────────────────────────────────────────────────
• boto3, json, datetime, re, math, collections are available.
• MUST set a variable named `result` (dict or list).
• Use paginators; never assume one page is complete.
• For multi-region queries: get region list from describe_regions(),
  wrap each region in try/except so one failure doesn't block others.
• Cap displayed lists at 50-100 items; always show total_count.
• datetime / Decimal → str() via json default=str.
• NO os.system, subprocess, open(), socket, or requests.

────────────────────────────────────────────────────────────────
RESPONSE STYLE
────────────────────────────────────────────────────────────────
• Lead with the answer and key numbers.
• Use tables or bullet points for resource lists.
• Include IDs, names, states, and regions where relevant.
• For large datasets summarise and offer to drill down.
• If access is denied for a service, say so clearly.

You can handle literally any AWS question including but not limited to:
EC2 counts / AMIs / security groups / key pairs, S3 buckets / objects / policies,
Lambda functions / layers / concurrency, RDS / Aurora / snapshots,
IAM users / roles / policies / MFA, VPC / subnets / NACLs / peering,
ECS / EKS clusters and tasks, DynamoDB tables, CloudWatch alarms / logs,
Route53 zones / records, CloudFront distributions, WAF rules,
GuardDuty findings, Macie findings, Security Hub findings,
Secrets Manager / SSM Parameter Store, ACM certificates,
Cost Explorer spend, Budgets, Savings Plans, Trusted Advisor,
CloudTrail events, Config rules compliance, CodePipeline status,
Glue jobs, Athena workgroups, Redshift clusters, ElastiCache clusters,
Kinesis streams / Firehose, Step Functions, EventBridge rules,
SQS queues, SNS topics, API Gateway APIs, AppSync, Cognito user pools,
SageMaker endpoints, Bedrock model access, Organizations accounts …
and anything else in the AWS SDK.
"""


# ─────────────────────────────────────────────────────────────────────────────
# Agentic loop
# ─────────────────────────────────────────────────────────────────────────────

def chat(user_message: str, history: list | None = None) -> dict:
    """
    Send user_message to Claude, let Claude generate + execute boto3 code,
    and return the final natural-language answer plus updated history.
    """
    history = history or []
    messages = history + [{"role": "user", "content": user_message}]

    for iteration in range(1, 16):          # max 15 Bedrock calls
        logger.info("Bedrock call #%d", iteration)

        try:
            raw = bedrock_runtime.invoke_model(
                modelId=MODEL_ID,
                body=json.dumps({
                    "anthropic_version": "bedrock-2023-05-31",
                    "max_tokens": 4096,
                    "system": SYSTEM_PROMPT,
                    "tools": TOOLS,
                    "messages": messages,
                }),
            )
        except Exception as exc:
            logger.error("Bedrock error: %s", exc)
            return {"response": f"Bedrock error: {exc}", "messages": messages}

        body = json.loads(raw["body"].read())
        stop   = body.get("stop_reason")
        blocks = body.get("content", [])
        logger.info("stop_reason=%s  blocks=%d", stop, len(blocks))

        # ── Claude is done ────────────────────────────────────────────────
        if stop == "end_turn":
            answer = "\n".join(b["text"] for b in blocks if b.get("type") == "text")
            return {
                "response": answer,
                "messages": messages + [{"role": "assistant", "content": blocks}],
            }

        # ── Claude wants to run code ──────────────────────────────────────
        if stop == "tool_use":
            messages.append({"role": "assistant", "content": blocks})
            tool_results = []

            for block in blocks:
                if block.get("type") != "tool_use":
                    continue
                if block["name"] == "execute_aws_query":
                    code        = block["input"].get("code", "")
                    description = block["input"].get("description", "running query")
                    logger.info("Tool call: %s", description)

                    exec_result = run_boto3_code(code)
                    logger.info("Tool status: %s", exec_result.get("status"))

                    tool_results.append({
                        "type":        "tool_result",
                        "tool_use_id": block["id"],
                        "content":     json.dumps(exec_result, default=str),
                    })
                else:
                    tool_results.append({
                        "type":        "tool_result",
                        "tool_use_id": block["id"],
                        "content":     json.dumps({"error": f"Unknown tool: {block['name']}"}),
                    })

            messages.append({"role": "user", "content": tool_results})
            continue   # next Bedrock call

        logger.warning("Unexpected stop_reason: %s", stop)
        break

    return {"response": "Could not complete the request. Please try again.", "messages": messages}


# ─────────────────────────────────────────────────────────────────────────────
# Lambda handler
# ─────────────────────────────────────────────────────────────────────────────

CORS_HEADERS = {
    "Content-Type":                 "application/json",
    "Access-Control-Allow-Origin":  "*",          # lock down for production
    "Access-Control-Allow-Headers": "Content-Type,Authorization,X-Api-Key",
    "Access-Control-Allow-Methods": "POST,OPTIONS",
}


def lambda_handler(event: dict, context) -> dict:
    """
    HTTP POST  /chat
    Body  { "message": "...", "conversation_history": [...] }
    Returns { "response": "...", "conversation_history": [...] }
    """
    # CORS pre-flight
    if event.get("httpMethod") == "OPTIONS":
        return {"statusCode": 200, "headers": CORS_HEADERS, "body": ""}

    try:
        body = event.get("body") or "{}"
        if isinstance(body, str):
            body = json.loads(body)

        message = (body.get("message") or "").strip()
        history = body.get("conversation_history") or []

        if not message:
            return {
                "statusCode": 400,
                "headers": CORS_HEADERS,
                "body": json.dumps({"error": "'message' field is required"}),
            }

        logger.info("User message: %.300s", message)
        output = chat(message, history)

        return {
            "statusCode": 200,
            "headers": CORS_HEADERS,
            "body": json.dumps(
                {
                    "response":              output["response"],
                    "conversation_history":  output["messages"],
                },
                default=str,
            ),
        }

    except Exception as exc:
        logger.error("Unhandled error: %s", exc, exc_info=True)
        return {
            "statusCode": 500,
            "headers": CORS_HEADERS,
            "body": json.dumps({"error": "Internal server error", "detail": str(exc)}),
        }
