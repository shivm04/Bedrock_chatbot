"""
AWS AI Chatbot - Lambda Backend
Uses AWS Bedrock (Claude 3) with Tool Use to query any AWS service dynamically.
Deploy this as a Lambda function with appropriate IAM permissions.
"""

import json
import boto3
import logging
from typing import Any

logger = logging.getLogger()
logger.setLevel(logging.INFO)

# ─────────────────────────────────────────────
# Bedrock client (same region as Lambda)
# ─────────────────────────────────────────────
bedrock_runtime = boto3.client("bedrock-runtime", region_name="us-east-1")

# ─────────────────────────────────────────────
# MODEL ID  –  Claude 3 Sonnet on Bedrock
# Change to "anthropic.claude-3-haiku-20240307-v1:0" for cheaper/faster
# ─────────────────────────────────────────────
MODEL_ID = "anthropic.claude-3-sonnet-20240229-v1:0"

# ─────────────────────────────────────────────
# Tool definitions – Claude decides which one to call
# ─────────────────────────────────────────────
TOOLS = [
    {
        "name": "get_ec2_instances",
        "description": "List all EC2 instances with their IDs, state, type, AMI ID, public/private IPs, tags, and launch time. Use when asked about EC2 servers, virtual machines, instance count, or running/stopped servers.",
        "input_schema": {
            "type": "object",
            "properties": {
                "region": {"type": "string", "description": "AWS region, e.g. us-east-1. Use 'all' to query all regions."},
                "state": {"type": "string", "description": "Filter by state: running, stopped, terminated, or 'all'", "default": "all"}
            },
            "required": []
        }
    },
    {
        "name": "get_lambda_functions",
        "description": "List all Lambda functions with their names, runtime, memory, timeout, last modified, and code size. Use when asked about Lambda functions or serverless functions.",
        "input_schema": {
            "type": "object",
            "properties": {
                "region": {"type": "string", "description": "AWS region. Use 'all' for all regions."}
            },
            "required": []
        }
    },
    {
        "name": "get_s3_buckets",
        "description": "List all S3 buckets with name, creation date, region, and object count/size. Use when asked about S3 buckets or object storage.",
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": []
        }
    },
    {
        "name": "get_rds_instances",
        "description": "List all RDS database instances with engine, status, instance class, storage, and endpoint. Use for questions about databases, RDS, or managed DB services.",
        "input_schema": {
            "type": "object",
            "properties": {
                "region": {"type": "string", "description": "AWS region."}
            },
            "required": []
        }
    },
    {
        "name": "get_iam_users",
        "description": "List all IAM users with creation date, last login, attached policies, and MFA status. Use when asked about IAM users, access management, or user accounts.",
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": []
        }
    },
    {
        "name": "get_cloudwatch_alarms",
        "description": "List CloudWatch alarms with their state, metric, and threshold. Use for monitoring, alarms, or alerts questions.",
        "input_schema": {
            "type": "object",
            "properties": {
                "region": {"type": "string", "description": "AWS region."},
                "state": {"type": "string", "description": "Filter: OK, ALARM, INSUFFICIENT_DATA, or 'all'", "default": "all"}
            },
            "required": []
        }
    },
    {
        "name": "get_vpc_info",
        "description": "List all VPCs with CIDR blocks, subnets, route tables, and internet gateways. Use for networking questions about VPC, subnets, or network topology.",
        "input_schema": {
            "type": "object",
            "properties": {
                "region": {"type": "string", "description": "AWS region."}
            },
            "required": []
        }
    },
    {
        "name": "get_ecs_clusters",
        "description": "List all ECS clusters, services, and tasks. Use when asked about containers, ECS, Fargate, or container orchestration.",
        "input_schema": {
            "type": "object",
            "properties": {
                "region": {"type": "string", "description": "AWS region."}
            },
            "required": []
        }
    },
    {
        "name": "get_eks_clusters",
        "description": "List all EKS Kubernetes clusters with their status and version. Use when asked about Kubernetes, EKS, or K8s clusters.",
        "input_schema": {
            "type": "object",
            "properties": {
                "region": {"type": "string", "description": "AWS region."}
            },
            "required": []
        }
    },
    {
        "name": "get_cloudformation_stacks",
        "description": "List all CloudFormation stacks with their status, resources, and outputs. Use when asked about infrastructure as code or CF stacks.",
        "input_schema": {
            "type": "object",
            "properties": {
                "region": {"type": "string", "description": "AWS region."}
            },
            "required": []
        }
    },
    {
        "name": "get_sns_topics",
        "description": "List SNS topics and their subscription counts. Use for messaging, notifications, or SNS questions.",
        "input_schema": {
            "type": "object",
            "properties": {
                "region": {"type": "string", "description": "AWS region."}
            },
            "required": []
        }
    },
    {
        "name": "get_sqs_queues",
        "description": "List SQS queues with approximate message counts and queue attributes. Use for queue, messaging, or SQS questions.",
        "input_schema": {
            "type": "object",
            "properties": {
                "region": {"type": "string", "description": "AWS region."}
            },
            "required": []
        }
    },
    {
        "name": "get_dynamodb_tables",
        "description": "List DynamoDB tables with item counts, size, status, and provisioned throughput. Use for NoSQL database or DynamoDB questions.",
        "input_schema": {
            "type": "object",
            "properties": {
                "region": {"type": "string", "description": "AWS region."}
            },
            "required": []
        }
    },
    {
        "name": "get_elb_load_balancers",
        "description": "List all Application, Network, and Classic Load Balancers with their DNS names, state, and target groups. Use for load balancer or traffic distribution questions.",
        "input_schema": {
            "type": "object",
            "properties": {
                "region": {"type": "string", "description": "AWS region."}
            },
            "required": []
        }
    },
    {
        "name": "get_ec2_amis",
        "description": "List EC2 AMIs (Amazon Machine Images) owned by this account with image ID, name, state, and creation date. Use when asked about AMIs, machine images, or server images.",
        "input_schema": {
            "type": "object",
            "properties": {
                "region": {"type": "string", "description": "AWS region."}
            },
            "required": []
        }
    },
    {
        "name": "get_security_groups",
        "description": "List all EC2 security groups with their rules, ports, and associated resources. Use for firewall, security group, or network security questions.",
        "input_schema": {
            "type": "object",
            "properties": {
                "region": {"type": "string", "description": "AWS region."}
            },
            "required": []
        }
    },
    {
        "name": "get_cost_and_usage",
        "description": "Get AWS cost and usage data for a time period broken down by service. Use when asked about billing, costs, spending, or AWS bill.",
        "input_schema": {
            "type": "object",
            "properties": {
                "days": {"type": "integer", "description": "Number of past days to query (default 30)", "default": 30}
            },
            "required": []
        }
    },
    {
        "name": "get_account_summary",
        "description": "Get a high-level summary of the AWS account including region, account ID, and key resource counts across all major services. Use when asked for an overview or summary of the account.",
        "input_schema": {
            "type": "object",
            "properties": {
                "region": {"type": "string", "description": "Primary AWS region to summarize."}
            },
            "required": []
        }
    }
]


# ─────────────────────────────────────────────────────────────────────────────
# AWS Tool Implementations
# ─────────────────────────────────────────────────────────────────────────────

def get_all_regions() -> list[str]:
    ec2 = boto3.client("ec2", region_name="us-east-1")
    resp = ec2.describe_regions(Filters=[{"Name": "opt-in-status", "Values": ["opt-in-not-required", "opted-in"]}])
    return [r["RegionName"] for r in resp["Regions"]]


def get_ec2_instances(region: str = "us-east-1", state: str = "all") -> dict:
    regions = get_all_regions() if region == "all" else [region]
    all_instances = []
    for reg in regions:
        ec2 = boto3.client("ec2", region_name=reg)
        filters = [] if state == "all" else [{"Name": "instance-state-name", "Values": [state]}]
        paginator = ec2.get_paginator("describe_instances")
        for page in paginator.paginate(Filters=filters):
            for reservation in page["Reservations"]:
                for inst in reservation["Instances"]:
                    name = next((t["Value"] for t in inst.get("Tags", []) if t["Key"] == "Name"), "N/A")
                    all_instances.append({
                        "Region": reg,
                        "InstanceId": inst["InstanceId"],
                        "Name": name,
                        "State": inst["State"]["Name"],
                        "InstanceType": inst["InstanceType"],
                        "ImageId": inst.get("ImageId", "N/A"),
                        "PublicIP": inst.get("PublicIpAddress", "N/A"),
                        "PrivateIP": inst.get("PrivateIpAddress", "N/A"),
                        "LaunchTime": str(inst.get("LaunchTime", "N/A"))
                    })
    return {"total_count": len(all_instances), "instances": all_instances}


def get_lambda_functions(region: str = "us-east-1") -> dict:
    regions = get_all_regions() if region == "all" else [region]
    all_functions = []
    for reg in regions:
        client = boto3.client("lambda", region_name=reg)
        paginator = client.get_paginator("list_functions")
        for page in paginator.paginate():
            for fn in page["Functions"]:
                all_functions.append({
                    "Region": reg,
                    "FunctionName": fn["FunctionName"],
                    "Runtime": fn.get("Runtime", "N/A"),
                    "MemoryMB": fn["MemorySize"],
                    "TimeoutSec": fn["Timeout"],
                    "LastModified": fn["LastModified"],
                    "CodeSizeMB": round(fn["CodeSize"] / 1024 / 1024, 2),
                    "Description": fn.get("Description", "")
                })
    return {"total_count": len(all_functions), "functions": all_functions}


def get_s3_buckets() -> dict:
    s3 = boto3.client("s3", region_name="us-east-1")
    response = s3.list_buckets()
    buckets = []
    for b in response.get("Buckets", []):
        try:
            loc = s3.get_bucket_location(Bucket=b["Name"])
            region = loc["LocationConstraint"] or "us-east-1"
        except Exception:
            region = "unknown"
        buckets.append({
            "Name": b["Name"],
            "CreationDate": str(b["CreationDate"]),
            "Region": region
        })
    return {"total_count": len(buckets), "buckets": buckets}


def get_rds_instances(region: str = "us-east-1") -> dict:
    client = boto3.client("rds", region_name=region)
    paginator = client.get_paginator("describe_db_instances")
    instances = []
    for page in paginator.paginate():
        for db in page["DBInstances"]:
            instances.append({
                "DBIdentifier": db["DBInstanceIdentifier"],
                "Engine": f"{db['Engine']} {db.get('EngineVersion', '')}",
                "Status": db["DBInstanceStatus"],
                "Class": db["DBInstanceClass"],
                "StorageGB": db["AllocatedStorage"],
                "MultiAZ": db["MultiAZ"],
                "Endpoint": db.get("Endpoint", {}).get("Address", "N/A")
            })
    return {"total_count": len(instances), "instances": instances}


def get_iam_users() -> dict:
    iam = boto3.client("iam")
    paginator = iam.get_paginator("list_users")
    users = []
    for page in paginator.paginate():
        for user in page["Users"]:
            mfa = iam.list_mfa_devices(UserName=user["UserName"])
            users.append({
                "UserName": user["UserName"],
                "UserId": user["UserId"],
                "Created": str(user["CreateDate"]),
                "LastLogin": str(user.get("PasswordLastUsed", "Never")),
                "MFAEnabled": len(mfa["MFADevices"]) > 0
            })
    return {"total_count": len(users), "users": users}


def get_cloudwatch_alarms(region: str = "us-east-1", state: str = "all") -> dict:
    client = boto3.client("cloudwatch", region_name=region)
    kwargs = {} if state == "all" else {"StateValue": state}
    paginator = client.get_paginator("describe_alarms")
    alarms = []
    for page in paginator.paginate(**kwargs):
        for alarm in page["MetricAlarms"]:
            alarms.append({
                "AlarmName": alarm["AlarmName"],
                "State": alarm["StateValue"],
                "Metric": alarm.get("MetricName", "N/A"),
                "Threshold": alarm.get("Threshold", "N/A"),
                "Namespace": alarm.get("Namespace", "N/A")
            })
    return {"total_count": len(alarms), "alarms": alarms}


def get_vpc_info(region: str = "us-east-1") -> dict:
    ec2 = boto3.client("ec2", region_name=region)
    vpcs_raw = ec2.describe_vpcs()["Vpcs"]
    subnets_raw = ec2.describe_subnets()["Subnets"]
    igws_raw = ec2.describe_internet_gateways()["InternetGateways"]
    vpcs = []
    for vpc in vpcs_raw:
        name = next((t["Value"] for t in vpc.get("Tags", []) if t["Key"] == "Name"), "N/A")
        subnet_count = sum(1 for s in subnets_raw if s["VpcId"] == vpc["VpcId"])
        vpcs.append({
            "VpcId": vpc["VpcId"],
            "Name": name,
            "CidrBlock": vpc["CidrBlock"],
            "IsDefault": vpc["IsDefault"],
            "SubnetCount": subnet_count,
            "State": vpc["State"]
        })
    return {
        "vpc_count": len(vpcs),
        "subnet_count": len(subnets_raw),
        "internet_gateway_count": len(igws_raw),
        "vpcs": vpcs
    }


def get_ecs_clusters(region: str = "us-east-1") -> dict:
    client = boto3.client("ecs", region_name=region)
    cluster_arns = client.list_clusters().get("clusterArns", [])
    clusters = []
    if cluster_arns:
        details = client.describe_clusters(clusters=cluster_arns)["clusters"]
        for c in details:
            clusters.append({
                "ClusterName": c["clusterName"],
                "Status": c["status"],
                "ActiveServices": c["activeServicesCount"],
                "RunningTasks": c["runningTasksCount"],
                "PendingTasks": c["pendingTasksCount"],
                "RegisteredContainerInstances": c["registeredContainerInstancesCount"]
            })
    return {"total_count": len(clusters), "clusters": clusters}


def get_eks_clusters(region: str = "us-east-1") -> dict:
    client = boto3.client("eks", region_name=region)
    names = client.list_clusters().get("clusters", [])
    clusters = []
    for name in names:
        detail = client.describe_cluster(name=name)["cluster"]
        clusters.append({
            "Name": name,
            "Status": detail["status"],
            "KubernetesVersion": detail["version"],
            "Endpoint": detail.get("endpoint", "N/A"),
            "CreatedAt": str(detail.get("createdAt", "N/A"))
        })
    return {"total_count": len(clusters), "clusters": clusters}


def get_cloudformation_stacks(region: str = "us-east-1") -> dict:
    client = boto3.client("cloudformation", region_name=region)
    paginator = client.get_paginator("describe_stacks")
    stacks = []
    for page in paginator.paginate():
        for stack in page["Stacks"]:
            stacks.append({
                "StackName": stack["StackName"],
                "Status": stack["StackStatus"],
                "Created": str(stack.get("CreationTime", "N/A")),
                "Description": stack.get("Description", ""),
                "OutputCount": len(stack.get("Outputs", []))
            })
    return {"total_count": len(stacks), "stacks": stacks}


def get_sns_topics(region: str = "us-east-1") -> dict:
    client = boto3.client("sns", region_name=region)
    paginator = client.get_paginator("list_topics")
    topics = []
    for page in paginator.paginate():
        for t in page["Topics"]:
            arn = t["TopicArn"]
            attrs = client.get_topic_attributes(TopicArn=arn).get("Attributes", {})
            topics.append({
                "TopicArn": arn,
                "Name": arn.split(":")[-1],
                "SubscriptionsConfirmed": attrs.get("SubscriptionsConfirmed", "N/A"),
                "SubscriptionsPending": attrs.get("SubscriptionsPending", "N/A")
            })
    return {"total_count": len(topics), "topics": topics}


def get_sqs_queues(region: str = "us-east-1") -> dict:
    client = boto3.client("sqs", region_name=region)
    urls = client.list_queues().get("QueueUrls", [])
    queues = []
    for url in urls:
        attrs = client.get_queue_attributes(
            QueueUrl=url,
            AttributeNames=["ApproximateNumberOfMessages", "ApproximateNumberOfMessagesNotVisible", "CreatedTimestamp"]
        ).get("Attributes", {})
        queues.append({
            "QueueName": url.split("/")[-1],
            "QueueUrl": url,
            "ApproxMessages": attrs.get("ApproximateNumberOfMessages", "N/A"),
            "MessagesInFlight": attrs.get("ApproximateNumberOfMessagesNotVisible", "N/A")
        })
    return {"total_count": len(queues), "queues": queues}


def get_dynamodb_tables(region: str = "us-east-1") -> dict:
    client = boto3.client("dynamodb", region_name=region)
    paginator = client.get_paginator("list_tables")
    tables = []
    for page in paginator.paginate():
        for table_name in page["TableNames"]:
            detail = client.describe_table(TableName=table_name)["Table"]
            tables.append({
                "TableName": table_name,
                "Status": detail["TableStatus"],
                "ItemCount": detail.get("ItemCount", 0),
                "SizeBytes": detail.get("TableSizeBytes", 0),
                "BillingMode": detail.get("BillingModeSummary", {}).get("BillingMode", "PROVISIONED")
            })
    return {"total_count": len(tables), "tables": tables}


def get_elb_load_balancers(region: str = "us-east-1") -> dict:
    client = boto3.client("elbv2", region_name=region)
    paginator = client.get_paginator("describe_load_balancers")
    lbs = []
    for page in paginator.paginate():
        for lb in page["LoadBalancers"]:
            lbs.append({
                "Name": lb["LoadBalancerName"],
                "Type": lb["Type"],
                "Scheme": lb["Scheme"],
                "State": lb["State"]["Code"],
                "DNSName": lb["DNSName"],
                "CreatedTime": str(lb.get("CreatedTime", "N/A"))
            })
    return {"total_count": len(lbs), "load_balancers": lbs}


def get_ec2_amis(region: str = "us-east-1") -> dict:
    ec2 = boto3.client("ec2", region_name=region)
    response = ec2.describe_images(Owners=["self"])
    images = []
    for img in response.get("Images", []):
        images.append({
            "ImageId": img["ImageId"],
            "Name": img.get("Name", "N/A"),
            "State": img["State"],
            "CreationDate": img.get("CreationDate", "N/A"),
            "Platform": img.get("PlatformDetails", "N/A"),
            "Architecture": img.get("Architecture", "N/A"),
            "VirtualizationType": img.get("VirtualizationType", "N/A")
        })
    return {"total_count": len(images), "images": images}


def get_security_groups(region: str = "us-east-1") -> dict:
    ec2 = boto3.client("ec2", region_name=region)
    paginator = ec2.get_paginator("describe_security_groups")
    groups = []
    for page in paginator.paginate():
        for sg in page["SecurityGroups"]:
            inbound_rules = len(sg.get("IpPermissions", []))
            outbound_rules = len(sg.get("IpPermissionsEgress", []))
            groups.append({
                "GroupId": sg["GroupId"],
                "GroupName": sg["GroupName"],
                "VpcId": sg.get("VpcId", "N/A"),
                "Description": sg.get("Description", ""),
                "InboundRules": inbound_rules,
                "OutboundRules": outbound_rules
            })
    return {"total_count": len(groups), "security_groups": groups}


def get_cost_and_usage(days: int = 30) -> dict:
    from datetime import datetime, timedelta
    ce = boto3.client("ce", region_name="us-east-1")
    end = datetime.utcnow().strftime("%Y-%m-%d")
    start = (datetime.utcnow() - timedelta(days=days)).strftime("%Y-%m-%d")
    response = ce.get_cost_and_usage(
        TimePeriod={"Start": start, "End": end},
        Granularity="MONTHLY",
        Metrics=["UnblendedCost"],
        GroupBy=[{"Type": "DIMENSION", "Key": "SERVICE"}]
    )
    services = []
    total = 0.0
    for result in response["ResultsByTime"]:
        for group in result["Groups"]:
            cost = float(group["Metrics"]["UnblendedCost"]["Amount"])
            if cost > 0:
                services.append({
                    "Service": group["Keys"][0],
                    "Cost_USD": round(cost, 4)
                })
                total += cost
    services.sort(key=lambda x: x["Cost_USD"], reverse=True)
    return {
        "period": f"{start} to {end}",
        "total_cost_USD": round(total, 2),
        "breakdown_by_service": services[:20]
    }


def get_account_summary(region: str = "us-east-1") -> dict:
    sts = boto3.client("sts")
    identity = sts.get_caller_identity()
    ec2 = boto3.client("ec2", region_name=region)
    s3 = boto3.client("s3", region_name="us-east-1")
    lambda_client = boto3.client("lambda", region_name=region)

    ec2_count = sum(
        len(r["Instances"])
        for page in ec2.get_paginator("describe_instances").paginate()
        for r in page["Reservations"]
    )
    s3_count = len(s3.list_buckets().get("Buckets", []))
    lambda_count = sum(
        len(p["Functions"])
        for p in lambda_client.get_paginator("list_functions").paginate()
    )

    return {
        "AccountId": identity["Account"],
        "Region": region,
        "EC2Instances": ec2_count,
        "S3Buckets": s3_count,
        "LambdaFunctions": lambda_count
    }


# ─────────────────────────────────────────────────────────────────────────────
# Tool Dispatcher
# ─────────────────────────────────────────────────────────────────────────────

TOOL_MAP = {
    "get_ec2_instances": get_ec2_instances,
    "get_lambda_functions": get_lambda_functions,
    "get_s3_buckets": get_s3_buckets,
    "get_rds_instances": get_rds_instances,
    "get_iam_users": get_iam_users,
    "get_cloudwatch_alarms": get_cloudwatch_alarms,
    "get_vpc_info": get_vpc_info,
    "get_ecs_clusters": get_ecs_clusters,
    "get_eks_clusters": get_eks_clusters,
    "get_cloudformation_stacks": get_cloudformation_stacks,
    "get_sns_topics": get_sns_topics,
    "get_sqs_queues": get_sqs_queues,
    "get_dynamodb_tables": get_dynamodb_tables,
    "get_elb_load_balancers": get_elb_load_balancers,
    "get_ec2_amis": get_ec2_amis,
    "get_security_groups": get_security_groups,
    "get_cost_and_usage": get_cost_and_usage,
    "get_account_summary": get_account_summary,
}


def execute_tool(tool_name: str, tool_input: dict) -> Any:
    """Execute the requested AWS tool and return the result."""
    if tool_name not in TOOL_MAP:
        return {"error": f"Unknown tool: {tool_name}"}
    try:
        logger.info(f"Executing tool: {tool_name} with input: {tool_input}")
        result = TOOL_MAP[tool_name](**tool_input)
        logger.info(f"Tool result count: {result.get('total_count', 'N/A')}")
        return result
    except Exception as e:
        logger.error(f"Tool execution error: {str(e)}")
        return {"error": str(e), "tool": tool_name}


# ─────────────────────────────────────────────────────────────────────────────
# Agentic Loop – Claude decides which tools to call
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are an expert AWS Cloud Assistant with deep knowledge of all AWS services.
You have access to tools that can query real-time AWS infrastructure data.

Your responsibilities:
- Answer questions about any AWS resource: EC2, Lambda, S3, RDS, IAM, VPC, ECS, EKS, DynamoDB, etc.
- Use tools to fetch live data from the AWS account before answering
- Provide clear, structured answers with exact counts, IDs, and relevant details
- Proactively call the account summary tool for broad "overview" questions
- For cost questions, use the cost tool
- Always use tools to get real data rather than guessing

Response style:
- Be concise and data-focused
- Present counts prominently (e.g., "You have 12 EC2 instances")
- Use tables or lists when listing multiple resources
- If a query spans multiple services, call multiple tools
"""


def chat_with_claude(user_message: str, conversation_history: list = None) -> dict:
    """
    Main agentic loop. Sends message to Claude, handles tool calls,
    and returns the final response.
    """
    if conversation_history is None:
        conversation_history = []

    # Add the new user message
    messages = conversation_history + [{"role": "user", "content": user_message}]

    max_iterations = 10  # Prevent infinite loops
    iteration = 0

    while iteration < max_iterations:
        iteration += 1
        logger.info(f"Bedrock API call iteration {iteration}")

        response = bedrock_runtime.invoke_model(
            modelId=MODEL_ID,
            body=json.dumps({
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 4096,
                "system": SYSTEM_PROMPT,
                "tools": TOOLS,
                "messages": messages
            })
        )

        response_body = json.loads(response["body"].read())
        stop_reason = response_body.get("stop_reason")
        content = response_body.get("content", [])

        logger.info(f"Stop reason: {stop_reason}")

        # If Claude is done (no more tool calls), return the final text
        if stop_reason == "end_turn":
            final_text = " ".join(
                block["text"] for block in content if block.get("type") == "text"
            )
            return {
                "response": final_text,
                "messages": messages + [{"role": "assistant", "content": content}]
            }

        # Claude wants to use tools
        if stop_reason == "tool_use":
            # Add Claude's response to message history
            messages.append({"role": "assistant", "content": content})

            # Execute all requested tools
            tool_results = []
            for block in content:
                if block.get("type") == "tool_use":
                    tool_result = execute_tool(block["name"], block.get("input", {}))
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block["id"],
                        "content": json.dumps(tool_result, default=str)
                    })

            # Send tool results back to Claude
            messages.append({"role": "user", "content": tool_results})
            continue

        # Unexpected stop reason
        break

    return {"response": "I encountered an issue processing your request.", "messages": messages}


# ─────────────────────────────────────────────────────────────────────────────
# Lambda Handler
# ─────────────────────────────────────────────────────────────────────────────

def lambda_handler(event: dict, context) -> dict:
    """
    Lambda entry point.
    
    Expected event body (JSON):
    {
        "message": "How many EC2 instances do I have?",
        "conversation_history": []   // optional, for multi-turn chat
    }
    """
    headers = {
        "Content-Type": "application/json",
        "Access-Control-Allow-Origin": "*",           # Update for production
        "Access-Control-Allow-Headers": "Content-Type,Authorization",
        "Access-Control-Allow-Methods": "POST,OPTIONS"
    }

    # Handle CORS preflight
    if event.get("httpMethod") == "OPTIONS":
        return {"statusCode": 200, "headers": headers, "body": ""}

    try:
        # Parse body
        body = event.get("body", "{}")
        if isinstance(body, str):
            body = json.loads(body)

        user_message = body.get("message", "").strip()
        conversation_history = body.get("conversation_history", [])

        if not user_message:
            return {
                "statusCode": 400,
                "headers": headers,
                "body": json.dumps({"error": "message field is required"})
            }

        logger.info(f"User message: {user_message[:200]}")

        # Run the agentic chat loop
        result = chat_with_claude(user_message, conversation_history)

        return {
            "statusCode": 200,
            "headers": headers,
            "body": json.dumps({
                "response": result["response"],
                "conversation_history": result["messages"]
            })
        }

    except Exception as e:
        logger.error(f"Lambda error: {str(e)}", exc_info=True)
        return {
            "statusCode": 500,
            "headers": headers,
            "body": json.dumps({"error": "Internal server error", "detail": str(e)})
        }
