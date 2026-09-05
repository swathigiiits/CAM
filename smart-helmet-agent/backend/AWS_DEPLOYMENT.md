# ─── AWS Deployment Guide ────────────────────────────────────────────────────
# Smart Helmet Attention Analytics Agent

## Architecture

┌─────────────────────────────────────────────────────────────┐
│                        AWS Architecture                      │
│                                                             │
│  CloudFront ──► S3 (Frontend)                              │
│       │                                                     │
│       ▼                                                     │
│  ALB ──► ECS Fargate (FastAPI Backend)                     │
│              │                                              │
│              ├── S3 (Video/CSV Storage)                    │
│              ├── DynamoDB (Sessions DB)                    │
│              ├── SageMaker (ML Models)                     │
│              └── Bedrock / Anthropic API (LLM Coaching)    │
└─────────────────────────────────────────────────────────────┘

## Environment Variables (set in ECS Task Definition)

```
S3_BUCKET=smart-helmet-sessions
AWS_REGION=us-east-1
ANTHROPIC_API_KEY=<your-key>
DYNAMODB_TABLE=smart-helmet-sessions
```

## Deploy Backend to ECS

```bash
# 1. Build & push Docker image
aws ecr create-repository --repository-name smart-helmet-backend
docker build -t smart-helmet-backend .
docker tag smart-helmet-backend:latest <account>.dkr.ecr.us-east-1.amazonaws.com/smart-helmet-backend:latest
aws ecr get-login-password | docker login --username AWS --password-stdin <account>.dkr.ecr.us-east-1.amazonaws.com
docker push <account>.dkr.ecr.us-east-1.amazonaws.com/smart-helmet-backend:latest

# 2. Create ECS cluster and service
aws ecs create-cluster --cluster-name smart-helmet-cluster

# Use task-definition.json (see below) to register task
aws ecs register-task-definition --cli-input-json file://task-definition.json
```

## task-definition.json

```json
{
  "family": "smart-helmet-backend",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "1024",
  "memory": "2048",
  "containerDefinitions": [
    {
      "name": "backend",
      "image": "<account>.dkr.ecr.us-east-1.amazonaws.com/smart-helmet-backend:latest",
      "portMappings": [{"containerPort": 8000}],
      "environment": [
        {"name": "S3_BUCKET", "value": "smart-helmet-sessions"},
        {"name": "AWS_REGION", "value": "us-east-1"}
      ],
      "secrets": [
        {
          "name": "ANTHROPIC_API_KEY",
          "valueFrom": "arn:aws:ssm:us-east-1:<account>:parameter/smart-helmet/anthropic-key"
        }
      ]
    }
  ]
}
```

## Deploy Frontend to S3 + CloudFront

```bash
cd frontend
npm run build
aws s3 sync dist/ s3://smart-helmet-frontend --delete
aws cloudfront create-invalidation --distribution-id <DIST_ID> --paths "/*"
```

## S3 Bucket Policy (CORS for video uploads)

```json
{
  "CORSRules": [
    {
      "AllowedHeaders": ["*"],
      "AllowedMethods": ["GET", "PUT", "POST", "DELETE"],
      "AllowedOrigins": ["https://your-domain.com"],
      "MaxAgeSeconds": 3000
    }
  ]
}
```

## DynamoDB Table (production sessions store)

Replace in-memory `sessions_db` with:
- Table: `smart-helmet-sessions`
- Partition key: `session_id` (String)
- GSI: `driver_id-timestamp-index`
