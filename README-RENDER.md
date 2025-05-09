# Deploying on Render.com

This document outlines how to deploy this project on Render.com.

## Deployment Options

### Option 1: Blueprint Deployment (Recommended)

1. Fork this repository to your GitHub/GitLab account
2. Sign up or log in to [Render.com](https://render.com)
3. Click "New" > "Blueprint"
4. Connect your repository
5. Render will use the `render.yaml` file to set up all services

### Option 2: Manual Deployment

If you prefer to deploy services manually:

1. **Web Service**:
   - Create a new Web Service
   - Connect your repository
   - Set build command: `pip install -r requirements.txt`
   - Set start command: `gunicorn -c gunicorn_config.py main:app`

2. **Worker Service**:
   - Create a new Background Worker
   - Connect your repository
   - Set build command: `pip install -r requirements.txt`
   - Set start command: `celery -A celery_worker worker --loglevel=info`

3. **Databases**:
   - Create PostgreSQL and Redis instances

## Required Environment Variables

Set these in your Render dashboard:

- `OPENAI_API_KEY`: Your OpenAI API key
- `DATABASE_URL`: Generated by Render for PostgreSQL
- `REDIS_URL`: Generated by Render for Redis
- `MILVUS_HOST`, `MILVUS_PORT`: For Milvus vector database
- `NEBULA_HOST`, `NEBULA_PORT`, `NEBULA_USER`, `NEBULA_PASSWORD`: For Nebula graph database
- `JWT_SECRET_KEY`: Secret key for JWT tokens
- `ENV`: Set to "production"

## Considerations for Vector and Graph Databases

For production use:
- Consider using [Zilliz Cloud](https://zilliz.com) for managed Milvus
- Consider using [NebulaGraph Cloud](https://nebula-graph.io) for managed Nebula

Update connection strings in environment variables accordingly.

## Scaling

- Web service and worker can be scaled in Render dashboard
- For high traffic, upgrade to higher-tier plans
- Monitor performance and scale as needed 