FROM milvusdb/milvus:v2.2.11

# Copy configuration files if needed
# COPY milvus.yaml /milvus/configs/milvus.yaml

# Environment variables
ENV ETCD_ENDPOINTS=localhost:2379
ENV MINIO_ADDRESS=localhost:9000

# Expose ports
EXPOSE 19530 9091

# Health check
HEALTHCHECK --interval=5s --timeout=5s --retries=3 CMD wget -q -O - http://localhost:9091/healthz || exit 1

# Default command
CMD ["milvus", "run", "standalone"] 