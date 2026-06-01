from kubernetes import client


# Namespace operator đang tạo resource.
NAMESPACE = "plagiarism"

# Internal services/configs.
MINIO_ENDPOINT = "minio-svc-private.storage.svc.cluster.local:9000"
POSTGRES_DSN = "postgresql://postgres:postgres@postgresdb.streaming.svc.cluster.local:5432/plagiarism"

MILVUS_HOST = "milvus.milvus.svc.cluster.local"
MILVUS_PORT = "19530"
EMBEDDING_MODEL = "/models/vietnamese-sbert"

# Image mới theo kiến trúc:
# Stage 1: preprocess-server giữ sống bằng FastAPI/Uvicorn.
# Stage 2: compare-worker gọi HTTP sang Stage 1 để lấy dữ liệu.
PREPROCESS_SERVER_IMAGE = "baoghetcode/preprocess-server:1.0"
COMPARE_WORKER_IMAGE = "baoghetcode/compare-worker:1.4"


def minio_env_vars() -> list[client.V1EnvVar]:
    return [
        client.V1EnvVar(
            name="MINIO_ENDPOINT",
            value=MINIO_ENDPOINT,
        ),
        client.V1EnvVar(
            name="MINIO_ACCESS_KEY",
            value_from=client.V1EnvVarSource(
                secret_key_ref=client.V1SecretKeySelector(
                    name="minio-credentials",
                    key="access-key",
                )
            ),
        ),
        client.V1EnvVar(
            name="MINIO_SECRET_KEY",
            value_from=client.V1EnvVarSource(
                secret_key_ref=client.V1SecretKeySelector(
                    name="minio-credentials",
                    key="secret-key",
                )
            ),
        ),
    ]


def build_preprocess_server_service(name: str) -> client.V1Service:
    """
    Service nội bộ để các compare-worker Pod gọi HTTP sang preprocess-server.

    DNS sẽ là:
    http://preprocess-server-<check-name>.plagiarism.svc.cluster.local:8000
    """

    service_name = f"preprocess-server-{name}"

    labels = {
        "app": "preprocess-server",
        "owner": "plagiarism-operator",
        "stage": "preprocess-server",
        "check-name": name,
    }

    return client.V1Service(
        metadata=client.V1ObjectMeta(
            name=service_name,
            labels=labels,
        ),
        spec=client.V1ServiceSpec(
            type="ClusterIP",
            selector={
                "app": "preprocess-server",
                "check-name": name,
            },
            ports=[
                client.V1ServicePort(
                    name="http",
                    port=8000,
                    target_port=8000,
                )
            ],
        ),
    )


def build_preprocess_server_pod(
    name: str,
    subject_id: str,
    input_pdf: str,
    result_output_path: str,
    expected_parts: int,
) -> client.V1Pod:
    """
    Stage 1 Pod.

    Pod này không exit sau khi preprocess xong.
    Nó chạy FastAPI/Uvicorn, giữ sentences/candidates trong RAM,
    và expose API:
    - GET /ready
    - GET /part
    - POST /part-result
    """

    pod_name = f"preprocess-server-{name}"

    labels = {
        "app": "preprocess-server",
        "owner": "plagiarism-operator",
        "stage": "preprocess-server",
        "check-name": name,
    }

    return client.V1Pod(
        metadata=client.V1ObjectMeta(
            name=pod_name,
            labels=labels,
        ),
        spec=client.V1PodSpec(
            restart_policy="Never",
            containers=[
                client.V1Container(
                    name="preprocess-server",
                    image=PREPROCESS_SERVER_IMAGE,
                    image_pull_policy="IfNotPresent",
                    ports=[
                        client.V1ContainerPort(
                            container_port=8000,
                        )
                    ],
                    env=[
                        client.V1EnvVar(
                            name="SUBJECT_ID",
                            value=subject_id,
                        ),
                        client.V1EnvVar(
                            name="INPUT_PDF_PATH",
                            value=input_pdf,
                        ),
                        client.V1EnvVar(
                            name="RESULT_OUTPUT_PATH",
                            value=result_output_path,
                        ),
                        client.V1EnvVar(
                            name="EXPECTED_PARTS",
                            value=str(expected_parts),
                        ),
                        client.V1EnvVar(
                            name="POSTGRES_DSN",
                            value=POSTGRES_DSN,
                        ),
                        client.V1EnvVar(
                            name="MINHASH_THRESHOLD",
                            value="0.05",
                        ),
                        *minio_env_vars(),
                    ],
                    readiness_probe=client.V1Probe(
                        http_get=client.V1HTTPGetAction(
                            path="/ready",
                            port=8000,
                        ),
                        initial_delay_seconds=5,
                        period_seconds=5,
                        failure_threshold=120,
                    ),
                    resources=client.V1ResourceRequirements(
                        requests={
                            "cpu": "500m",
                            "memory": "1Gi",
                        },
                        limits={
                            "cpu": "2",
                            "memory": "4Gi",
                        },
                    ),
                )
            ],
        ),
    )


def build_compare_job(
    name: str,
    part_index: int,
    part_count: int,
) -> client.V1Job:
    """
    Stage 2 Job.

    Mỗi compare job:
    - gọi GET /part?part_index=X&part_count=N
    - nhận sentences + candidate slice
    - check đạo văn
    - POST /part-result về preprocess-server
    - exit 0
    """

    job_name = f"compare-{name}-part-{part_index}"

    labels = {
        "app": "compare-worker",
        "owner": "plagiarism-operator",
        "stage": "compare",
        "check-name": name,
        "part-index": str(part_index),
    }

    stage1_url = (
        f"http://preprocess-server-{name}."
        f"{NAMESPACE}.svc.cluster.local:8000"
    )

    return client.V1Job(
        metadata=client.V1ObjectMeta(
            name=job_name,
            labels=labels,
        ),
        spec=client.V1JobSpec(
            backoff_limit=1,
            ttl_seconds_after_finished=86400,
            template=client.V1PodTemplateSpec(
                metadata=client.V1ObjectMeta(
                    labels=labels,
                ),
                spec=client.V1PodSpec(
                    restart_policy="Never",
                    containers=[
                        client.V1Container(
                            name="compare-worker",
                            image=COMPARE_WORKER_IMAGE,
                            image_pull_policy="IfNotPresent",
                            env=[
                                client.V1EnvVar(
                                    name="STAGE1_URL",
                                    value=stage1_url,
                                ),
                                client.V1EnvVar(
                                    name="PART_INDEX",
                                    value=str(part_index),
                                ),
                                client.V1EnvVar(
                                    name="PART_COUNT",
                                    value=str(part_count),
                                ),
                                client.V1EnvVar(
                                    name="MILVUS_HOST",
                                    value=MILVUS_HOST,
                                ),
                                client.V1EnvVar(
                                    name="MILVUS_PORT",
                                    value=MILVUS_PORT,
                                ),
                                client.V1EnvVar(
                                    name="EMBEDDING_MODEL",
                                    value=EMBEDDING_MODEL,
                                ),
                                client.V1EnvVar(
                                    name="POSTGRES_DSN",
                                    value=POSTGRES_DSN,
                                ),
                                client.V1EnvVar(
                                    name="MINIO_BUCKET",
                                    value="uploads",
                                ),
                                *minio_env_vars(),
                                client.V1EnvVar(
                                    name="HF_HUB_DISABLE_XET",
                                    value="1",
                                ),

                            ],
                            resources=client.V1ResourceRequirements(
                                requests={
                                    "cpu": "500m",
                                    "memory": "1Gi",
                                },
                                limits={
                                    "cpu": "2",
                                    "memory": "4Gi",
                                },
                            ),
                        )
                    ],
                ),
            ),
        ),
    )
