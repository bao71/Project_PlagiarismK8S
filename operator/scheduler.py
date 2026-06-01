from kubernetes import client


MAX_PREPROCESS_SERVERS = 3
DEFAULT_COMPARE_PODS = 5

GROUP = "plagiarism.io"
VERSION = "v1"
PLURAL = "plagiarismchecks"


PHASE_PENDING_PREPROCESS_SERVER = "PENDING_PREPROCESS_SERVER"
PHASE_PREPROCESS_SERVER_STARTING = "PREPROCESS_SERVER_STARTING"
PHASE_PREPROCESS_READY = "PREPROCESS_READY"
PHASE_COMPARING = "COMPARING"
PHASE_CHECK_SUCCEEDED = "CHECK_SUCCEEDED"
PHASE_CHECK_FAILED = "CHECK_FAILED"


def get_checks_by_phase(
    namespace: str,
    phase: str,
    include_missing_phase: bool = False,
) -> list[dict]:
    api = client.CustomObjectsApi()

    objs = api.list_namespaced_custom_object(
        group=GROUP,
        version=VERSION,
        namespace=namespace,
        plural=PLURAL,
    )

    matched: list[dict] = []

    for item in objs.get("items", []):
        status = item.get("status", {})
        current_phase = status.get("phase")

        if current_phase == phase:
            matched.append(item)
            continue

        if include_missing_phase and current_phase is None:
            matched.append(item)

    matched.sort(
        key=lambda x: x["metadata"]["creationTimestamp"]
    )

    return matched


def get_pending_preprocess_server_checks(namespace: str) -> list[dict]:
    return get_checks_by_phase(
        namespace=namespace,
        phase=PHASE_PENDING_PREPROCESS_SERVER,
        include_missing_phase=True,
    )


def get_preprocess_ready_checks(namespace: str) -> list[dict]:
    return get_checks_by_phase(
        namespace=namespace,
        phase=PHASE_PREPROCESS_READY,
    )


def count_running_pods_by_labels(
    namespace: str,
    required_labels: dict[str, str],
) -> int:
    core_api = client.CoreV1Api()

    pods = core_api.list_namespaced_pod(
        namespace=namespace,
    )

    count = 0

    for pod in pods.items:
        labels = pod.metadata.labels or {}

        matched = all(
            labels.get(key) == value
            for key, value in required_labels.items()
        )

        if not matched:
            continue

        if pod.status.phase in ("Pending", "Running"):
            count += 1

    return count


def count_running_preprocess_server_pods(namespace: str) -> int:
    return count_running_pods_by_labels(
        namespace=namespace,
        required_labels={
            "owner": "plagiarism-operator",
            "stage": "preprocess-server",
        },
    )
