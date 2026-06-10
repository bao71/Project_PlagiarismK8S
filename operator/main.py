import logging

import kopf

from kubernetes import client, config
from kubernetes.client.exceptions import ApiException

from scheduler import (
    DEFAULT_COMPARE_PODS,
    MAX_PREPROCESS_SERVERS,
    get_pending_preprocess_server_checks,
    get_preprocess_ready_checks,
    count_running_preprocess_server_pods,
    PHASE_PENDING_PREPROCESS_SERVER,
    PHASE_PREPROCESS_SERVER_STARTING,
    PHASE_PREPROCESS_READY,
    PHASE_COMPARING,
    PHASE_CHECK_SUCCEEDED,
    PHASE_CHECK_FAILED,
)

from job_builder import (
    build_preprocess_server_pod,
    build_preprocess_server_service,
    build_compare_job,
)


try:
    config.load_incluster_config()
except Exception:
    config.load_kube_config()


GROUP = "plagiarism.io"
VERSION = "v1"
PLURAL = "plagiarismchecks"
NAMESPACE = "plagiarism"


# =====================================================
# PATH / SPEC HELPERS
# =====================================================

def default_result_path(name: str) -> str:
    return f"minio://results/{name}/result.json"


def get_result_output_path(
    name: str,
    spec: dict,
    status: dict | None = None,
) -> str:
    status = status or {}

    return (
        status.get("resultPath")
        or spec.get("outputResult")
        or default_result_path(name)
    )


def get_compare_part_count(
    spec: dict,
    status: dict | None = None,
) -> int:
    status = status or {}

    raw = (
        status.get("comparePartCount")
        or spec.get("comparePods")
        or DEFAULT_COMPARE_PODS
    )

    try:
        value = int(raw)
    except Exception:
        value = DEFAULT_COMPARE_PODS

    if value <= 0:
        value = DEFAULT_COMPARE_PODS

    return value


# =====================================================
# STATUS HELPER
# =====================================================

def patch_check_status(
    name: str,
    phase: str,
    message: str | None = None,
    extra: dict | None = None,
):
    api = client.CustomObjectsApi()

    new_status = {
        "phase": phase,
    }

    if message:
        new_status["message"] = message

    if extra:
        new_status.update(extra)

    body = {
        "status": new_status,
    }

    try:
        api.patch_namespaced_custom_object_status(
            group=GROUP,
            version=VERSION,
            namespace=NAMESPACE,
            plural=PLURAL,
            name=name,
            body=body,
        )

    except ApiException:
        # Fallback nếu CRD chưa bật status subresource.
        api.patch_namespaced_custom_object(
            group=GROUP,
            version=VERSION,
            namespace=NAMESPACE,
            plural=PLURAL,
            name=name,
            body=body,
        )


# =====================================================
# K8S RESOURCE HELPERS
# =====================================================

def pod_exists(
    core_api: client.CoreV1Api,
    namespace: str,
    pod_name: str,
) -> bool:
    try:
        core_api.read_namespaced_pod(
            name=pod_name,
            namespace=namespace,
        )
        return True

    except ApiException as e:
        if e.status == 404:
            return False

        raise


def service_exists(
    core_api: client.CoreV1Api,
    namespace: str,
    service_name: str,
) -> bool:
    try:
        core_api.read_namespaced_service(
            name=service_name,
            namespace=namespace,
        )
        return True

    except ApiException as e:
        if e.status == 404:
            return False

        raise


def job_exists(
    batch_api: client.BatchV1Api,
    namespace: str,
    job_name: str,
) -> bool:
    try:
        batch_api.read_namespaced_job(
            name=job_name,
            namespace=namespace,
        )
        return True

    except ApiException as e:
        if e.status == 404:
            return False

        raise


def is_pod_ready(pod) -> bool:
    conditions = pod.status.conditions or []

    for condition in conditions:
        if condition.type == "Ready" and condition.status == "True":
            return True

    return False


def get_pod_failed_reason(pod) -> str | None:
    if pod.status.phase == "Failed":
        return "Pod phase is Failed"

    container_statuses = pod.status.container_statuses or []

    for cs in container_statuses:
        state = cs.state

        if state and state.terminated and state.terminated.exit_code != 0:
            return (
                f"Container {cs.name} terminated with "
                f"exit_code={state.terminated.exit_code}, "
                f"reason={state.terminated.reason}"
            )

    return None


def get_job_condition(job) -> str | None:
    conditions = job.status.conditions or []

    for condition in conditions:
        if condition.type == "Complete" and condition.status == "True":
            return "Complete"

        if condition.type == "Failed" and condition.status == "True":
            return "Failed"

    if job.status.succeeded and job.status.succeeded >= 1:
        return "Complete"

    if job.status.failed and job.status.failed >= 1:
        return "Failed"

    return None


def list_compare_jobs(
    namespace: str,
    name: str,
):
    batch_api = client.BatchV1Api()

    return batch_api.list_namespaced_job(
        namespace=namespace,
        label_selector=(
            f"owner=plagiarism-operator,"
            f"stage=compare,"
            f"check-name={name}"
        ),
    ).items


def cleanup_resources(
    name: str,
    logger,
):
    core_api = client.CoreV1Api()
    batch_api = client.BatchV1Api()

    # Delete compare jobs.
    try:
        jobs = list_compare_jobs(
            namespace=NAMESPACE,
            name=name,
        )

        for job in jobs:
            try:
                batch_api.delete_namespaced_job(
                    name=job.metadata.name,
                    namespace=NAMESPACE,
                    propagation_policy="Background",
                )
                logger.info(f"Deleted compare job {job.metadata.name}")

            except ApiException as e:
                if e.status != 404:
                    raise

    except ApiException as e:
        if e.status != 404:
            raise

    # Delete preprocess-server Pod.
    pod_name = f"preprocess-server-{name}"

    try:
        core_api.delete_namespaced_pod(
            name=pod_name,
            namespace=NAMESPACE,
            grace_period_seconds=0,
        )
        logger.info(f"Deleted pod {pod_name}")

    except ApiException as e:
        if e.status != 404:
            raise

    # Delete preprocess-server Service.
    service_name = f"preprocess-server-{name}"

    try:
        core_api.delete_namespaced_service(
            name=service_name,
            namespace=NAMESPACE,
        )
        logger.info(f"Deleted service {service_name}")

    except ApiException as e:
        if e.status != 404:
            raise


# =====================================================
# CREATE EVENT
# =====================================================

@kopf.on.create(
    GROUP,
    VERSION,
    PLURAL,
)
def on_create(
    patch,
    meta,
    logger,
    **kwargs,
):
    name = meta["name"]

    logger.info(f"New check created: {name}")

    patch.status["phase"] = PHASE_PENDING_PREPROCESS_SERVER
    patch.status["message"] = "Waiting for preprocess server"


# =====================================================
# STARTUP
# =====================================================

@kopf.on.startup()
def configure(
    settings: kopf.OperatorSettings,
    **_,
):
    settings.posting.level = logging.INFO


# =====================================================
# STAGE 1 - CREATE PREPROCESS SERVER POD + SERVICE
# =====================================================

@kopf.timer(
    GROUP,
    VERSION,
    PLURAL,
    interval=5,
)
def dispatch_preprocess_server(
    logger,
    **kwargs,
):
    namespace = NAMESPACE

    running = count_running_preprocess_server_pods(namespace)
    available_slots = MAX_PREPROCESS_SERVERS - running

    logger.info(
        f"Preprocess servers running={running}, "
        f"available={available_slots}"
    )

    if available_slots <= 0:
        return

    pending_checks = get_pending_preprocess_server_checks(namespace)

    if not pending_checks:
        return

    core_api = client.CoreV1Api()

    for item in pending_checks[:available_slots]:
        name = item["metadata"]["name"]
        spec = item.get("spec", {})

        subject_id = spec.get("subjectId")
        input_pdf = spec.get("originalFilePath")
        
        
        if not subject_id:
            patch_check_status(
                name=name,
                phase="PREPROCESS_SERVER_DISPATCH_FAILED",
                message="Missing spec.subjectId",
            )
            continue

        if not input_pdf:
            patch_check_status(
                name=name,
                phase="PREPROCESS_SERVER_DISPATCH_FAILED",
                message="Missing spec.originalFilePath",
            )
            continue

        result_path = get_result_output_path(
            name=name,
            spec=spec,
        )

        compare_part_count = get_compare_part_count(
            spec=spec,
        )

        pod_name = f"preprocess-server-{name}"
        service_name = f"preprocess-server-{name}"

        stage1_url = (
            f"http://{service_name}."
            f"{namespace}.svc.cluster.local:8000"
        )

        try:
            if not service_exists(
                core_api=core_api,
                namespace=namespace,
                service_name=service_name,
            ):
                service = build_preprocess_server_service(
                    name=name,
                )

                core_api.create_namespaced_service(
                    namespace=namespace,
                    body=service,
                )

                logger.info(f"Created service {service_name}")

            if not pod_exists(
                core_api=core_api,
                namespace=namespace,
                pod_name=pod_name,
            ):
                pod = build_preprocess_server_pod(
                    name=name,
                    subject_id=subject_id,
                    input_pdf=input_pdf,
                    result_output_path=result_path,
                    expected_parts=compare_part_count,
                )

                core_api.create_namespaced_pod(
                    namespace=namespace,
                    body=pod,
                )

                logger.info(f"Created preprocess server pod {pod_name}")

            patch_check_status(
                name=name,
                phase=PHASE_PREPROCESS_SERVER_STARTING,
                message="Preprocess server is starting",
                extra={
                    "preprocessServerPodName": pod_name,
                    "preprocessServerServiceName": service_name,
                    "stage1Url": stage1_url,
                    "comparePartCount": compare_part_count,
                    "resultPath": result_path,
                },
            )

        except ApiException as e:
            logger.exception(e)

            patch_check_status(
                name=name,
                phase="PREPROCESS_SERVER_DISPATCH_FAILED",
                message=str(e),
                extra={
                    "preprocessServerPodName": pod_name,
                    "preprocessServerServiceName": service_name,
                    "stage1Url": stage1_url,
                    "comparePartCount": compare_part_count,
                    "resultPath": result_path,
                },
            )


# =====================================================
# STAGE 1 - WAIT FOR PREPROCESS SERVER READY
# =====================================================

@kopf.timer(
    GROUP,
    VERSION,
    PLURAL,
    interval=5,
)
def monitor_preprocess_server(
    meta,
    status,
    logger,
    **kwargs,
):
    name = meta["name"]
    phase = status.get("phase")

    if phase != PHASE_PREPROCESS_SERVER_STARTING:
        return

    pod_name = status.get(
        "preprocessServerPodName",
        f"preprocess-server-{name}",
    )

    core_api = client.CoreV1Api()

    try:
        pod = core_api.read_namespaced_pod(
            name=pod_name,
            namespace=NAMESPACE,
        )

    except ApiException as e:
        if e.status == 404:
            patch_check_status(
                name=name,
                phase="PREPROCESS_SERVER_MISSING",
                message=f"Preprocess server pod {pod_name} not found",
            )
            return

        raise

    failed_reason = get_pod_failed_reason(pod)

    if failed_reason:
        patch_check_status(
            name=name,
            phase="PREPROCESS_SERVER_FAILED",
            message=failed_reason,
            extra={
                "preprocessServerPodName": pod_name,
            },
        )
        return

    if is_pod_ready(pod):
        patch_check_status(
            name=name,
            phase=PHASE_PREPROCESS_READY,
            message="Preprocess server is ready",
            extra={
                "preprocessServerPodName": pod_name,
                "preprocessServerServiceName": status.get(
                    "preprocessServerServiceName",
                    f"preprocess-server-{name}",
                ),
                "stage1Url": status.get(
                    "stage1Url",
                    f"http://preprocess-server-{name}.{NAMESPACE}.svc.cluster.local:8000",
                ),
                "comparePartCount": status.get(
                    "comparePartCount",
                    DEFAULT_COMPARE_PODS,
                ),
                "resultPath": status.get(
                    "resultPath",
                    default_result_path(name),
                ),
            },
        )

        logger.info(f"Preprocess server is ready for {name}")
        return

    logger.info(
        f"Preprocess server {pod_name} is not ready yet. "
        f"phase={pod.status.phase}"
    )


# =====================================================
# STAGE 2 - CREATE COMPARE JOBS
# =====================================================

@kopf.timer(
    GROUP,
    VERSION,
    PLURAL,
    interval=5,
)
def dispatch_compare_jobs(
    logger,
    **kwargs,
):
    namespace = NAMESPACE

    ready_checks = get_preprocess_ready_checks(namespace)

    if not ready_checks:
        return

    batch_api = client.BatchV1Api()

    for item in ready_checks:
        name = item["metadata"]["name"]
        status = item.get("status", {})
        spec = item.get("spec", {})

        part_count = get_compare_part_count(
            spec=spec,
            status=status,
        )

        created_jobs = []

        for part_index in range(part_count):
            job_name = f"compare-{name}-part-{part_index}"

            try:
                if job_exists(
                    batch_api=batch_api,
                    namespace=namespace,
                    job_name=job_name,
                ):
                    created_jobs.append(job_name)
                    continue

                job = build_compare_job(
                    name=name,
                    part_index=part_index,
                    part_count=part_count,
                )

                batch_api.create_namespaced_job(
                    namespace=namespace,
                    body=job,
                )

                created_jobs.append(job_name)
                logger.info(f"Created compare job {job_name}")

            except ApiException as e:
                logger.exception(e)

                patch_check_status(
                    name=name,
                    phase="COMPARE_DISPATCH_FAILED",
                    message=str(e),
                    extra={
                        "comparePartCount": part_count,
                        "compareJobNames": created_jobs,
                    },
                )
                break

        else:
            patch_check_status(
                name=name,
                phase=PHASE_COMPARING,
                message="Compare jobs are running",
                extra={
                    "comparePartCount": part_count,
                    "compareJobNames": created_jobs,
                    "resultPath": status.get(
                        "resultPath",
                        default_result_path(name),
                    ),
                    "stage1Url": status.get(
                        "stage1Url",
                        f"http://preprocess-server-{name}.{namespace}.svc.cluster.local:8000",
                    ),
                },
            )


# =====================================================
# STAGE 2 - MONITOR COMPARE JOBS
# =====================================================

@kopf.timer(
    GROUP,
    VERSION,
    PLURAL,
    interval=5,
)
def monitor_compare_jobs(
    meta,
    status,
    logger,
    **kwargs,
):
    name = meta["name"]
    phase = status.get("phase")

    if phase != PHASE_COMPARING:
        return

    part_count = int(
        status.get(
            "comparePartCount",
            DEFAULT_COMPARE_PODS,
        )
    )

    jobs = list_compare_jobs(
        namespace=NAMESPACE,
        name=name,
    )

    if len(jobs) < part_count:
        logger.info(
            f"Compare jobs for {name}: "
            f"{len(jobs)}/{part_count} created"
        )
        return

    complete = 0
    failed = 0
    active = 0

    for job in jobs:
        condition = get_job_condition(job)

        if condition == "Complete":
            complete += 1
            continue

        if condition == "Failed":
            failed += 1
            continue

        if job.status.active:
            active += 1

    logger.info(
        f"Compare status for {name}: "
        f"complete={complete}, failed={failed}, "
        f"active={active}, expected={part_count}"
    )

    if failed > 0:
        patch_check_status(
            name=name,
            phase=PHASE_CHECK_FAILED,
            message="At least one compare job failed",
            extra={
                "compareComplete": complete,
                "compareFailed": failed,
                "compareActive": active,
                "resultPath": status.get(
                    "resultPath",
                    default_result_path(name),
                ),
            },
        )
        return

    if complete >= part_count:
        patch_check_status(
            name=name,
            phase=PHASE_CHECK_SUCCEEDED,
            message="All compare jobs completed successfully",
            extra={
                "compareComplete": complete,
                "compareFailed": failed,
                "compareActive": active,
                "resultPath": status.get(
                    "resultPath",
                    default_result_path(name),
                ),
                "preprocessServerPodName": status.get(
                    "preprocessServerPodName",
                    f"preprocess-server-{name}",
                ),
                "preprocessServerServiceName": status.get(
                    "preprocessServerServiceName",
                    f"preprocess-server-{name}",
                ),
            },
        )

        # Không cleanup tự động để bạn còn xem log Stage 1.
        # Muốn tự cleanup khi thành công thì bật dòng dưới:
        # cleanup_resources(name=name, logger=logger)

        logger.info(f"Check succeeded for {name}")
        return


# =====================================================
# DELETE EVENT
# =====================================================

@kopf.on.delete(
    GROUP,
    VERSION,
    PLURAL,
)
def on_delete(
    meta,
    logger,
    **kwargs,
):
    name = meta["name"]

    cleanup_resources(
        name=name,
        logger=logger,
    )


# =====================================================
# MAIN
# =====================================================

if __name__ == "__main__":
    kopf.run()
