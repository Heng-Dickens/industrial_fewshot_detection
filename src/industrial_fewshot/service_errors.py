"""Identify service-wide errors that cannot be fixed by trying another image."""


class ModelServiceError(RuntimeError):
    def __init__(self, message, *, status_code=None, record=None):
        super().__init__(message)
        self.status_code = status_code
        self.record = record


def terminal_service_error(exc):
    status = getattr(exc, 'status_code', None)
    if status is None:
        status = getattr(getattr(exc, 'response', None), 'status_code', None)
    if status in {401, 403, 404}:
        return ModelServiceError(
            f'模型服务返回 HTTP {status}，已停止本次实验，不再逐图重试。'
            '请检查 CC Switch 上游鉴权、转发供应商及模型地址。', status_code=status)
    return None


def connection_failure(exc):
    return isinstance(exc, (ConnectionError, TimeoutError)) or any(
        cls.__name__ in {'APIConnectionError', 'APITimeoutError'} for cls in type(exc).__mro__)
