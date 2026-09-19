"""Google Cloud Storage 연결 관리 및 범용 파일 유틸리티.

사용 예
-------
업로드 (바이트):
    >>> from shared.storage.gcs_client import get_gcs_client
    >>> gcs = get_gcs_client()
    >>> gcs.upload_bytes("reports/hello.txt", b"Hello!", "text/plain")

업로드 (로컬 파일):
    >>> gcs.upload_file("reports/data.csv", "/tmp/data.csv", "text/csv")

다운로드 (바이트):
    >>> data = gcs.download_bytes("reports/hello.txt")

다운로드 (로컬 파일):
    >>> gcs.download_file("reports/data.csv", "/tmp/downloaded.csv")

삭제:
    >>> gcs.delete("reports/hello.txt")

존재 확인:
    >>> gcs.exists("reports/hello.txt")
    True

목록 조회:
    >>> gcs.list_blobs("reports/")
    ['reports/hello.txt', 'reports/data.csv']
"""
from __future__ import annotations

from pathlib import Path

from google.cloud import storage

from shared.config import settings


class StorageError(RuntimeError):
    """GCS 관련 모든 오류의 단일 타입."""


class GCSClient:
    """Google Cloud Storage 버킷 관리 및 범용 파일 유틸리티.

    ADC(Application Default Credentials)로 인증하며,
    단일 버킷에 대한 파일 업로드·다운로드·삭제·조회 기능을 제공한다.
    """

    def __init__(self, *, bucket_name: str | None = None) -> None:
        self._bucket_name = bucket_name or settings.gcs_bucket
        self._client = storage.Client()
        self._bucket = self._client.bucket(self._bucket_name)

    @property
    def bucket_name(self) -> str:
        """현재 버킷 이름을 반환한다."""
        return self._bucket_name

    def upload_bytes(
        self, blob_name: str, data: bytes, content_type: str = "application/octet-stream"
    ) -> None:
        """바이트 데이터를 GCS에 업로드한다.

        Parameters
        ----------
        blob_name:
            GCS 내 저장 경로. 예: ``"reports/hello.txt"``
        data:
            업로드할 바이트 데이터.
        content_type:
            MIME 타입. 기본값 ``"application/octet-stream"``.

        Raises
        ------
        StorageError
            업로드 중 예외 발생 시.
        """
        try:
            blob = self._bucket.blob(blob_name)
            blob.upload_from_string(data, content_type=content_type)
        except Exception as exc:
            raise StorageError(str(exc)) from exc

    def upload_file(
        self, blob_name: str, file_path: str | Path, content_type: str = "application/octet-stream"
    ) -> None:
        """로컬 파일을 GCS에 업로드한다.

        Parameters
        ----------
        blob_name:
            GCS 내 저장 경로.
        file_path:
            로컬 파일 경로.
        content_type:
            MIME 타입. 기본값 ``"application/octet-stream"``.

        Raises
        ------
        StorageError
            업로드 중 예외 발생 시.
        """
        try:
            blob = self._bucket.blob(blob_name)
            blob.upload_from_filename(str(file_path), content_type=content_type)
        except Exception as exc:
            raise StorageError(str(exc)) from exc

    def download_bytes(self, blob_name: str) -> bytes:
        """GCS 파일을 바이트로 다운로드한다.

        Parameters
        ----------
        blob_name:
            다운로드할 GCS 파일 경로.

        Returns
        -------
        bytes
            파일 내용.

        Raises
        ------
        StorageError
            다운로드 중 예외 발생 시.
        """
        try:
            blob = self._bucket.blob(blob_name)
            return blob.download_as_bytes()
        except Exception as exc:
            raise StorageError(str(exc)) from exc

    def download_file(self, blob_name: str, dest_path: str | Path) -> None:
        """GCS 파일을 로컬에 저장한다.

        Parameters
        ----------
        blob_name:
            다운로드할 GCS 파일 경로.
        dest_path:
            저장할 로컬 경로.

        Raises
        ------
        StorageError
            다운로드 중 예외 발생 시.
        """
        try:
            blob = self._bucket.blob(blob_name)
            blob.download_to_filename(str(dest_path))
        except Exception as exc:
            raise StorageError(str(exc)) from exc

    def delete(self, blob_name: str) -> None:
        """GCS 파일을 삭제한다.

        Parameters
        ----------
        blob_name:
            삭제할 GCS 파일 경로.

        Raises
        ------
        StorageError
            삭제 중 예외 발생 시.
        """
        try:
            blob = self._bucket.blob(blob_name)
            blob.delete()
        except Exception as exc:
            raise StorageError(str(exc)) from exc

    def exists(self, blob_name: str) -> bool:
        """GCS 파일 존재 여부를 확인한다.

        Parameters
        ----------
        blob_name:
            확인할 GCS 파일 경로.

        Returns
        -------
        bool
            파일이 존재하면 True, 없으면 False.

        Raises
        ------
        StorageError
            확인 중 예외 발생 시.
        """
        try:
            blob = self._bucket.blob(blob_name)
            return blob.exists()
        except Exception as exc:
            raise StorageError(str(exc)) from exc

    def list_blobs(self, prefix: str = "") -> list[str]:
        """특정 경로 아래 파일 목록을 반환한다.

        Parameters
        ----------
        prefix:
            조회할 경로 접두사. 예: ``"reports/"``.
            빈 문자열이면 버킷 전체를 조회한다.

        Returns
        -------
        list[str]
            blob 이름 리스트.

        Raises
        ------
        StorageError
            조회 중 예외 발생 시.
        """
        try:
            blobs = self._client.list_blobs(self._bucket_name, prefix=prefix or None)
            return [blob.name for blob in blobs]
        except Exception as exc:
            raise StorageError(str(exc)) from exc


_gcs_client: GCSClient | None = None


def get_gcs_client() -> GCSClient:
    """GCSClient 싱글턴을 반환한다. 최초 호출 시 1회만 생성."""
    global _gcs_client  # noqa: PLW0603
    if _gcs_client is None:
        _gcs_client = GCSClient()
    return _gcs_client
