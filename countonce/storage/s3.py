"""``S3BlobStore``: any S3-compatible bucket. Cloudflare R2 works by setting ``endpoint_url``
(``https://<account>.r2.cloudflarestorage.com``). Needs the ``storage`` extra (boto3).

Credentials come from the caller (the factory reads them from the environment); nothing is
hardcoded. With none given, boto3's default chain applies.
"""

from __future__ import annotations

from typing import Any

from countonce.storage.base import BlobInfo, BlobNotFound, check_key


class S3BlobStore:
    def __init__(
        self,
        bucket: str,
        *,
        endpoint_url: str | None = None,
        region: str | None = None,
        access_key_id: str | None = None,
        secret_access_key: str | None = None,
        prefix: str = "",
        client: Any | None = None,
    ) -> None:
        self.bucket = bucket
        self.prefix = prefix.strip("/") + "/" if prefix.strip("/") else ""
        if client is None:
            try:
                import boto3
                from botocore.config import Config
            except ImportError as exc:
                raise ImportError(
                    "the s3 storage backend needs boto3: pip install 'countonce[storage]'"
                ) from exc
            client = boto3.client(
                "s3",
                endpoint_url=endpoint_url,
                region_name=region or "auto",
                aws_access_key_id=access_key_id,
                aws_secret_access_key=secret_access_key,
                config=Config(signature_version="s3v4"),
            )
        self._c = client

    def _k(self, key: str) -> str:
        return self.prefix + check_key(key)

    def put(self, key: str, data: bytes) -> None:
        self._c.put_object(Bucket=self.bucket, Key=self._k(key), Body=data)

    def get(self, key: str) -> bytes:
        try:
            body: bytes = self._c.get_object(Bucket=self.bucket, Key=self._k(key))["Body"].read()
            return body
        except self._c.exceptions.NoSuchKey:
            raise BlobNotFound(key) from None

    def exists(self, key: str) -> bool:
        try:
            self._c.head_object(Bucket=self.bucket, Key=self._k(key))
            return True
        except self._c.exceptions.ClientError as exc:
            if exc.response.get("Error", {}).get("Code") in ("404", "NoSuchKey", "NotFound"):
                return False
            raise

    def delete(self, key: str) -> bool:
        existed = self.exists(key)
        self._c.delete_object(Bucket=self.bucket, Key=self._k(key))
        return existed

    def presign_get(self, key: str, ttl_s: float = 900.0) -> str:
        return self._presign("get_object", key, ttl_s)

    def presign_put(self, key: str, ttl_s: float = 900.0) -> str:
        return self._presign("put_object", key, ttl_s)

    def _presign(self, op: str, key: str, ttl_s: float) -> str:
        url: str = self._c.generate_presigned_url(
            op,
            Params={"Bucket": self.bucket, "Key": self._k(key)},
            ExpiresIn=max(1, int(ttl_s)),
        )
        return url

    def list_prefix(self, prefix: str = "") -> list[BlobInfo]:
        out: list[BlobInfo] = []
        pager = self._c.get_paginator("list_objects_v2")
        for page in pager.paginate(Bucket=self.bucket, Prefix=self.prefix + prefix):
            for o in page.get("Contents", []):
                out.append(
                    BlobInfo(
                        o["Key"][len(self.prefix) :], int(o["Size"]), o["LastModified"].timestamp()
                    )
                )
        return sorted(out, key=lambda b: b.key)
