"""Persistence for immutable Egyptian ID OCR submissions.

This module deliberately contains no Streamlit or OCR code.  It accepts the
OCR result exactly as returned by ``process_image`` and writes it to MySQL in
one parameterized INSERT statement.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mysql.connector
from dotenv import load_dotenv
from mysql.connector import Error


# Local development credentials are loaded from Backend/.env if it exists.
# Existing system environment variables always take precedence.
load_dotenv(Path(__file__).resolve().parents[1] / ".env")


class DatabaseConfigurationError(RuntimeError):
    """Raised when the required MySQL connection settings are missing."""


class DatabaseStorageError(RuntimeError):
    """Raised when MySQL cannot persist an OCR submission."""


@dataclass(frozen=True)
class MySQLSettings:
    host: str
    port: int
    user: str
    password: str
    database: str


def load_mysql_settings() -> MySQLSettings:
    """Load connection settings from environment variables or ``Backend/.env``.

    MYSQL_USER and MYSQL_PASSWORD are intentionally never hard-coded in the
    repository.  MYSQL_HOST, MYSQL_PORT, and MYSQL_DATABASE have safe local
    development defaults.
    """
    user = os.getenv("MYSQL_USER")
    password = os.getenv("MYSQL_PASSWORD")
    if not user or password is None:
        raise DatabaseConfigurationError(
            "Set MYSQL_USER and MYSQL_PASSWORD before saving OCR results."
        )

    try:
        port = int(os.getenv("MYSQL_PORT", "3306"))
    except ValueError as exc:
        raise DatabaseConfigurationError("MYSQL_PORT must be a number.") from exc

    return MySQLSettings(
        host=os.getenv("MYSQL_HOST", "127.0.0.1"),
        port=port,
        user=user,
        password=password,
        database=os.getenv("MYSQL_DATABASE", "egyptian_id_ocr"),
    )


def save_ocr_result(
    *,
    result: dict[str, Any],
    image_bytes: bytes,
    original_filename: str | None,
    image_mime_type: str | None,
    model_version: str = "v1",
) -> int:
    """Insert one immutable OCR submission and return its database ID."""
    settings = load_mysql_settings()
    image_hash = hashlib.sha256(image_bytes).hexdigest()

    statement = """
        INSERT INTO id_ocr_results (
            original_filename, image_mime_type, image_sha256, card_image,
            national_id, full_name, first_name, last_name, birth_date,
            gender, governorate, address, serial_number, ocr_result,
            capture_quality, ocr_model_version
        ) VALUES (
            %(original_filename)s, %(image_mime_type)s, %(image_sha256)s, %(card_image)s,
            %(national_id)s, %(full_name)s, %(first_name)s, %(last_name)s, %(birth_date)s,
            %(gender)s, %(governorate)s, %(address)s, %(serial_number)s, %(ocr_result)s,
            %(capture_quality)s, %(ocr_model_version)s
        )
    """
    values = {
        "original_filename": original_filename,
        "image_mime_type": image_mime_type or "application/octet-stream",
        "image_sha256": image_hash,
        "card_image": image_bytes,
        "national_id": result.get("national_id"),
        "full_name": result.get("full_name"),
        "first_name": result.get("first_name"),
        "last_name": result.get("last_name"),
        "birth_date": result.get("birth_date"),
        "gender": result.get("gender"),
        "governorate": result.get("governorate"),
        "address": result.get("address"),
        "serial_number": result.get("serial_number"),
        "ocr_result": json.dumps(result, ensure_ascii=False),
        "capture_quality": result.get("capture_quality"),
        "ocr_model_version": model_version,
    }

    connection = None
    cursor = None
    try:
        connection = mysql.connector.connect(
            host=settings.host,
            port=settings.port,
            user=settings.user,
            password=settings.password,
            database=settings.database,
            connection_timeout=10,
        )
        cursor = connection.cursor()
        cursor.execute(statement, values)
        connection.commit()
        return int(cursor.lastrowid)
    except Error as exc:
        if connection is not None:
            connection.rollback()
        raise DatabaseStorageError(f"Could not save OCR result: {exc}") from exc
    finally:
        if cursor is not None:
            cursor.close()
        if connection is not None and connection.is_connected():
            connection.close()
