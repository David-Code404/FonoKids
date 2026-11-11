import os

try:
    import mysql.connector
    from mysql.connector import pooling
except ImportError:
    mysql = None

DB_HOST = os.environ.get("FONOKIDS_DB_HOST", "127.0.0.1")
DB_PORT = int(os.environ.get("FONOKIDS_DB_PORT", "3306"))
DB_USER = os.environ.get("FONOKIDS_DB_USER", "root")
DB_PASSWORD = os.environ.get("FONOKIDS_DB_PASSWORD", "")
DB_NAME = os.environ.get("FONOKIDS_DB_NAME", "fonokids")

_pool = None


def _get_pool():
    global _pool
    if _pool is not None:
        return _pool
    if mysql is None:
        return None
    try:
        _pool = pooling.MySQLConnectionPool(
            pool_name="fonokids_pool",
            pool_size=5,
            host=DB_HOST, port=DB_PORT, user=DB_USER, password=DB_PASSWORD, database=DB_NAME,
        )
    except Exception as e:
        print(f"[AVISO] No se pudo conectar a MySQL ({e}) -- Logros/Para practicar solo van a "
              "usar los archivos, sin guardar en base de datos.")
        _pool = None
    return _pool


def init_db():
    pool = _get_pool()
    if pool is None:
        return
    try:
        conn = pool.get_connection()
        cur = conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS practice_attempts (
                id INT AUTO_INCREMENT PRIMARY KEY,
                class_name VARCHAR(150) NOT NULL,
                palabra VARCHAR(100) NOT NULL,
                correcta TINYINT(1) NOT NULL,
                tipo_error VARCHAR(50) NULL,
                persona VARCHAR(100) NOT NULL DEFAULT 'web',
                prob FLOAT NOT NULL,
                created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
                INDEX idx_palabra (palabra),
                INDEX idx_created_at (created_at)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS ejercicio_intentos (
                id INT AUTO_INCREMENT PRIMARY KEY,
                ejercicio_id VARCHAR(40) NOT NULL,
                tipo VARCHAR(20) NOT NULL,
                correcto TINYINT(1) NOT NULL,
                fuerza TINYINT NULL,
                aguante TINYINT NULL,
                parejo TINYINT NULL,
                voz TINYINT NULL,
                porcentaje_ok FLOAT NULL,
                tipo_error VARCHAR(50) NULL,
                persona VARCHAR(100) NOT NULL DEFAULT 'web',
                created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
                INDEX idx_ejercicio (ejercicio_id),
                INDEX idx_created_at (created_at)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
            """
        )
        conn.commit()
        cur.close()
        conn.close()
        print("[db] Conectado a MySQL -- Logros/Para practicar quedan guardados en la base.")
    except Exception as e:
        print(f"[AVISO] No se pudo inicializar la tabla en MySQL ({e}).")


def save_attempt(class_name, palabra, correcta, tipo_error, prob, persona="web"):
    pool = _get_pool()
    if pool is None:
        return
    try:
        conn = pool.get_connection()
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO practice_attempts (class_name, palabra, correcta, tipo_error, persona, prob) "
            "VALUES (%s, %s, %s, %s, %s, %s)",
            (class_name, palabra, 1 if correcta else 0, tipo_error, persona, float(prob)),
        )
        conn.commit()
        cur.close()
        conn.close()
    except Exception as e:
        print(f"[AVISO] No se pudo guardar el intento en MySQL ({e}).")


def save_attempt_at(class_name, palabra, correcta, tipo_error, prob, created_at_ts, persona="web"):
    import datetime

    pool = _get_pool()
    if pool is None:
        return
    try:
        conn = pool.get_connection()
        cur = conn.cursor()
        created_at = datetime.datetime.fromtimestamp(created_at_ts)
        cur.execute(
            "INSERT INTO practice_attempts (class_name, palabra, correcta, tipo_error, persona, prob, created_at) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s)",
            (class_name, palabra, 1 if correcta else 0, tipo_error, persona, float(prob), created_at),
        )
        conn.commit()
        cur.close()
        conn.close()
    except Exception as e:
        print(f"[AVISO] No se pudo guardar el intento migrado en MySQL ({e}).")


def get_recordings_grouped():
    pool = _get_pool()
    if pool is None:
        return None
    try:
        conn = pool.get_connection()
        cur = conn.cursor(dictionary=True)
        cur.execute(
            """
            SELECT
                DATE(created_at) AS date,
                DATE_FORMAT(MAX(created_at), '%H:%i') AS time,
                class_name,
                palabra AS word,
                correcta,
                tipo_error,
                persona AS person,
                COUNT(*) AS count
            FROM practice_attempts
            GROUP BY DATE(created_at), class_name, persona
            ORDER BY MAX(created_at) DESC
            """
        )
        rows = cur.fetchall()
        cur.close()
        conn.close()
        for r in rows:
            r["date"] = str(r["date"])
            r["correcta"] = bool(r["correcta"])
            r["thumbnail_file"] = None
        return rows
    except Exception as e:
        print(f"[AVISO] No se pudo leer de MySQL ({e}).")
        return None


def save_ejercicio(ejercicio_id, tipo, correcto, fuerza=None, aguante=None, parejo=None, voz=None,
                   porcentaje_ok=None, tipo_error=None, persona="web"):
    pool = _get_pool()
    if pool is None:
        return False
    try:
        conn = pool.get_connection()
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO ejercicio_intentos "
            "(ejercicio_id, tipo, correcto, fuerza, aguante, parejo, voz, porcentaje_ok, tipo_error, persona) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
            (ejercicio_id, tipo, 1 if correcto else 0, fuerza, aguante, parejo, voz,
             None if porcentaje_ok is None else float(porcentaje_ok), tipo_error, persona),
        )
        conn.commit()
        cur.close()
        conn.close()
        return True
    except Exception as e:
        print(f"[AVISO] No se pudo guardar el ejercicio en MySQL ({e}).")
        return False


def get_ejercicios_resumen():
    pool = _get_pool()
    if pool is None:
        return None
    try:
        conn = pool.get_connection()
        cur = conn.cursor(dictionary=True)
        cur.execute(
            """
            SELECT
                ejercicio_id,
                tipo,
                COUNT(*) AS intentos,
                SUM(correcto) AS bien,
                ROUND(AVG(fuerza), 1) AS fuerza_media,
                ROUND(AVG(aguante), 1) AS aguante_medio,
                ROUND(AVG(parejo), 1) AS parejo_medio,
                ROUND(AVG(voz), 1) AS voz_media,
                DATE_FORMAT(MAX(created_at), '%Y-%m-%d %H:%i') AS ultimo
            FROM ejercicio_intentos
            GROUP BY ejercicio_id, tipo
            ORDER BY MAX(created_at) DESC
            """
        )
        rows = cur.fetchall()
        cur.close()
        conn.close()
        for r in rows:
            r["bien"] = int(r["bien"] or 0)
            for k in ("fuerza_media", "aguante_medio", "parejo_medio", "voz_media"):
                r[k] = None if r[k] is None else float(r[k])
        return rows
    except Exception as e:
        print(f"[AVISO] No se pudo leer los ejercicios de MySQL ({e}).")
        return None
