"""
db.py
-----
Persistencia en MySQL (XAMPP, base "speakshadow", tabla "practice_attempts")
de cada intento de práctica con confianza suficiente -- pedido explícito:
que Logros y Para practicar no dependan solo de los archivos en
data/sesiones_continuas/ (que se pueden perder o moverse, como ya pasó una
vez en esta PC), sino que queden guardados en una base de datos aparte.

Nunca tira si la base no está disponible (XAMPP apagado, etc.) -- la app
sigue funcionando igual, solo sin persistencia extra, mismo criterio que el
resto del proyecto (todo lo "extra" se degrada con gracia).
"""
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
DB_NAME = os.environ.get("FONOKIDS_DB_NAME", "speakshadow")

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
    """Crea la tabla si todavía no existe -- se llama una vez al arrancar el
    servidor. No rompe el arranque si MySQL no está disponible."""
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
        conn.commit()
        cur.close()
        conn.close()
        print("[db] Conectado a MySQL -- Logros/Para practicar quedan guardados en la base.")
    except Exception as e:
        print(f"[AVISO] No se pudo inicializar la tabla en MySQL ({e}).")


def save_attempt(class_name, palabra, correcta, tipo_error, prob, persona="web"):
    """Guarda un intento con confianza suficiente. Nunca tira -- si falla,
    solo avisa por consola y la predicción sigue devolviéndose normal."""
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
    """Igual que save_attempt(), pero con una fecha/hora explícita (timestamp
    Unix) en vez de CURRENT_TIMESTAMP -- solo para la migración única de
    datos viejos (ver migrate_recordings_to_db.py), que reconstruye el
    historial usando la fecha real de cada archivo."""
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
    """Devuelve la práctica agrupada por (fecha, clase, persona), mismo
    formato que devolvía /dataset/recordings leyendo el filesystem -- para
    que el frontend (Logros, Para practicar) no necesite cambiar nada."""
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
