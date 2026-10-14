"""
PropIQ - Portable PostgreSQL 16 Setup & Server Launcher for Windows
Downloads standalone PostgreSQL binaries, initializes data directory,
starts PostgreSQL daemon on port 5432, and provisions 'propiq_db'.
"""

import os
import sys
import zipfile
import subprocess
import time
import requests

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PG_DIR = os.path.join(BASE_DIR, "pgsql")
PG_BIN = os.path.join(PG_DIR, "bin")
PG_DATA = os.path.join(PG_DIR, "data")
PG_ZIP = os.path.join(BASE_DIR, "scripts", "postgres_binaries.zip")
ZIP_URL = "https://get.enterprisedb.com/postgresql/postgresql-16.8-1-windows-x64-binaries.zip"

def download_file(url: str, dest_path: str):
    print(f"[1/4] Downloading PostgreSQL standalone binaries from {url}...")
    response = requests.get(url, stream=True)
    response.raise_for_status()
    total_size = int(response.headers.get("content-length", 0))
    downloaded = 0
    with open(dest_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            if chunk:
                f.write(chunk)
                downloaded += len(chunk)
                if total_size > 0:
                    pct = (downloaded / total_size) * 100
                    print(f"\r  Downloaded: {downloaded / (1024*1024):.1f} MB / {total_size / (1024*1024):.1f} MB ({pct:.1f}%)", end="", flush=True)
    print("\n[OK] Download complete.")

def extract_zip(zip_path: str, extract_to: str):
    print(f"[2/4] Extracting binaries to {extract_to}...")
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(extract_to)
    print("[OK] Extraction complete.")

def init_and_start_postgres():
    print("[3/4] Initializing PostgreSQL cluster and starting server...")
    os.makedirs(PG_DATA, exist_ok=True)

    initdb_exe = os.path.join(PG_BIN, "initdb.exe")
    pg_ctl_exe = os.path.join(PG_BIN, "pg_ctl.exe")
    pw_file = os.path.join(BASE_DIR, "scripts", "pg_password.txt")

    with open(pw_file, "w", encoding="utf-8") as f:
        f.write("postgrespassword\n")

    # Only run initdb if data directory is empty
    if not os.path.exists(os.path.join(PG_DATA, "PG_VERSION")):
        cmd_init = [
            initdb_exe,
            "-U", "postgres",
            "-A", "md5",
            f"--pwfile={pw_file}",
            "-D", PG_DATA,
            "-E", "UTF8"
        ]
        res = subprocess.run(cmd_init, capture_output=True, text=True)
        print("  initdb output:", res.stdout)
        if res.returncode != 0:
            print("  initdb error:", res.stderr)
            raise RuntimeError(f"initdb failed: {res.stderr}")

    # Start server
    log_file = os.path.join(PG_DIR, "postgres.log")
    cmd_start = [
        pg_ctl_exe,
        "-D", PG_DATA,
        "-l", log_file,
        "-o", "-p 5432",
        "start"
    ]
    res_start = subprocess.run(cmd_start, capture_output=True, text=True)
    print("  pg_ctl start output:", res_start.stdout)
    time.sleep(3)

    # Create database propiq_db if not exists
    createdb_exe = os.path.join(PG_BIN, "createdb.exe")
    env = os.environ.copy()
    env["PGPASSWORD"] = "postgrespassword"

    cmd_createdb = [
        createdb_exe,
        "-U", "postgres",
        "-p", "5432",
        "propiq_db"
    ]
    res_db = subprocess.run(cmd_createdb, capture_output=True, text=True, env=env)
    print("  createdb output:", res_db.stdout or res_db.stderr)

    print("[4/4] PostgreSQL running successfully on port 5432!")

def main():
    if not os.path.exists(os.path.join(PG_BIN, "initdb.exe")):
        if not os.path.exists(PG_ZIP):
            download_file(ZIP_URL, PG_ZIP)
        extract_zip(PG_ZIP, BASE_DIR)
        
    init_and_start_postgres()

if __name__ == "__main__":
    main()
