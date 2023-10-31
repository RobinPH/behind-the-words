import atexit
import os
import platform
import shutil
import stat
import subprocess
import tempfile
import time
import zipfile
from pathlib import Path
from threading import Timer
from typing import Dict, List, Optional

import requests
import yaml
from flask import Flask

__all__ = ["run_with_ngrok", "start_ngrok", "get_host", "NgrokAPI"]

_ngrok_address = ""


def _get_command() -> str:
    system = platform.system()
    if system in ["Darwin", "Linux"]:
        command = "ngrok"
    elif system == "Windows":
        command = "ngrok.exe"
    else:
        raise Exception(f"{system} is not supported")
    return command


def get_host() -> str:
    return _ngrok_address


def _check_ngrok_available() -> bool:
    cmd = "where" if platform.system() == "Windows" else "which"
    try:
        res = subprocess.call([cmd, "ngrok"])
        return (
            False if res else True
        )  # subprocess will return 1 if not found otherwise 0
    except:
        print("Try installing ngrok")
        return False


def _run_ngrok(host: str, port: int, config: Optional[Dict[str, str]] = None) -> str:
    command = _get_command()
    ngrok_path = str(Path(tempfile.gettempdir(), "ngrok"))
    if not _check_ngrok_available():
        _download_ngrok(ngrok_path)
        executable = str(Path(ngrok_path, command))
        # Make file executable for the current user.
        os.chmod(executable, stat.S_IEXEC)
    else:
        executable = "ngrok"

    if config:
        ngrok_config_name = "config-custom.yml"
        ngrok_config_path = str(Path(ngrok_path, ngrok_config_name))

        with open(ngrok_config_path, 'w') as yamlfile:
            yaml.safe_dump(config, yamlfile, default_flow_style=False)

        with open(ngrok_config_path, 'r') as yamlfile:
            print(yamlfile.read())

    ngrok = subprocess.Popen(
        [executable, "http", f"--config={ngrok_config_path}" if config else "", f"{host}:{port}"])
    atexit.register(ngrok.terminate)
    localhost_url = f"http://localhost:4040/api/tunnels"  # Url with tunnel details
    for _ in range(5):
        time.sleep(1)
        tunnels_data = requests.get(localhost_url).json()["tunnels"]
        if len(tunnels_data):
            return tunnels_data[0]["public_url"].replace("https", "http")

    raise ValueError("Not found ngrok tunnel public url after start")


def _download_ngrok(ngrok_path: str):
    if Path(ngrok_path).exists():
        return
    system = platform.system()
    if system == "Darwin":
        url = "https://bin.equinox.io/c/bNyj1mQVY4c/ngrok-v3-stable-darwin-amd64.zip"
    elif system == "Windows":
        url = "https://bin.equinox.io/c/bNyj1mQVY4c/ngrok-v3-stable-windows-amd64.zip"
    elif system == "Linux":
        url = "https://bin.equinox.io/c/bNyj1mQVY4c/ngrok-v3-stable-linux-amd64.tgz"
    else:
        raise Exception(f"{system} is not supported")
    download_path = _download_file(url)
    # with zipfile.ZipFile(download_path, "r") as zip_ref:
    #     zip_ref.extractall(ngrok_path)
    shutil.unpack_archive(download_path, ngrok_path)


def _download_file(url: str) -> str:
    local_filename = url.split("/")[-1]
    r = requests.get(url, stream=True)
    download_path = str(Path(tempfile.gettempdir(), local_filename))
    with open(download_path, "wb") as f:
        shutil.copyfileobj(r.raw, f)
    return download_path


def start_ngrok(host: str, port: int, auth_token: Optional[str] = None):
    global _ngrok_address
    _ngrok_address = _run_ngrok(host, port, auth_token)
    print(f" * Running on {_ngrok_address}")
    print(f" * Traffic stats available on http://127.0.0.1:4040")
    requests.get(f"http://{host}:{port}/")


def run_with_ngrok(app: Flask, config: Optional[Dict[str, str]] = None):
    """
    The provided Flask app will be securely exposed to the public internet via ngrok when run,
    and the its ngrok address will be printed to stdout
    :param app: a Flask application object
    :param auth_token: ngrok authtoken if exists
    :return: None
    """
    old_run = app.run

    def new_run(*args, **kwargs):
        host = kwargs.get("host", "127.0.0.1")
        port = kwargs.get("port", 5000)
        thread = Timer(1, start_ngrok, args=(host, port, config))
        thread.setDaemon(True)
        thread.start()
        old_run(*args, **kwargs)

    app.run = new_run


class NgrokAPI():
    def __init__(self, ngrok_api_key: str):
        self.ngrok_api_key = ngrok_api_key

    def get_all_sessions(self):
        return requests.get(
            "https://api.ngrok.com/tunnel_sessions",
            headers={"Authorization": f"Bearer {self.ngrok_api_key}",
                     "Ngrok-Version": "2"},
        ).json()["tunnel_sessions"]

    def get_session(self, session_id: str):
        return requests.get(
            f"https://api.ngrok.com/tunnel_sessions/{session_id}",
            headers={"Authorization": f"Bearer {self.ngrok_api_key}",
                     "Ngrok-Version": "2"},
        ).json()

    def get_endpoints(self):
        return requests.get(
            "https://api.ngrok.com/endpoints",
            headers={"Authorization": f"Bearer {self.ngrok_api_key}",
                     "Ngrok-Version": "2"},
        ).json()["endpoints"]

    def get_tunnel(self, tunnel_id: str):
        return requests.get(
            f"https://api.ngrok.com/tunnels/{tunnel_id}",
            headers={"Authorization": f"Bearer {self.ngrok_api_key}",
                     "Ngrok-Version": "2"},
        ).json()

    def get_all_tunnels(self):
        endpoints = self.get_endpoints()

        tunnels = list(map(lambda tunnel: tunnel["tunnel"], filter(
            lambda endpoint: "tunnel" in endpoint, endpoints)))
        tunnels = [self.get_tunnel(tunnel["id"]) for tunnel in tunnels]
        tunnels = [{**tunnel, "tunnel_session": self.get_session(
            tunnel["tunnel_session"]["id"])} for tunnel in tunnels if "tunnel_session" in tunnel]

        return tunnels


def kill_ngrok_sessions(ngrok_api_key: str, session_ids: List[str]):
    for session_id in session_ids:
        requests.post(
            f"https://api.ngrok.com/tunnel_sessions/{session_id}/stop",
            json={},
            headers={
                "Authorization": f"Bearer {ngrok_api_key}",
                "Content-Type": "application/json",
                "Ngrok-Version": "2",
            },
        )


if __name__ == "__main__":
    print(_check_ngrok_available())
