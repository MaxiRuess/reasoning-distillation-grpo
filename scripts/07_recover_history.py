"""Read-only Modal inventory/archive. Never starts a Modal Function or changes the volume."""

import argparse
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

from src.artifacts import atomic_json, file_hash, fingerprint


def recover(volume, destination, download_weights=False):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)

    def ls(remote):
        return json.loads(
            subprocess.check_output(
                [sys.executable, "-m", "modal", "volume", "ls", volume, remote, "--json"], text=True
            )
        )

    def get(remote):
        local = destination / remote.lstrip("/")
        local.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run([sys.executable, "-m", "modal", "volume", "get", volume, remote, str(local)], check=True)
        return {"remote": remote, "local": str(local), "sha256": file_hash(local), "bytes": local.stat().st_size}

    inventory, files = {}, []
    selections = [f"outputs/{name}" for name in ["baseline", "sft_traces", "grpo_only", "sft_then_grpo", "re_distill"]]
    selections.append("outputs/sft_then_grpo/checkpoint-117")
    selections.append("eval_checkpoints")
    for remote in selections:
        entries = ls(remote)
        inventory[remote] = entries
        for entry in entries:
            name = Path(entry["filename"]).name
            if entry["type"] != "file":
                continue
            if name.endswith((".json", ".jinja", ".md", ".txt")) or (
                download_weights and name.endswith(".safetensors")
            ):
                files.append(get(entry["filename"]))
        checkpoints = [
            entry["filename"]
            for entry in entries
            if entry["type"] == "dir" and Path(entry["filename"]).name.startswith("checkpoint-")
        ]
        if checkpoints:
            latest = max(checkpoints, key=lambda path: int(path.rsplit("-", 1)[1]))
            child = ls(latest)
            inventory[latest] = child
            for entry in child:
                if Path(entry["filename"]).name == "trainer_state.json":
                    files.append(get(entry["filename"]))
        atomic_json(
            destination / "archive.json", {"volume": volume, "inventory": inventory, "files": files, "complete": False}
        )
    result = {
        "volume": volume,
        "inventory": inventory,
        "files": files,
        "complete": True,
        "weights_downloaded": download_weights,
        "archive_id": fingerprint(inventory),
    }
    atomic_json(destination / "archive.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--volume", default="reasoning-distillation-vol")
    parser.add_argument("--destination", default="outputs/historical/" + datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ"))
    parser.add_argument("--download-weights", action="store_true")
    args = parser.parse_args()
    result = recover(args.volume, args.destination, args.download_weights)
    atomic_json(
        Path("results/recovery/archive-index.json"),
        {
            "archive_id": result["archive_id"],
            "local_path": args.destination,
            "complete": True,
            "weights_downloaded": args.download_weights,
            "files": [{k: v for k, v in row.items() if k != "local"} for row in result["files"]],
        },
    )
    print(f"Archived {len(result['files'])} files without modifying the remote volume.")


if __name__ == "__main__":
    main()
