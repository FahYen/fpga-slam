# Registration Kernel API

This service exposes API endpoints for sending and receiving registration-kernel payloads.

## Endpoints

- `GET /healthz`
  - Liveness probe.

- `GET /v1/registration/limits`
  - Returns kernel limits, including `max_correspondences`.

- `POST /v1/registration/accumulate`
  - Input: flattened source/target XYZ vectors, labels, kernel, backend.
  - Output: `jtj_out`, `jtr_out`, `used_count`, `dropped_count`.

## Local run

```bash
cd fpga/api
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn registration_api:app --host 0.0.0.0 --port 8080
```

## Example request

```bash
curl -sS http://127.0.0.1:8080/v1/registration/accumulate \
  -H 'content-type: application/json' \
  -d '{
    "backend": "cpu-proto",
    "kernel": 0.333333,
    "correspondence_count": 2,
    "src_xyz": [1.0,2.0,3.0, 2.0,3.0,4.0],
    "tgt_xyz": [1.1,2.0,3.0, 1.9,3.1,4.0],
    "labels": [18, 9]
  }'
```

## AWS deployment notes

You can run this API on an AWS Linux instance (EC2) and route requests from your host process.

- Recommended for development:
  - `c5.4xlarge` for host/API iteration and software emulation.
- Recommended for FPGA hardware tests:
  - `f1.2xlarge` (VU9P) with FPGA Developer AMI.

Keep the API on a private network interface when possible. If public access is required, lock down the security group to trusted source IPs and use TLS via a reverse proxy.
