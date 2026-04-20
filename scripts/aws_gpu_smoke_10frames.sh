#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

REGION="${REGION:-us-east-1}"
INSTANCE_TYPE="${INSTANCE_TYPE:-g5.xlarge}"
INSTANCE_PROFILE_NAME="${INSTANCE_PROFILE_NAME:-EC2-S3-ReadOnly}"
REMOTE_USER="${REMOTE_USER:-ubuntu}"
REMOTE_ROOT="${REMOTE_ROOT:-/home/ubuntu/workspace}"
VOLUME_SIZE_GB="${VOLUME_SIZE_GB:-200}"
KEEP_INSTANCE_ON_FAILURE="${KEEP_INSTANCE_ON_FAILURE:-0}"
PACKAGE_LOCAL_GTSAM="${PACKAGE_LOCAL_GTSAM:-1}"

RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
AWS_RUN_DIR="${AWS_RUN_DIR:-$REPO_ROOT/aws_runs/$RUN_ID}"
KEY_NAME="slam-gpu-smoke-$RUN_ID"
SG_NAME="slam-gpu-smoke-$RUN_ID"
KEY_PATH="$AWS_RUN_DIR/$KEY_NAME.pem"
REMOTE_SMOKE_ROOT="$REMOTE_ROOT/runs/gpu_smoke_10frames"

INSTANCE_ID=""
SECURITY_GROUP_ID=""
PUBLIC_IP=""
AMI_ID=""
SUBNET_ID=""
cleanup() {
  local exit_code=$?

  if [[ -n "$INSTANCE_ID" && "$KEEP_INSTANCE_ON_FAILURE" != "1" ]]; then
    aws ec2 terminate-instances --instance-ids "$INSTANCE_ID" --region "$REGION" >/dev/null 2>&1 || true
    aws ec2 wait instance-terminated --instance-ids "$INSTANCE_ID" --region "$REGION" >/dev/null 2>&1 || true
  fi

  if [[ -n "$SECURITY_GROUP_ID" ]]; then
    aws ec2 delete-security-group --group-id "$SECURITY_GROUP_ID" --region "$REGION" >/dev/null 2>&1 || true
  fi

  aws ec2 delete-key-pair --key-name "$KEY_NAME" --region "$REGION" >/dev/null 2>&1 || true
  rm -f "$KEY_PATH"

  exit "$exit_code"
}
trap cleanup EXIT

mkdir -p "$AWS_RUN_DIR"

aws sts get-caller-identity > "$AWS_RUN_DIR/sts-get-caller-identity.json"

if [[ "$PACKAGE_LOCAL_GTSAM" == "1" && -f /usr/local/include/gtsam/geometry/Pose3.h ]]; then
  "$REPO_ROOT/scripts/package_local_gtsam.sh" "$REPO_ROOT/scripts/gtsam-local-install.tar.gz"
fi

PUBLIC_CIDR="$(curl -fsSL https://checkip.amazonaws.com | tr -d '\n')/32"
echo "$PUBLIC_CIDR" > "$AWS_RUN_DIR/public-cidr.txt"

VPC_ID="$(aws ec2 describe-vpcs \
  --region "$REGION" \
  --filters Name=is-default,Values=true \
  --query 'Vpcs[0].VpcId' \
  --output text)"

AZ="$(aws ec2 describe-instance-type-offerings \
  --region "$REGION" \
  --location-type availability-zone \
  --filters "Name=instance-type,Values=$INSTANCE_TYPE" \
  --query 'InstanceTypeOfferings[0].Location' \
  --output text)"

SUBNET_ID="$(aws ec2 describe-subnets \
  --region "$REGION" \
  --filters Name=default-for-az,Values=true "Name=availability-zone,Values=$AZ" \
  --query 'Subnets[0].SubnetId' \
  --output text)"

AMI_ID="$(aws ec2 describe-images \
  --owners amazon \
  --region "$REGION" \
  --filters "Name=name,Values=Deep Learning Base OSS Nvidia Driver GPU AMI (Ubuntu 22.04)*" \
            "Name=state,Values=available" \
  --query 'reverse(sort_by(Images,&CreationDate))[0].ImageId' \
  --output text)"

SECURITY_GROUP_ID="$(aws ec2 create-security-group \
  --group-name "$SG_NAME" \
  --description "SLAM GPU smoke test" \
  --vpc-id "$VPC_ID" \
  --region "$REGION" \
  --query GroupId \
  --output text)"

aws ec2 authorize-security-group-ingress \
  --group-id "$SECURITY_GROUP_ID" \
  --protocol tcp \
  --port 22 \
  --cidr "$PUBLIC_CIDR" \
  --region "$REGION" > "$AWS_RUN_DIR/security-group-ingress.json"

aws ec2 create-key-pair \
  --key-name "$KEY_NAME" \
  --region "$REGION" \
  --query KeyMaterial \
  --output text > "$KEY_PATH"
chmod 600 "$KEY_PATH"

INSTANCE_ID="$(aws ec2 run-instances \
  --region "$REGION" \
  --image-id "$AMI_ID" \
  --instance-type "$INSTANCE_TYPE" \
  --subnet-id "$SUBNET_ID" \
  --security-group-ids "$SECURITY_GROUP_ID" \
  --key-name "$KEY_NAME" \
  --iam-instance-profile "Name=$INSTANCE_PROFILE_NAME" \
  --block-device-mappings "[{\"DeviceName\":\"/dev/sda1\",\"Ebs\":{\"VolumeSize\":$VOLUME_SIZE_GB,\"VolumeType\":\"gp3\",\"DeleteOnTermination\":true}}]" \
  --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=$KEY_NAME},{Key=Project,Value=slam},{Key=Purpose,Value=gpu-smoke-10frames}]" \
                      "ResourceType=volume,Tags=[{Key=Name,Value=$KEY_NAME-root},{Key=Project,Value=slam},{Key=Purpose,Value=gpu-smoke-10frames}]" \
  --query 'Instances[0].InstanceId' \
  --output text)"

cat > "$AWS_RUN_DIR/launch-summary.txt" <<EOF
region=$REGION
instance_type=$INSTANCE_TYPE
instance_id=$INSTANCE_ID
ami_id=$AMI_ID
subnet_id=$SUBNET_ID
availability_zone=$AZ
security_group_id=$SECURITY_GROUP_ID
instance_profile_name=$INSTANCE_PROFILE_NAME
EOF

aws ec2 wait instance-running --instance-ids "$INSTANCE_ID" --region "$REGION"
aws ec2 wait instance-status-ok --instance-ids "$INSTANCE_ID" --region "$REGION"

PUBLIC_IP="$(aws ec2 describe-instances \
  --instance-ids "$INSTANCE_ID" \
  --region "$REGION" \
  --query 'Reservations[0].Instances[0].PublicIpAddress' \
  --output text)"
echo "$PUBLIC_IP" > "$AWS_RUN_DIR/public-ip.txt"

ssh_base=(ssh -o StrictHostKeyChecking=accept-new -i "$KEY_PATH" "$REMOTE_USER@$PUBLIC_IP")
scp_base=(scp -i "$KEY_PATH")
rsync_ssh="ssh -i $KEY_PATH -o StrictHostKeyChecking=accept-new"

"${ssh_base[@]}" "mkdir -p '$REMOTE_ROOT/slam'"

rsync -az --delete -e "$rsync_ssh" \
  --exclude '.git/' \
  --exclude 'data/' \
  --exclude 'aws_runs/' \
  --exclude 'sgslam_runs/' \
  --exclude '**/build/' \
  --exclude 'third_party/gtsam/build*/' \
  "$REPO_ROOT/" "$REMOTE_USER@$PUBLIC_IP:$REMOTE_ROOT/slam/"

"${ssh_base[@]}" "mkdir -p '$REMOTE_SMOKE_ROOT'"

"${ssh_base[@]}" "cd '$REMOTE_ROOT/slam' && ./scripts/gpu_setup_env.sh && ./scripts/gpu_sync_test_data.sh && RUN_ROOT='$REMOTE_SMOKE_ROOT' RUN_ID='$RUN_ID' ./scripts/gpu_smoke_10frames.sh" \
  | tee "$AWS_RUN_DIR/remote-smoke.log"

mkdir -p "$AWS_RUN_DIR/remote-results"
"${scp_base[@]}" -r "$REMOTE_USER@$PUBLIC_IP:$REMOTE_SMOKE_ROOT/$RUN_ID/." "$AWS_RUN_DIR/remote-results/"

echo "Smoke run artifacts copied to $AWS_RUN_DIR/remote-results"
