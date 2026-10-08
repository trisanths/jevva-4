#!/bin/bash
# Launch, stop, start or reach the GPU box for these experiments.
# Usage: infra/gpu.sh up [type, default g6e.2xlarge] | stop | start | down | status | ssh-config | allow-ip | sync
#
# One box per name, tagged Name=$MOEFY_BOX_NAME (default moefy-gpu) and
# reached through the ssh alias $MOEFY_SSH_ALIAS (default moefy). Setting
# both lets more than one box run side by side. "stop" keeps the root EBS volume
# and loses the NVMe instance store, which is why data lives in S3 and the
# first-boot script restores it. "down" terminates.
set -euo pipefail
# Required: MOEFY_BUCKET (S3 bucket for code, results and adapters), MOEFY_KEY_NAME (EC2 key pair).
# Optional: MOEFY_REGION (default us-east-1), MOEFY_KEY_FILE, MOEFY_ROLE, MOEFY_HF_SECRET (Secrets Manager
# id holding a Hugging Face token, default huggingface), MOEFY_BOX_NAME, MOEFY_SSH_ALIAS, MOEFY_SPOT.
REGION=${MOEFY_REGION:-us-east-1}

# Credentials come from the usual AWS CLI chain (AWS_PROFILE, env keys, SSO).
aws sts get-caller-identity >/dev/null 2>&1 || { echo "no working AWS credentials" >&2; exit 1; }
NAME=${MOEFY_BOX_NAME:-moefy-gpu}
ALIAS=${MOEFY_SSH_ALIAS:-moefy}
KEY_NAME=${MOEFY_KEY_NAME:?set MOEFY_KEY_NAME to an EC2 key pair name}
KEY_FILE=${MOEFY_KEY_FILE:-$HOME/.ssh/$KEY_NAME.pem}
ROLE=${MOEFY_ROLE:-moefy-worker}
BUCKET=${MOEFY_BUCKET:?set MOEFY_BUCKET to an S3 bucket you own}
HF_SECRET=${MOEFY_HF_SECRET:-huggingface}
HERE=$(cd "$(dirname "$0")" && pwd)

cmd=${1:-up}

find_instance() {
  aws ec2 describe-instances --region $REGION \
    --filters "Name=tag:Name,Values=$NAME" "Name=instance-state-name,Values=pending,running,stopping,stopped" \
    --query 'Reservations[0].Instances[0].InstanceId' --output text
}

no_instance() { [ -z "${1:-}" ] || [ "$1" = "None" ]; }

my_ip() {
  local ip
  ip=$(curl -sf https://checkip.amazonaws.com | tr -d '[:space:]') || ip=""
  printf '%s' "$ip" | grep -Eq '^[0-9.]+$' || { echo "could not read public IP" >&2; return 1; }
  printf '%s' "$ip"
}

ensure_bucket() {
  aws s3api head-bucket --bucket $BUCKET 2>/dev/null || aws s3 mb "s3://$BUCKET" --region $REGION
}

ensure_role() {
  if ! aws iam get-role --role-name $ROLE >/dev/null 2>&1; then
    aws iam create-role --role-name $ROLE --assume-role-policy-document '{
      "Version":"2012-10-17","Statement":[{"Effect":"Allow",
      "Principal":{"Service":"ec2.amazonaws.com"},"Action":"sts:AssumeRole"}]}' >/dev/null
    aws iam create-instance-profile --instance-profile-name $ROLE >/dev/null
    aws iam add-role-to-instance-profile --instance-profile-name $ROLE --role-name $ROLE
    sleep 10
  fi
  ACCOUNT=$(aws sts get-caller-identity --query Account --output text)
  sed -e "s/__BUCKET__/$BUCKET/g" -e "s/__ACCOUNT__/$ACCOUNT/g" -e "s/__REGION__/$REGION/g" -e "s#__HF_SECRET__#$HF_SECRET#g" \
    "$HERE/role-policy.json" > /tmp/moefy-role-policy.json
  aws iam put-role-policy --role-name $ROLE --policy-name $ROLE-inline \
    --policy-document file:///tmp/moefy-role-policy.json >/dev/null
}

ensure_sg() {
  SG=$(aws ec2 describe-security-groups --region $REGION \
    --filters "Name=group-name,Values=$NAME" --query 'SecurityGroups[0].GroupId' --output text)
  if [ "$SG" = "None" ]; then
    MY_IP=$(my_ip)
    SG=$(aws ec2 create-security-group --region $REGION --group-name $NAME \
      --description "jevva-4 GPU box" --query GroupId --output text)
    aws ec2 authorize-security-group-ingress --region $REGION --group-id $SG \
      --protocol tcp --port 22 --cidr "$MY_IP/32" >/dev/null
  fi
  echo "$SG"
}

allow_ip() {
  SG=$(aws ec2 describe-security-groups --region $REGION \
    --filters "Name=group-name,Values=$NAME" --query 'SecurityGroups[0].GroupId' --output text)
  [ "$SG" != "None" ] || { echo "no security group $NAME" >&2; exit 1; }
  NEW_CIDR="$(my_ip)/32"
  EXISTING=$(aws ec2 describe-security-groups --region $REGION --group-ids "$SG" \
    --query 'SecurityGroups[0].IpPermissions[?ToPort==`22`].IpRanges[].CidrIp' --output text)
  case " $EXISTING " in
    *" $NEW_CIDR "*) echo "port 22 already open to $NEW_CIDR" ;;
    *) aws ec2 authorize-security-group-ingress --region $REGION --group-id "$SG" \
         --protocol tcp --port 22 --cidr "$NEW_CIDR" >/dev/null; echo "authorized $NEW_CIDR" ;;
  esac
  for cidr in $EXISTING; do
    [ "$cidr" = "$NEW_CIDR" ] && continue
    aws ec2 revoke-security-group-ingress --region $REGION --group-id "$SG" \
      --protocol tcp --port 22 --cidr "$cidr" >/dev/null
    echo "revoked $cidr"
  done
}

instance_state() {
  aws ec2 describe-instances --region $REGION --instance-ids "$1" \
    --query 'Reservations[0].Instances[0].State.Name' --output text
}

public_ip() {
  aws ec2 describe-instances --region $REGION --instance-ids "$1" \
    --query 'Reservations[0].Instances[0].PublicIpAddress' --output text
}

case $cmd in
  up)
    TYPE=${2:-g6e.2xlarge}
    ID=$(find_instance)
    if ! no_instance "$ID"; then echo "instance $ID already exists; use start/stop/down" >&2; exit 1; fi
    ensure_bucket; ensure_role; SG=$(ensure_sg)
    AMI=$(aws ssm get-parameter --region $REGION \
      --name /aws/service/deeplearning/ami/x86_64/base-oss-nvidia-driver-gpu-ubuntu-24.04/latest/ami-id \
      --query Parameter.Value --output text)
    sed -e "s/__BUCKET__/$BUCKET/g" -e "s/__REGION__/$REGION/g" -e "s#__HF_SECRET__#$HF_SECRET#g" \
      "$HERE/gpu-userdata.sh" > /tmp/moefy-userdata.sh
    # GPU capacity is per zone; try each zone until one accepts the launch.
    ID=""
    for AZ in ${REGION}a ${REGION}b ${REGION}c ${REGION}d ${REGION}e ${REGION}f; do
      ID=$(aws ec2 run-instances --region $REGION --image-id "$AMI" --instance-type "$TYPE" \
        --placement "AvailabilityZone=$AZ" \
        --key-name "$KEY_NAME" --security-group-ids "$SG" \
        --iam-instance-profile Name=$ROLE \
        --block-device-mappings '[{"DeviceName":"/dev/sda1","Ebs":{"VolumeSize":300,"VolumeType":"gp3"}}]' \
        --user-data file:///tmp/moefy-userdata.sh \
        ${MOEFY_SPOT:+--instance-market-options '{"MarketType":"spot","SpotOptions":{"SpotInstanceType":"one-time","InstanceInterruptionBehavior":"terminate"}}'} \
        --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=$NAME},{Key=project,Value=moefy}]" \
        --query 'Instances[0].InstanceId' --output text 2>/tmp/gpu-launch-err) && break
      echo "no capacity in $AZ: $(tr -d '\n' < /tmp/gpu-launch-err | cut -c1-120)" >&2
      ID=""
    done
    [ -n "$ID" ] || { echo "no zone accepted a $TYPE launch" >&2; exit 1; }
    aws ec2 wait instance-running --region $REGION --instance-ids "$ID"
    echo "instance $ID ($TYPE, ami $AMI) at $(public_ip "$ID")"
    ;;
  stop)
    ID=$(find_instance); no_instance "$ID" && { echo "no instance"; exit 0; }
    aws ec2 stop-instances --region $REGION --instance-ids "$ID" >/dev/null
    echo "stopping $ID, waiting for it to stop"
    aws ec2 wait instance-stopped --region $REGION --instance-ids "$ID" || true
    STATE=$(instance_state "$ID")
    [ "$STATE" = "stopped" ] || { echo "$ID is $STATE, not stopped; it may still be billing" >&2; exit 1; }
    echo "$ID is stopped"
    ;;
  start)
    ID=$(find_instance); no_instance "$ID" && { echo "no instance"; exit 1; }
    aws ec2 start-instances --region $REGION --instance-ids "$ID" >/dev/null
    aws ec2 wait instance-running --region $REGION --instance-ids "$ID"
    echo "instance $ID at $(public_ip "$ID"); the NVMe instance store is empty after a stop"
    ;;
  down)
    ID=$(find_instance); no_instance "$ID" && { echo "no instance"; exit 0; }
    aws ec2 terminate-instances --region $REGION --instance-ids "$ID" >/dev/null
    echo "terminating $ID, waiting for it to finish"
    aws ec2 wait instance-terminated --region $REGION --instance-ids "$ID" || true
    STATE=$(instance_state "$ID")
    [ "$STATE" = "terminated" ] || { echo "$ID is $STATE, not terminated; it may still be billing" >&2; exit 1; }
    echo "$ID is terminated"
    ;;
  status)
    # Capture first: piping straight into grep let a failed describe call read as
    # "no instance", a false all-clear on exactly the question this answers.
    LIVE=$(aws ec2 describe-instances --region $REGION \
      --filters "Name=tag:Name,Values=$NAME" "Name=instance-state-name,Values=pending,running,shutting-down,stopping,stopped" \
      --query 'Reservations[].Instances[].[InstanceId,InstanceType,State.Name,LaunchTime]' --output text) \
      || { echo "could not list instances; billing state unknown" >&2; exit 1; }
    if [ -n "$LIVE" ]; then echo "$LIVE"; else echo "no $NAME instance in any live state"; fi
    ;;
  ssh-config)
    ID=$(find_instance); no_instance "$ID" && { echo "no instance" >&2; exit 1; }
    IP=$(public_ip "$ID")
    python3 - "$IP" "$KEY_FILE" "$ALIAS" <<'PY'
import re, sys, pathlib
ip, key, alias = sys.argv[1], sys.argv[2], sys.argv[3]
p = pathlib.Path.home() / ".ssh" / "config"
text = p.read_text() if p.exists() else ""
block = f"Host {alias}\n  HostName {ip}\n  User ubuntu\n  IdentityFile {key}\n  StrictHostKeyChecking accept-new\n"
text = re.sub(rf"Host {alias}\n(?:  .*\n)*", "", text)
p.write_text(text.rstrip("\n") + ("\n\n" if text.strip() else "") + block)
print(block)
PY
    ;;
  allow-ip) allow_ip ;;
  sync)
    rsync -az --delete --exclude .venv --exclude .git --exclude results --exclude __pycache__ \
      "$HERE/../" "$ALIAS":/mnt/nvme/moefy/code/
    ;;
  *) echo "usage: infra/gpu.sh up [type]|stop|start|down|status|ssh-config|allow-ip|sync" >&2; exit 1 ;;
esac
