#!/bin/bash
# First boot of the GPU box (Deep Learning Base OSS Nvidia Driver AMI, Ubuntu 24.04).
# gpu.sh fills in the __BUCKET__, __REGION__ and __HF_SECRET__ placeholders before launch.
# Mounts the largest instance-store NVMe at /mnt/nvme, installs uv, exports the
# Hugging Face token from Secrets Manager, and restores mirrored data from S3.
set -euxo pipefail
BUCKET=__BUCKET__
ROOT=/mnt/nvme/moefy

# Instance store. The Deep Learning AMI already formats the ephemeral NVMe as an
# LVM volume and mounts it at /opt/dlami/nvme, so reuse that mount when it
# exists. The raw-device path below only runs on an AMI that leaves the store
# untouched.
if mountpoint -q /opt/dlami/nvme; then
  # An empty /mnt/nvme directory left by an earlier failed mount is replaced by the link.
  if [ -d /mnt/nvme ] && [ ! -L /mnt/nvme ] && ! mountpoint -q /mnt/nvme; then rmdir /mnt/nvme; fi
  [ -e /mnt/nvme ] || ln -s /opt/dlami/nvme /mnt/nvme
elif ! mountpoint -q /mnt/nvme; then
  ROOT_DEV=$(findmnt -n -o SOURCE / | sed -E 's/p?[0-9]+$//')
  DEV=$(lsblk -dn -o NAME,SIZE,TYPE -b | awk '$3=="disk"{print "/dev/"$1, $2}' | grep -v "^$ROOT_DEV " | sort -k2 -n | tail -1 | cut -d' ' -f1)
  if [ -n "$DEV" ]; then
    blkid "$DEV" >/dev/null 2>&1 || mkfs.ext4 -F "$DEV"
    mkdir -p /mnt/nvme && mount "$DEV" /mnt/nvme
  fi
fi
mkdir -p "$ROOT" && chown -R ubuntu:ubuntu "$ROOT" && chown ubuntu:ubuntu "$(readlink -f /mnt/nvme)"

sudo -u ubuntu bash -c '
  set -euo pipefail
  curl -LsSf https://astral.sh/uv/install.sh | sh
  TOKEN=$(aws secretsmanager get-secret-value --region __REGION__ --secret-id __HF_SECRET__ --query SecretString --output text)
  # Exports go at the TOP of .bashrc. Ubuntu'"'"'s stock .bashrc returns early for
  # non-interactive shells, and every `ssh moefy "cmd"` in this project is one;
  # bash still reads .bashrc for a remote command, but only up to that return.
  cat > ~/.moefy_env <<ENV
export PATH=\$HOME/.local/bin:/usr/local/cuda/bin:\$PATH
export HF_TOKEN=$TOKEN
export HF_HOME='"$ROOT"'/hf
export HF_XET_HIGH_PERFORMANCE=1
export MOEFY_ROOT='"$ROOT"'
export MOEFY_BUCKET='"$BUCKET"'
ENV
  chmod 600 ~/.moefy_env
  grep -q "moefy_env" ~/.bashrc || { { echo ". ~/.moefy_env"; cat ~/.bashrc; } > ~/.bashrc.new && mv ~/.bashrc.new ~/.bashrc; }
  mkdir -p '"$ROOT"'/hf
  aws s3 ls s3://'"$BUCKET"'/ >/dev/null 2>&1 && for d in results; do
    aws s3 sync s3://'"$BUCKET"'/$d '"$ROOT"'/$d --only-show-errors || true
  done
'
df -h /mnt/nvme > /home/ubuntu/first-boot.txt
nvidia-smi >> /home/ubuntu/first-boot.txt || true
