#!/bin/sh
# Fetch only the pinned Change-Agent source/assets and the permitted academic example.
set -eu

PROJECT_ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
SOURCE_DIR="$PROJECT_ROOT/.cache/upstream/change-agent"
CHECKPOINT="$PROJECT_ROOT/weights/MCI_model.pth"
AUXILIARY="$SOURCE_DIR/Multi_change/model/pretrained/mit_b1.pth"
VOCABULARY="$SOURCE_DIR/Multi_change/data/LEVIR_MCI/vocab.json"
BEFORE="$PROJECT_ROOT/data/input/before.png"
AFTER="$PROJECT_ROOT/data/input/after.png"

CHANGE_AGENT_REVISION=68cbaa7f388b36e4fc10872f7a2911482d26ae5b
RSICC_REVISION=d1505e514c450c3728782ca723e82761e70bafd3

download_if_missing() {
    url=$1
    destination=$2
    if [ -f "$destination" ]; then
        echo "Using existing $destination"
        return
    fi
    mkdir -p "$(dirname -- "$destination")"
    curl --fail --location --retry 3 --output "$destination.part" "$url"
    mv "$destination.part" "$destination"
}

verify_sha256() {
    expected=$1
    path=$2
    actual=$(shasum -a 256 "$path" | awk '{print $1}')
    if [ "$actual" != "$expected" ]; then
        echo "SHA-256 mismatch for $path: expected $expected, got $actual" >&2
        exit 1
    fi
    echo "Verified $path ($actual)"
}

if [ ! -d "$SOURCE_DIR/.git" ]; then
    mkdir -p "$(dirname -- "$SOURCE_DIR")"
    git clone --filter=blob:none --no-checkout \
        https://github.com/Chen-Yang-Liu/Change-Agent.git "$SOURCE_DIR"
    git -C "$SOURCE_DIR" sparse-checkout init --no-cone
    git -C "$SOURCE_DIR" sparse-checkout set \
        /README.md \
        /LICENSE.txt \
        '/Multi_change/model/*.py' \
        /Multi_change/data/LEVIR_MCI/vocab.json
    git -C "$SOURCE_DIR" checkout --detach "$CHANGE_AGENT_REVISION"
fi

actual_revision=$(git -C "$SOURCE_DIR" rev-parse HEAD)
if [ "$actual_revision" != "$CHANGE_AGENT_REVISION" ]; then
    echo "Unexpected Change-Agent revision: $actual_revision" >&2
    exit 1
fi

download_if_missing \
    https://huggingface.co/lcybuaa/Change-Agent/resolve/main/MCI_model.pth \
    "$CHECKPOINT"
download_if_missing \
    "https://raw.githubusercontent.com/Chen-Yang-Liu/Change-Agent/$CHANGE_AGENT_REVISION/Multi_change/model/pretrained/mit_b1.pth" \
    "$AUXILIARY"
download_if_missing \
    "https://raw.githubusercontent.com/Chen-Yang-Liu/Change-Agent/$CHANGE_AGENT_REVISION/Multi_change/data/LEVIR_MCI/vocab.json" \
    "$VOCABULARY"

# These two RSICC example files are used only for the documented academic test.
# The upstream repository has no explicit dataset licence; do not redistribute them.
download_if_missing \
    "https://raw.githubusercontent.com/Chen-Yang-Liu/RSICC/$RSICC_REVISION/Example/A/train_000016.png" \
    "$BEFORE"
download_if_missing \
    "https://raw.githubusercontent.com/Chen-Yang-Liu/RSICC/$RSICC_REVISION/Example/B/train_000016.png" \
    "$AFTER"

verify_sha256 34c6926342c40fdd6d50b43d50257cb46cb6b61763448de9ee551828a64b3eb9 "$CHECKPOINT"
verify_sha256 26f8b1a980ad08ddda46f82ba5008b2f6322f11939d59313285c4b477bfe60b1 "$AUXILIARY"
verify_sha256 c9323565663797aac437e1ad4cd5f0f0a7da8b08d8d438bcfa70708ddcca8fc5 "$VOCABULARY"
verify_sha256 8e5d586a4d6af2c658591fd2b9b16142fc9d26100f76ba58069f90de623668f7 "$BEFORE"
verify_sha256 c92e9b37043c9fec9ef9cbfba6c2b4b4d40f3be2546efd314cfe70a9ed00a3fc "$AFTER"

"$PROJECT_ROOT/.venv/bin/python" "$PROJECT_ROOT/scripts/patch_change_agent.py" "$SOURCE_DIR"
