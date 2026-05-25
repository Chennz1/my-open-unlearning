#!/usr/bin/env bash
set -euo pipefail

# End-to-end wrapper matching the original TVC workflow.
# By default it assumes the original TVC training scripts have already produced:
#   saves/finetune/tofu_${MODEL}_${FORGET_SPLIT}_epoch${EPOCH}
#   saves/finetune/tofu_${MODEL}_${RETAIN_SPLIT}_mimic_epoch${EPOCH}
#
# Set RUN_TRAIN=1 only if you want to call the original training scripts first.
# Note: keep the split/epoch settings in those original scripts aligned with this
# search script before enabling RUN_TRAIN.

RUN_TRAIN=${RUN_TRAIN:-0}
if [[ "$RUN_TRAIN" == "1" ]]; then
  bash scripts/tofu_finetune_forget.sh
  bash scripts/tofu_finetune_sretain.sh
fi

bash scripts/search_hyperparams_ia_tvc_tofu_7b_forget10.sh
