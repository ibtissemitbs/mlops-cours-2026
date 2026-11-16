#!/usr/bin/env bash

K=(kubectl --context kind-mlops-lab -n scoring)
ISVC=${1:-scoring-model}
SEL="serving.kserve.io/inferenceservice=${ISVC}"

while true; do
  clear
  echo "=== $(date +%T) | Service ${ISVC} ==="
  "${K[@]}" get isvc "$ISVC"
  echo
  echo "=== Autoscaling Knative ==="
  "${K[@]}" get kpa -l "$SEL"
  echo
  echo "=== Pods du modèle ==="
  "${K[@]}" get pods -l "$SEL"
  echo
  echo "=== CPU et mémoire ==="
  "${K[@]}" top pods -l "$SEL" --containers
  sleep 2
done
