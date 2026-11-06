#!/bin/sh
# The evaluation: compare.py against S0 and S1 with ATTACKERS_LIST attackers (default 80 of 200 clients, 40%;
# ATTACKERS_LIST="0 80" adds the honest run),
# each from fresh server state. Results in mltest/results/n{CLIENTS_PER_ROUND}-a{ATTACKERS}.jsonl.
# Other settings pass through, e.g. N_ROUNDS=1 CLIENTS_PER_ROUND=4 ATTACKERS_LIST="0 1" ROUND_SECONDS=30 UPLOAD_MARGIN_S=10 mltest/run.sh
# for a dry run. The run takes ~5 h (estimate: ~6 min per round with 80 attackers, ~10 for an honest round).
set -e
cd "$(dirname "$0")/.."
# compare writes the results as this user; not UID, which bash keeps read-only
export HOST_UID="$(id -u)" HOST_GID="$(id -g)"
export S1_TOKEN="${S1_TOKEN:-$(head -c 32 /dev/urandom | od -An -tx1 | tr -d ' \n')}"
mkdir -p mltest/results
compose="docker compose -f compose.yaml -f compose.mltest.yaml"
$compose build
status=0
for attackers in ${ATTACKERS_LIST:-80}; do
    $compose down -v
    # run, not up --exit-code-from: that stops everything as soon as the one-shot export exits
    ATTACKERS=$attackers $compose run --rm compare || status=1
    $compose logs --no-color s0 s1 > "mltest/results/servers-n${CLIENTS_PER_ROUND:-200}-a$attackers.log"
done
$compose down -v
exit $status
