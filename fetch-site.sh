#!/usr/bin/env bash
# Récupère ce qui est déjà publié (historique des tuiles, climat) avant de le mettre à jour.
set -uo pipefail
rm -rf site
if git clone -q --depth 1 -b gh-pages "https://github.com/${GITHUB_REPOSITORY}.git" site; then rm -rf site/.git; else mkdir -p site; fi
