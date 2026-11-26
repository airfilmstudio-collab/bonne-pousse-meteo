#!/usr/bin/env bash
# Publie le dossier site/ sur la branche gh-pages (un seul commit, pour garder le dépôt léger).
set -euo pipefail
MSG="$1"
cp -r pages/. site/ 2>/dev/null || true
touch site/.nojekyll
cd site
rm -rf .git
git init -q
git checkout -q -b gh-pages
git add -A
git -c user.name="bonne-pousse-bot" -c user.email="bonne-pousse-bot@users.noreply.github.com" commit -qm "$MSG"
git push -qf "https://x-access-token:${GITHUB_TOKEN}@github.com/${GITHUB_REPOSITORY}.git" gh-pages
echo "Publié : $MSG"
