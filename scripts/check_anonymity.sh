#!/usr/bin/env bash
# Grep every release candidate file (tracked plus untracked-but-not-ignored) for strings that could
# de-anonymize the submission. Exits 1 and prints file:line:match for every hit, 0 when clean.
#
#   scripts/check_anonymity.sh
#
# The script itself names nobody. Identity terms come from the local git identity (user.name words,
# user.email), the login name, $HOME, and an optional private file of extra terms, one extended regex
# per line (default: .anonymity_terms in the repo root, which is git-ignored; override with
# ANON_TERMS=path). Put author names, institution names and cluster host names there.
set -u
cd "$(git rev-parse --show-toplevel)" || exit 2

terms_file="${ANON_TERMS:-.anonymity_terms}"
self="scripts/check_anonymity.sh"

# Generic patterns: e-mail addresses, absolute home and cluster paths, scheduler identity lines.
patterns=(
  '[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}'
  '/Users/[A-Za-z0-9._-]+'
  '/home/[A-Za-z0-9._-]+'
  '/(scratch|work|project|projects|lustre|gpfs|nfs|mnt/[a-z]+)/[A-Za-z0-9._-]+'
  '#SBATCH[[:space:]]+--(account|mail-user|partition)'
)

# Identity terms from the local environment (whole words, case-insensitive).
add_word() { [ "${#1}" -ge 3 ] && patterns+=("\\b$1\\b"); }
for word in $(git config user.name 2>/dev/null); do add_word "$word"; done
email="$(git config user.email 2>/dev/null)"; [ -n "$email" ] && patterns+=("${email%@*}")
add_word "$(id -un)"
patterns+=("$HOME")
if [ -f "$terms_file" ]; then
  while IFS= read -r line; do [ -n "$line" ] && [ "${line#\#}" = "$line" ] && patterns+=("$line"); done < "$terms_file"
else
  echo "note: $terms_file not found; checking generic patterns and the git identity only" >&2
fi

files=()
while IFS= read -r f; do
  [ -f "$f" ] && [ "$f" != "$self" ] && [ "$f" != "$terms_file" ] && files+=("$f")
done < <({ git ls-files; git ls-files --others --exclude-standard; } | sort -u)

hits=0
for p in "${patterns[@]}"; do
  out="$(grep -I -n -i -o -E -- "$p" "${files[@]}" 2>/dev/null)"
  if [ -n "$out" ]; then
    echo "$out"
    hits=$((hits + $(printf '%s\n' "$out" | wc -l)))
  fi
done
echo "checked ${#files[@]} files against ${#patterns[@]} patterns: $hits hit(s)"
[ "$hits" -eq 0 ]
