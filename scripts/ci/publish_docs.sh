#!/usr/bin/env bash
# Push the site that `pixi run docs` builds to the docs-html branch, which Read the Docs
# serves. The branch holds one orphan commit with the site in html/ and
# docs/rtd/readthedocs.yaml as its .readthedocs.yaml.
#
# Environment
#   DOCS_REMOTE   git remote to push to (default origin)
#   DOCS_DRY_RUN  1 builds the commit without pushing it
set -euo pipefail

SITE="build/docs/sphinx"
BRANCH="docs-html"
RTD_CONFIG="docs/rtd/readthedocs.yaml"

REMOTE="${DOCS_REMOTE:-origin}"

[ -d "$SITE" ] || { echo "No built site at $SITE. Run 'pixi run docs' first." >&2; exit 1; }
[ -f "$RTD_CONFIG" ] || { echo "Missing $RTD_CONFIG." >&2; exit 1; }

# Build the commit from a temporary tree with git plumbing. The working tree and the
# current branch do not change.
STAGE="$(mktemp -d)"
INDEX="$(mktemp -u)"
trap 'rm -rf "$STAGE" "$INDEX"' EXIT

mkdir -p "$STAGE/html"
cp -R "$SITE/." "$STAGE/html/"
# Drop the Sphinx build caches.
rm -rf "$STAGE/html/.doctrees" "$STAGE/html/.buildinfo"
touch "$STAGE/html/.nojekyll"
cp "$RTD_CONFIG" "$STAGE/.readthedocs.yaml"

# -f adds files that .gitignore lists, such as searchindex.json.
GIT_INDEX_FILE="$INDEX" git --work-tree="$STAGE" add -f -A
TREE="$(GIT_INDEX_FILE="$INDEX" git --work-tree="$STAGE" write-tree)"
# commit-tree needs an identity. GIT_AUTHOR_NAME and GIT_AUTHOR_EMAIL replace the default.
GIT_NAME="${GIT_AUTHOR_NAME:-geodex-docs}"
GIT_EMAIL="${GIT_AUTHOR_EMAIL:-geodex-docs@users.noreply.github.com}"
COMMIT="$(git -c user.name="$GIT_NAME" -c user.email="$GIT_EMAIL" \
  commit-tree "$TREE" -m "docs: publish rendered site")"

if [ "${DOCS_DRY_RUN:-0}" = "1" ]; then
  echo "[dry-run] built orphan commit $COMMIT for $BRANCH (not pushed)"
  echo "[dry-run] tree contents:"
  git ls-tree -r --name-only "$TREE" | sed 's/^/  /' | head -40
  exit 0
fi

git push -f "$REMOTE" "$COMMIT:refs/heads/$BRANCH"
echo "Published $SITE -> $REMOTE/$BRANCH"
