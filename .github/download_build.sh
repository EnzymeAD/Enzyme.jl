#!/bin/bash -e

# handle user inputs
[ $# -ne 3 ] && { echo "Usage: $0 <version> <build_name> <destination_file>" >&2; exit 1; }
VERSION="$1"
BUILD_NAME="$2"
DEST_FILE="$3"
[ -z "$BUILDKITE_TOKEN" ] && { echo "BUILDKITE_TOKEN not set." >&2; exit 1; }

API_BASE="https://api.buildkite.com/v2"
ORG="julialang"

# derive pipeline and branch filter from version
if [ "$VERSION" = "master" ]; then
    PIPELINE="julia-master"
    BRANCH_FILTER='.branch == "master"'
else
    PIPELINE="julia-release-${VERSION//./-dot-}"
    BRANCH_FILTER="(.branch == \"release-$VERSION\") or (.branch | startswith(\"v$VERSION\"))"
fi

# find the first successful job and get its artifacts url
ARTIFACTS_URL=$(curl -s -H "Authorization: Bearer $BUILDKITE_TOKEN" \
    "$API_BASE/organizations/$ORG/pipelines/$PIPELINE/builds?per_page=100" | \
    jq -r "first(.[] | select($BRANCH_FILTER) | .jobs[] | select(.step_key == \"$BUILD_NAME\" and .exit_status == 0) | .artifacts_url)")
[ -z "$ARTIFACTS_URL" ] || [ "$ARTIFACTS_URL" = "null" ] && { echo "No successful build found."; exit 1; }

# fetch the url of the first artifact
ARTIFACT_URL=$(curl -s -H "Authorization: Bearer $BUILDKITE_TOKEN" "$ARTIFACTS_URL" | \
    jq -r '.[0].download_url')
[ -z "$ARTIFACT_URL" ] || [ "$ARTIFACT_URL" = "null" ] && { echo "No artifact found."; exit 1; }

curl -s -L -H "Authorization: Bearer $BUILDKITE_TOKEN" -o "$DEST_FILE" "$ARTIFACT_URL"
echo "Artifact downloaded as $DEST_FILE"
