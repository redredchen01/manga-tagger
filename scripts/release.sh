#!/bin/bash
set -e

VERSION=$1
if [ -z "$VERSION" ]; then
    echo "Usage: $0 <version>"
    echo "Example: $0 2.1.0"
    exit 1
fi

# Update version in pyproject.toml
sed -i "s/version = .*/version = \"$VERSION\"/" pyproject.toml

# Update version in app/main.py
sed -i "s/version=\".*\"/version=\"$VERSION\"/" app/main.py

# Commit changes
git add pyproject.toml app/main.py
git commit -m "chore: bump version to $VERSION"

# Create tag
git tag -a "v$VERSION" -m "Release v$VERSION"

# Push
git push origin main --tags

echo "✅ Released v$VERSION"
echo "GitHub Actions will create the release automatically"
