#!/usr/bin/env python
"""Script to remove generate-manga-description endpoint and fix file."""

with open("app/api/routes.py", "r", encoding="utf-8") as f:
    content = f.read()

# Remove the generate-manga-description function
start_marker = '@router.post("/generate-manga-description")'
end_marker = '            status_code=500, detail=f"Failed to generate manga description: {str(e)}"'

start_idx = content.find(start_marker)
end_idx = content.find(end_marker)

if start_idx != -1 and end_idx != -1:
    # Find the end of the line
    end_line_end = content.find("\n", end_idx)

    # Remove the section
    new_content = content[:start_idx] + content[end_line_end + 1 :]

    # Write the file
    with open("app/api/routes.py", "w", encoding="utf-8") as f:
        f.write(new_content)

    print(f"Removed generate-manga-description endpoint")
else:
    print("Could not find the section to remove")
