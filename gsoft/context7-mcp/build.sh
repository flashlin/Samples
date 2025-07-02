docker build -t context7-mcp .

echo '"context7-mcp": {
  "command": "docker",
  "args": ["run", "-i", "--rm", "context7-mcp"]
}
'