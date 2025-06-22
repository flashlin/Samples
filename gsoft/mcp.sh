npx playwright-mcp &
cd interactive-feeback-mcp
# uv run fastmcp dev server.py
uv run ./server.py &
cd ..

# 請按照下面步驟設定
# https://chromewebstore.google.com/detail/browser-mcp-automate-your/bjfgambnhccakkhmkepdoekmckoijdlc
# https://docs.browsermcp.io/setup-extension
# https://docs.browsermcp.io/setup-server



# uvx --from git+https://github.com/oraios/serena serena-mcp-server
# serena: {
#   "command": "uvx",
#   "args": ["--from", "git+https://github.com/oraios/serena", "serena-mcp-server"]
# }

# git clone https://github.com/oraios/serena
cd serena
uv run serena-mcp-server &
cd ..