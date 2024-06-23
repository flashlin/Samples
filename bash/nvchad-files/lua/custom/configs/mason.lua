local mason = {
    ensure_installed = {
      --Some packages don't install with Mason on ARM devices or on Android tablets in Termux. Install them with info given in comment
  
      --c/cpp stuff
    --   "clangd",       --c/cpp lsp and linter ($pkg install clang)
    --   "clang-format", --c/cpp formatter (just ignore, it should work with clangd)
  
      --html/css stuff
      "prettier",                    --formatter for a lot of different things
      "emmet-language-server",       --emmet snippets
      "css-lsp",                     --css lsp
      "tailwindcss-language-server", --tailwindcss
      "html-lsp",                    --html lsp
  
      --javascript/nodejs stuff
      "typescript-language-server", --javascript lsp
      "eslint-lsp",                 --javascript linter
      "vue-language-server",
  
      --php stuff
    --   "intelephense", --php lsp and linter
    --   "php-cs-fixer", --php formatter
  
      --lua stuff
      "lua-language-server", --lua lsp and linter ($pkg install lua-language-server)
      "stylua",              --lua formatter ($pkg install stylua)
  
      --python
      "python-lsp-server", --python lsp and linter
      "black",             --python formatter
  
      --go
    --   "gopls",             --go lsp and linter
    --   "gofumpt",           --go formatter
    --   "goimports-reviser", --go formater for imports ($go install -v github.com/incu6us/goimports-reviser/v3@latest)
    --   "golines",           --go formatter for line length
  
      --ruby
    --   "solargraph", --ruby lsp and linter
  
      --bash
      "bash-language-server",
    },
  }
  
  return mason