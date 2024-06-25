-- https://www.josean.com/posts/how-to-setup-neovim-2024
-- 當建立 lspconfig.lua , 就得去修改 nvim-cmp.lua 並進行以下變更以將 lsp 新增為完成來源(在 `sources = cmp.config.sources` 地方)
return {
    "neovim/nvim-lspconfig",
    event = { "BufReadPre", "BufNewFile" },
    dependencies = {
      "hrsh7th/cmp-nvim-lsp",
      { "antosha417/nvim-lsp-file-operations", config = true },
      { "folke/neodev.nvim", opts = {} },
    },
    config = function()
      -- import lspconfig plugin
      local lspconfig = require("lspconfig")
  
      -- import mason_lspconfig plugin
      local mason_lspconfig = require("mason-lspconfig")
  
      -- import cmp-nvim-lsp plugin
      local cmp_nvim_lsp = require("cmp_nvim_lsp")
  
      local keymap = vim.keymap -- for conciseness
  
      vim.api.nvim_create_autocmd("LspAttach", {
        group = vim.api.nvim_create_augroup("UserLspConfig", {}),
        callback = function(ev)
          -- Buffer local mappings.
          -- See `:help vim.lsp.*` for documentation on any of the below functions
          local opts = { buffer = ev.buf, silent = true }
  
          -- set keybinds
          opts.desc = "Show LSP references"
          keymap.set("n", "gR", "<cmd>Telescope lsp_references<CR>", opts) -- show definition, references
  
          opts.desc = "Go to declaration"
          keymap.set("n", "gD", vim.lsp.buf.declaration, opts) -- go to declaration
  
          opts.desc = "Show LSP definitions"
          keymap.set("n", "gd", "<cmd>Telescope lsp_definitions<CR>", opts) -- show lsp definitions
  
          opts.desc = "Show LSP implementations"
          keymap.set("n", "gi", "<cmd>Telescope lsp_implementations<CR>", opts) -- show lsp implementations
  
          opts.desc = "Show LSP type definitions"
          keymap.set("n", "gt", "<cmd>Telescope lsp_type_definitions<CR>", opts) -- show lsp type definitions
  
          opts.desc = "See available code actions"
          keymap.set({ "n", "v" }, "<leader>ca", vim.lsp.buf.code_action, opts) -- see available code actions, in visual mode will apply to selection
  
          opts.desc = "Smart rename"
          keymap.set("n", "<leader>rn", vim.lsp.buf.rename, opts) -- smart rename
  
          opts.desc = "Show buffer diagnostics"
          keymap.set("n", "<leader>D", "<cmd>Telescope diagnostics bufnr=0<CR>", opts) -- show  diagnostics for file
  
          opts.desc = "Show line diagnostics"
          keymap.set("n", "<leader>d", vim.diagnostic.open_float, opts) -- show diagnostics for line
  
          opts.desc = "Go to previous diagnostic"
          keymap.set("n", "[d", vim.diagnostic.goto_prev, opts) -- jump to previous diagnostic in buffer
  
          opts.desc = "Go to next diagnostic"
          keymap.set("n", "]d", vim.diagnostic.goto_next, opts) -- jump to next diagnostic in buffer
  
          opts.desc = "Show documentation for what is under cursor"
          keymap.set("n", "K", vim.lsp.buf.hover, opts) -- show documentation for what is under cursor
  
          opts.desc = "Restart LSP"
          keymap.set("n", "<leader>rs", ":LspRestart<CR>", opts) -- mapping to restart lsp if necessary
        end,
      })
  
      -- used to enable autocompletion (assign to every lsp server config)
      local capabilities = cmp_nvim_lsp.default_capabilities()
  
      -- Change the Diagnostic symbols in the sign column (gutter)
      -- (not in youtube nvim video)
      local signs = { Error = " ", Warn = " ", Hint = "󰠠 ", Info = " " }
      for type, icon in pairs(signs) do
        local hl = "DiagnosticSign" .. type
        vim.fn.sign_define(hl, { text = icon, texthl = hl, numhl = "" })
      end

      local home = os.getenv("HOME")
      local util = require 'lspconfig.util'
      local function get_typescript_server_path(root_dir)
        local global_ts = home .. '/flash/.npm/lib/node_modules/typescript/lib'
        -- Alternative location if installed as root:
        -- local global_ts = '/usr/local/lib/node_modules/typescript/lib'
        local found_ts = ''
        local function check_dir(path)
          found_ts =  util.path.join(path, 'node_modules', 'typescript', 'lib')
          if util.path.exists(found_ts) then
            return path
          end
        end
        if util.search_ancestors(root_dir, check_dir) then
          return found_ts
        else
          return global_ts
        end
      end

      mason_lspconfig.setup_handlers({
        -- default handler for installed servers
        function(server_name)
          -- lspconfig[server_name].setup({
          --   capabilities = capabilities,
          -- })
          local server_config = {}
          -- if require("neoconf").get(server_name .. ".disable") then
          --   return
          -- end
          if server_name == "tsserver" then
            server_config.init_options = {
              plugins = {{
                  name = "@vue/typescript-plugin",
                  location = home .. "/.local/share/pnpm/global/5/node_modules/@vue/typescript-plugin",
                  languages = { "javascript", "typescript", "vue" },
                }
              },
            }
            server_config.filetypes = { 'vue', 'typescript', 'javascript' }
          end
          -- if server_name == "volar" then
          --   server_config.init_options = {
          --     typescript = {
          --       tsdk = home .. '/.local/share/pnpm/global/5/node_modules/typescript/lib',
          --     }
          --   }
          --   server_config.filetypes = { 'vue', 'typescript', 'javascript' }
          --   return
          -- end
          lspconfig[server_name].setup(server_config)
        end,
        ["volar"] = function()
          lspconfig["volar"].setup({
            capabilities = capabilities,
            filetypes = { "typescript", "javascript", "javascriptreact", "typescriptreact", "vue", "json" },
            init_options = {
              typescript = {
                --tsdk = vim.fn.expand("$HOME/.local/share/nvim/mason/packages/typescript-language-server/node_modules/typescript/lib")
                tsdk = vim.fn.expand(home .. "/.local/share/nvim/mason/packages/typescript-language-server/node_modules/typescript/lib")
              },
              vue = {
                hybridMode = true, -- 啟用混合模式，可以更好地處理 .vue 文件
              },
            },
            on_new_config = function(new_config, new_root_dir)
              new_config.init_options.typescript.tsdk = get_typescript_server_path(new_root_dir)
            end,
          })
        end,
        ["svelte"] = function()
          -- configure svelte server
          lspconfig["svelte"].setup({
            capabilities = capabilities,
            on_attach = function(client, bufnr)
              vim.api.nvim_create_autocmd("BufWritePost", {
                pattern = { "*.js", "*.ts" },
                callback = function(ctx)
                  -- Here use ctx.match instead of ctx.file
                  client.notify("$/onDidChangeTsOrJsFile", { uri = ctx.match })
                end,
              })
            end,
          })
        end,
        ["graphql"] = function()
          -- configure graphql language server
          lspconfig["graphql"].setup({
            capabilities = capabilities,
            filetypes = { "graphql", "gql", "svelte", "typescriptreact", "javascriptreact" },
          })
        end,
        ["emmet_ls"] = function()
          -- configure emmet language server
          lspconfig["emmet_ls"].setup({
            capabilities = capabilities,
            filetypes = { "html", "typescriptreact", "javascriptreact", "css", "sass", "scss", "less", "svelte" },
          })
        end,
        ["lua_ls"] = function()
          -- configure lua server (with special settings)
          lspconfig["lua_ls"].setup({
            capabilities = capabilities,
            settings = {
              Lua = {
                -- make the language server recognize "vim" global
                diagnostics = {
                  globals = { "vim" },
                },
                completion = {
                  callSnippet = "Replace",
                },
              },
            },
          })
        end,
      })
    end,
  }