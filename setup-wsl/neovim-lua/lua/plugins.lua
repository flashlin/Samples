local status, packer = pcall(require, 'packer')
if (not status) then
   print("Packer is not installed")
   return
end

vim.cmd [[packadd packer.nvim]]

-- PackerInstall
packer.startup(function(use)
   use 'wbthomason/packer.nvim'

   -- Load on specific commands
   use { 'folke/lazy.nvim', lazy = false, priority = 1000, opts = {} }
   use('joshdick/onedark.vim') -- Theme Dark
   use('hoob3rt/lualine.nvim') -- Statusline
   -- use 'akinsho/nvim-bufferline.lua'

   use {
      'phaazon/hop.nvim',
      branch = 'v2', -- optional but strongly recommended
      config = function()
        -- you can configure Hop the way you like here; see :h hop-config
        require'hop'.setup { keys = 'etovxqpdygfblzhckisuran' }
      end
    }

   -- file browser 
   use 'nvim-lua/plenary.nvim' -- common utilities
   use 'nvim-telescope/telescope.nvim'
   use 'nvim-telescope/telescope-file-browser.nvim'
   use {"kyazdani42/nvim-tree.lua"}
   use({
      'glepnir/zephyr-nvim',
      requires = { 'nvim-treesitter/nvim-treesitter', opt = true },
   })

   use 'windwp/nvim-autopairs'
   use 'kyazdani42/nvim-web-devicons' -- File icons

   use "nvim-lua/popup.nvim"

   use 'lewis6991/gitsigns.nvim'
   use 'dinhhuy258/git.nvim' 
   
   -- cmp plugins
   use 'hrsh7th/nvim-cmp' -- Completion
   use 'hrsh7th/cmp-buffer' -- for buffer words
   use 'hrsh7th/cmp-path'
   --use 'hrsh7th/cmp-cmpline'
   --use 'saadparwaiz1/cmp-luasnip' -- snippet completions
   
   -- snippets
   use 'L3MON4D3/LuaSnip' -- snippet
   use 'rafamadriz/friendly-snippets'

   -- use 'neovim/nvim-lspconfig' --LSP :LspInstallInfo
   -- :LspInstall --sync sumneko_lua 安裝 lua
   -- :LspInstall --sync lua_ls 安裝 lua
   use({
      "williamboman/nvim-lsp-installer",
      "neovim/nvim-lspconfig",
   })

   use 'jose-elias-alvarez/null-ls.nvim' -- Use neovim as a lanuage server
   use 'MunifTanjim/prettier.nvim' -- prettier plugin for neovim's build-in LSP client

   use 'onsails/lspkind-nvim'  --vscode-like pictograms
   --use 'glepnir/lspsaga.nvim' -- LSP UIs
   use 'hrsh7th/cmp-nvim-lsp' -- nvim-cmp source for neovim's build-in LSP
   --use 'williamboman/nvim-lsp-installer' -- HELP install LSP
   
end)


