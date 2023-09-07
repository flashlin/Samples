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
   use 'L3MON4D3/LuaSnip' -- snippet

   use 'lewis6991/gitsigns.nvim'
   use 'dinhhuy1558/git.nvim' 

   use 'neovim/nvim-lspconfig' --LSP :LspInstallInfo
   use 'jose-elias-alvarez/null-ls.nvim' -- Use neovim as a lanuage server
   use 'MunifTanjim/prettier.nvim' -- prettier plugin for neovim's build-in LSP client

   use 'onsails/lspkind-nvim'  --vscode-like pictograms
   --use 'glepnir/lspsaga.nvim' -- LSP UIs
   use 'hrsh7th/cmp-buffer' -- for buffer words
   use 'hrsh7th/cmp-nvim-lsp' -- nvim-cmp source for neovim's build-in LSP
   use 'hrsh7th/nvim-cmp' -- Completion
   use 'williamboman/nvim-lsp-installer' -- HELP install LSP
   
end)


