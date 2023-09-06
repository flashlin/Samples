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

   -- Theme Dark
   use('joshdick/onedark.vim')
   -- Statusline
   use('hoob3rt/lualine.nvim')

   use 'nvim-lua/plenary.nvim' -- common utilities
   use 'nvim-telescope/telescope.nvim'
   use 'nvim-telescope/telescope-file-browser.nvim'

   use 'kyazdani42/nvim-web-devicons' -- File icons
   use 'L3MON4D3/LuaSnip' -- snippet

   --use 'onsails/lspkind-nvim'  --vscode-like pictograms
   --use 'hrsh7th/cmp-buffer' -- for buffer words
   --use 'hrsh7th/cmp-nvim-lsp' -- nvim-cmp source for neovim's build-in LSP
   --use 'hrsh7th/nvim-cmp' -- Completion
   --use 'neovim/nvim-lspconfig' --LSP

   use {"kyazdani42/nvim-tree.lua"}

   use({
      'glepnir/zephyr-nvim',
      requires = { 'nvim-treesitter/nvim-treesitter', opt = true },
   })
   
end)


