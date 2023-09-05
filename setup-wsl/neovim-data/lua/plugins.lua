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

   -- Theme inspired by Atom
   use('joshdick/onedark.vim')

   use({
      'glepnir/zephyr-nvim',
      requires = { 'nvim-treesitter/nvim-treesitter', opt = true },
   })
   
end)


