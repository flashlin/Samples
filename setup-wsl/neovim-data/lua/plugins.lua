return require('packer').startup(function(use)
   -- Load on specific commands
   use { 'folke/lazy.nvim', lazy = false, priority = 1000, opts = {} }
   colorscheme tokyonight-night
   
end)
