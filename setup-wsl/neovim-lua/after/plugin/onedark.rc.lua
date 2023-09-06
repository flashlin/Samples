local status, n pcall(require, 'onedark')
if (not status) then return end

vim.opt.termguicolors = true
vim.cmd('colorscheme onedark')
