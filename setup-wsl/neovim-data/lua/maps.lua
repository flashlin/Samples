local keymap = vim.keymap
local opt = {noremap = true, silent = true }
-- Don't yank with x
keymap.set('n', 'x', '"_x')

-- Increment/decrement
keymap.set('n', '+', '<C-a>')
keymap.set('n', '-', '<C-x>')

-- Select All
keymap.set('n', '<C-a>', 'gg<S-v>G')

-- New tab
--vim.api.nvim_set_keymap('n', 's', '<Nop>', { noremap = true, silent = true }) -- 關掉原本 s 刪除功能
keymap.set('n', '<A-\\>', ':split<CR><C-w>w')
keymap.set('n', '<A-|>', ':vsplit<CR><C-w>w')

keymap.set("n", "<A-h>", "<C-w>h", opt)
keymap.set("n", "<A-j>", "<C-w>j", opt)
keymap.set("n", "<A-k>", "<C-w>k", opt)
keymap.set("n", "<A-l>", "<C-w>l", opt)


keymap.set("n", "+", "<C-w>>", opt)
keymap.set("n", "-", "<C-w><", opt)