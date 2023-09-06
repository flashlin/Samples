local status, telescope = pcall(require, 'telescope')
if (not status) then return end

local actions = require('telescope.actions')

function telescope_buffer_dir()
   return vim.fn.expand('%:p:h')
end

local fb_actions = require 'telescope'.extensions.file_browser.actions

telescope.setup{
   defaults = {
      mappings = {
         n = {
            ['q'] = actions.close
         }
      }
   },
   extensions = {
      file_browser = {
         theme = 'dropdown',
         -- disables netrw and add use file-browser
         hijack_netrw = true,
         mappings = {
            -- your custom insert mode mappings
            ['i'] = {
               ['<C-w>'] = function() vim.cmd('normal vbd') end,
            },
            ['n'] = {
               ['N'] = fb_actions.create,
               ['h'] = fb_actions.goto_parent_dir,
               ['/'] = function()
                  vim.cmd('startinsert')
               end
            }
         }
      }
   }
}

telescope.load_extension('file_browser')
local opts = { noremap = true, silent = true }
vim.keymap.set('n', '<C-t>', '<cmd>lua require("telescope.builtin").find_files({ ignore = true, hidden = false })<cr>', opts)
vim.keymap.set('n', '<C-f>', '<cmd>lua require("telescope.builtin").live_grep()<cr>', opt)
vim.keymap.set('n', '<C-\\>', '<cmd>lua require("telescope.builtin").buffers()<cr>', opt)
vim.keymap.set('n', ';;', '<cmd>lua require("telescope.builtin").help_tags()<cr>', opt)
vim.keymap.set('n', ';;', '<cmd>lua require("telescope.builtin").resume()<cr>', opt)
vim.keymap.set('n', ';;', '<cmd>lua require("telescope.builtin").diagnostics()<cr>', opt)
vim.keymap.set('n', '<C-e>', '<cmd>lua require("telescope").extensions.file_browser.file_browser({ path = "%:p:h", cwd = telescope_buffer_dir(), respect_git_ignore = false, hidden = true, grouped = true, previewer = false, initial_mode = "normal", layout_config = {height=40} })<cr>', opt)


