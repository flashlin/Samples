local status, telescope = pcall(require, 'telescope')
if (not status) then return end

local actions = require('telescope.actions')

local function telescope_buffer_dir()
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
         Theme = 'dropdown',
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
vim.keymap.set('n', ';f', '<cmd>lua require("telescope.builtin").find_files({ no_ignore=false, hidden=true })<CR>', 
   opts)


