require "nvchad.mappings"
local map = vim.keymap.set

map("n", ";", ":", { desc = "CMD enter command mode" })
map("i", "jk", "<ESC>")

map("n", "<C-p>",
    function( )
        require( 'telescope.builtin'). find_files({
            cwd = vim.fn. expand('%:p:h'),
            hidden = true,
            file_ignore_patterns = {
                ".*/.git/",
                ".*/node_modules/",
                ".*/bin/",
                ".*/obj/",
            }
        })
    end,
    { desc = "Find files in current directory" }
)

map( "n", "<C-t>", 
    function()
        require('telescope.builtin'). live_grep({
            cwd = vim.fn. expand ('%:p:h'),
            grep_open_files = false,
            hidden = true,
        })
    end, 
    { desc = "Search content in current directory" }
)

map({"n", "v", "i", "c"}, "<C-c>", [["+y]], { desc = "Copy to system clipboard" })
