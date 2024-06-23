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

map("n", "<C-t>", 
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


map("n", "<leader>tt", ":tabnew<CR>", { desc = "New tab space" })
map("n", "<leader>tp", ":tabprev<CR>", { desc = "Go prev tab space" })
map("n", "<leader>tn", ":tabnext<CR>", { desc = "Go next tab space" })


require "pounce-mappings"


map("n", "<leader>jm", 
    function()
        require("custom.functions").jump_to_next_ts_method()
    end,
    { desc = "Jump to next TypeScript method in Vue file" }
)

map("n", "<leader>|", ":vsp<CR>", { desc = "垂直分割" })
map("n", "<leader>\\", ":sp<CR>", { desc = "水平分割" })
map("n", "<A-h>", "<C-w>h", { desc="Window left" })
map("n", "<A-l>", "<C-w>l", { desc="Window right" })
map("n", "<A-k>", "<C-w>k", { desc="Window up" })
map("n", "<A-j>", "<C-w>j", { desc="Window down" })