local M = {}

M.jump_to_next_ts_method1 = function()
    local ts_utils = require'nvim-treesitter.ts_utils'
    local parser = vim.treesitter.get_parser(0, "vue")
    local tree = parser:parse()[1]
    local query = vim.treesitter.query.parse("vue", [[
        (script_element
            (raw_text) @script
            (#match? @script "^<script.*lang=\"ts\"")
            (raw_text
                (method_definition) @method
            )
        )
    ]])

    local cursor = vim.api.nvim_win_get_cursor(0)
    local current_line = cursor[1]

    for id, node in query:iter_captures(tree:root(), 0) do
        local name = query.captures[id]
        if name == "method" then
            local start_row, _, _, _ = node:range()
            if start_row + 1 > current_line then
                ts_utils.goto_node(node)
                return
            end
        end
    end

    print("No more TypeScript methods found")
end

M.jump_to_next_ts_method = function()
    local lsp = require("lspconfig")
    local client = lsp.get_active_client()
    if client then
        local current_method = client:get_method()
        local next_method = client:get_next_method()
        if next_method then
            vim.api.nvim_win_set_cursor(0, { line = next_method.line, col = next_method.col })
        end
    end
end

return M