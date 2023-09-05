local status, nvim_lsp = pcall(require, 'lsconfig')
if (not status) then return end
local protocol = require('vim.lsp.protocol')
local on_attach = function(client, bufnr)
    if client.server_capabilities.documentFromattingProvider then
        vim.api.nvim_command [[augroup Format]]
        vim.api.nvim_command [[autocmd! * <buffer>]]
        vim.api.nvim_command [[autocmd BufWritePRe <buffer> lua vim.lsp.buf.formtting_se_sync()]]
        vim.api.nvim_command [[augroup END]]
    end
end    

nvim_lsp.tsserver.setup {
    on_attach = on_attach,
    filetypes = { "typescript", "typescriptreact", "typescript.tsx" },
    cmd = { "typescript-language-server", "--stdio" }
}

nvim_lsp.sumneko_lua.setup {
    on_attach = on_attach,
    settings = {
        Lua = {
            diagnostics = {
                globals = { 'vim' }
            },
            workspace = {
                library = vim.api.nvim_get_runtime_file("", true)
            }
        }
    }
}