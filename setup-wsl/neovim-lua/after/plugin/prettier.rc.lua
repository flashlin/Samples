local status, prettier = pcall(require, 'prettier')
if (not status) then return end

prettier.setup {
    bin = 'prettierd',
    filetypes = {
        'css',
        'javascript',
        'javascriptrect',
        'typescript',
        'typescriptreact',
        'json',
        'scss',
        'less'
    }
}