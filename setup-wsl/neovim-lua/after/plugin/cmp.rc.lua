local status, cmp = pcall(require, 'cmp')
if (not status) then return end

local snip_status, luasnip = pcall(require, 'luasnip')
if (not snip_status) then return end

require('luasnip/loaders/from_vscode').lazy_load() 

local lspkind = require 'lspkind'

cmp.setup({
   snippet = {
      expand = function(args)
         require('luasnip').lsp_expand(args.body)
      end
   },
   mapping = cmp.mapping.preset.insert({
      ['<C-d>'] = cmp.mapping.scroll_docs(-4),
      ['<C-f>'] = cmp.mapping.scroll_docs(4),
      ['<C-j>'] = cmp.mapping.complete(),
      ['<C-e>'] = cmp.mapping.close(),
      ['<CR>'] = cmp.mapping.confirm({
         behavior = cmp.ConfirmBehavior.Replace,
         select = true
      }),
   }),
   sources = cmp.config.sources({
      { name = 'nvim_lsp' },
      { name = 'buffer' },
   }),
   formatting = {
      format = lspkind.cmp_format({ wirth_text = false, maxwidth = 50})
   }
})

vim.cmd [[
   set completeopt=menuone,noinsert,noselect
   highlight! default link CmpItemKind CmpItemMenuDefault
]]