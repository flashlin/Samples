-- utf8
vim.scriptencoding = 'utf-8'
vim.opt.encoding = 'utf-8'
vim.opt.fileencoding = 'utf-8'
vim.g.encoding = "UTF-8"
vim.o.fileencoding = 'utf-8'
-- jk移動時光標下上方保留8行
vim.o.scrolloff = 8
vim.o.sidescrolloff = 8
-- 使用相對行號
vim.wo.number = true
vim.wo.relativenumber = true
vim.opt.title = true
-- 高亮所在行
vim.opt.cursorline = true
vim.opt.termguicolors = true
vim.opt.winblend = 0
vim.opt.wildoptions = 'pum'
vim.opt.pumblend = 5
vim.opt.background = 'dark'
-- 显示左側圖標指示列
vim.wo.signcolumn = "yes"
-- 右側參考线，超過表示代碼太長了，考慮換行
vim.wo.colorcolumn = "80"
-- 缩进2個空格等于一個Tab
vim.o.tabstop = 2
vim.bo.tabstop = 2
vim.o.softtabstop = 2
vim.o.shiftround = true
-- >> << 時移動長度
vim.o.shiftwidth = 2
vim.bo.shiftwidth = 2
-- 新行對齊當前行，空格替代tab
vim.o.expandtab = true
vim.bo.expandtab = true
vim.o.autoindent = true
vim.bo.autoindent = true
vim.o.smartindent = true
-- 搜索大小寫不敏感，除非包含大寫
vim.o.ignorecase = true
vim.o.smartcase = true
-- 搜索不要高亮
vim.o.hlsearch = true
-- 邊输入邊搜索
vim.o.incsearch = true
-- 使用增强狀态栏后不再需要 vim 的模式提示
vim.o.showmode = false
-- 命令行高为2，提供足够的显示空间
vim.o.cmdheight = 2
-- 當文件被外部程序修改時，自動加载
vim.o.autoread = true
vim.bo.autoread = true
-- 禁止折行
vim.o.wrap = false
vim.wo.wrap = false
-- 行结尾可以跳到下一行
vim.o.whichwrap = 'b,s,<,>,[,],h,l'
-- 允许隐藏被修改過的buffer
vim.o.hidden = true
-- 鼠標支持
vim.o.mouse = "a"
-- 禁止创建备份文件
vim.o.backup = false
vim.o.writebackup = false
vim.o.swapfile = false
-- smaller updatetime 
vim.o.updatetime = 300
-- 等待mappings
vim.o.timeoutlen = 100
-- split window 从下邊和右邊出现
vim.o.splitbelow = true
vim.o.splitright = true
-- 自動补全不自動选中
vim.g.completeopt = "menu,menuone,noselect,noinsert"
-- 样式
vim.o.background = "dark"
vim.o.termguicolors = true
vim.opt.termguicolors = true
-- 不可见字符的显示，这里只把空格显示为一個点
vim.o.list = true
vim.o.listchars = "space:·"
-- 补全增强
vim.o.wildmenu = true
-- Dont' pass messages to |ins-completin menu|
vim.o.shortmess = vim.o.shortmess .. 'c'
vim.o.pumheight = 10
-- always show tabline
vim.o.showtabline = 2
vim.opt.path:append { '**' } --Finding files: search down into subfolders
vim.opt.wildignore:append { '*/node_modules/* '}