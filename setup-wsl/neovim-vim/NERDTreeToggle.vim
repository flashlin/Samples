" 不顯示隐藏文件
let g:NERDTreeHidden=0
" 過濾: 所有指定文件和資料夾不顯示
let NERDTreeIgnore = ['\.pyc$', '\.swp', '\.swo', '\.vscode', '\node_modules', '__pycache__'] 

" F3 開啟關閉
nnoremap <F3> :NERDTreeToggle<CR>

"當 NERDTree 為剩下的唯一窗口時自動關閉
autocmd bufenter * if (winnr("$") == 1 && exists("b:NERDTree") && b:NERDTree.isTabTree()) | q | endif

"設定樹的顯示Icon
let g:NERDTreeDirArrowExpandable = '▸'
let g:NERDTreeDirArrowCollapsible = '▾'
