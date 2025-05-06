param(
    [string]$action,
    [string]$arg1,
    [string]$arg2
)

# $env:psm1Home = "D:\OneDriveCode\psm1"
Import-Module $env:psm1Home\common.psm1 -Force

$MyScriptFullname = $MyInvocation.MyCommand.Path
$MyScriptPath = Split-Path $MyScriptFullname -Parent

function UndoAllUncommittedOrUnsavedChanges {
    Write-Host "Undo all uncommitted or unsaved changes"
    InvoleCmd "git checkout -- ."
}

function GitInit {
    $currentPath = Get-Location
    Copy-Item -Path $MyScriptPath\.gitignore-template -Destination $currentPath\.gitignore
    InvokeCmd "git init"
}

function MergeToBranch {
    param(
        [string]$branch
    )
    Info "取得遠端資料並更新本地 master 程式碼"
    & git fetch origin master
    & git checkout master
    & git merge origin/master
    Info "將 master 的程式碼合併至新分支中"
    & git checkout $branch
    & git merge master
}

if( "init" -eq $action ) 
{
    GitInit
    return
}

if( "mtb" -eq $action )
{
    $branch = $arg1
    MergeToBranch $branch 
    return
}

if( "frf" -eq $action)
{
    $folder = $arg1
    $answer = Read-Host "強迫刪除 git repo $folder 整個資料夾, 包含歷史紀錄 (yes/no)?"
    if ($answer -ne "yes") {
        return
    }
    Info "強迫刪除整個 $folder 資料夾包含歷史紀錄..."
    InvokeCmd "git filter-branch --force --index-filter 'git rm --cached -r --ignore-unmatch $folder/' --prune-empty --tag-name-filter cat -- --all"
    InvokeCmd "git push origin master --force --tags"

    Remove-Item -Recurse -Force .\.git\refs\original\
    InvokeCmd "git reflog expire --expire=now --all"
    InvokeCmd "git gc --prune=now"
    InvokeCmd "git gc --aggressive --prune=now"
    return
}


if( "r" -eq $action ) {
    Info "復原上一個提交"
    InvokeCmd "git reset --hard HEAD~1"
    return
}

if( "rd" -eq $action ) {
    Info "移除已加入的資料夾"
    $folder = $arg1
    if( "" -eq $folder ) {
        ShowError "gt rd <folder>"
        return
    }
    InvokeCmd "git rm -r --cached $folder"
    # InvokeCmd "Removed Folder $folder"
    return
}

if( "u" -eq $action ) {
    $specificFile = $arg1
    if( "" -ne $specificFile ) {
        InvokeCmd "git checkout -- $specificFile"
        InvokeCmd "git clean -f"
        return
    }
    UndoAllUncommittedOrUnsavedChanges
    git status
    return
}

if( "uf" -eq $action ){
    $hashId = $arg1
    if( "" -eq $hashId ) {
        Write-Host "Please input hash id"
        return
    }
    $file = $arg2
    if( "" -eq $file ) {
        Write-Host "Please input file name"
        return
    }
    Write-Host "針對指定 commit $hashId 的特定檔案進行 revert"
    InvokeCmd "git checkout $hashId^ -- $file"
    InvokeCmd "git status"
    InvokeCmd "git commit -m ""Revert $file to $hashId version"""
    return
}

if( "uc" -eq $action ) {
    InvokeCmd "git reset HEAD~1"
    InvokeCmd "git status"
    return
}

if( "m" -eq $action ) {
    $description = $arg1
    InvokeCmd "git commit -m '$description'"
    return
}

if( "h" -eq $action ) {
    $hash = $arg1
    InvokeCmd "git show -s --format='fuller' $hash"
    return
}

if( "l" -eq $action ) {
    # 簡短 log
    InvokeCmd 'git log --pretty=format:"%h - %an, %cd : %s"'
    return
}

if( "lf" -eq $action ){
    Write-Host "只列出該 commit 中變更的檔案名稱，而不顯示內容變更細節"
    $hashId = $arg1
    InvokeCmd "git diff-tree --no-commit-id --name-only -r $hashId"
    return
}

if( "ll" -eq $action ) {
    # 長 log
    # InvokeCmd "git log -s --format='fuller'"
    InvokeCmd "git log -p --format='fuller' --graph"
    return
}

if( "lc" -eq $action )
{
    # 列出尚未 push 的 commit
    $name = git branch --show-current
    InvokeCmd "git log --oneline origin/$name..HEAD"
    return
}

if( "c" -eq $action ) {
    $branch = $arg1
    if( $branch -eq "" ) {
        $branch = "master"
    }
    InvokeCmd "git checkout $branch"
    return
}

if( "df" -eq $action )
{
    # 取得所有修改但尚未 commit 的檔案
    $changes = git diff --name-status

    foreach ($line in $changes) {
        $status, $path = $line -split "`t"

        switch ($status) {
            'M' {
                Write-Host " M " -BackgroundColor Yellow -ForegroundColor Black -NoNewline
                Write-Host " $path"

                # 匯出暫存區版本（若無，會失敗）
                $stagedPath = "$env:TEMP\__staged__.txt"
                git show ":$path" 2>$null > $stagedPath

                # 匯出當前工作目錄版本
                $workingPath = "$env:TEMP\__working__.txt"
                Get-Content $path > $workingPath

                # 使用 side-by-side diff
                diff -y --width=160 --suppress-common-lines $stagedPath $workingPath

                Remove-Item $stagedPath, $workingPath
            }
            'A' {
                Write-Host " A " -BackgroundColor Green -ForegroundColor White -NoNewline
                Write-Host " $path"
            }
            'D' {
                Write-Host " D " -BackgroundColor Red -ForegroundColor White -NoNewline
                Write-Host " $path"
            }
            default {
                Write-Host " $status " -BackgroundColor DarkGray -ForegroundColor White -NoNewline
                Write-Host " $path"
            }
        }
    }

    return
}

if( "dff" -eq $action )
{
    InvokeCmd "git difftool --tool=vimdiff -y"
    return
}

if( "pl" -eq $action ) {
    InvokeCmd "git pull"
    InvokeCmd "git submodule sync"
    InvokeCmd "git submodule update --init --remote --recursive"
    return
}

if( "p" -eq $action ) {
    invokecmd "git push"
    return
}

if( "a" -eq $action ){
    $fileOrFolder = $arg1
    if( "" -eq $fileOrFolder ) {
        $fileOrFolder = "."
    }
    InvokeCmd "git add $fileOrFolder"
    InvokeCmd "git status"
    return
}

if( "am" -eq $action ) {
    $description = $arg1

    if( "" -eq $description) {
        WriteHostColor "gt am ", "'comment'" -Color White, Red
        Write-Host "please add comment description"
        return
    }

    InvokeCmd "git add ."
    InvokeCmd "git commit -m '$description'"
    InvokeCmd "git push"
    return
}

if( "lb" -eq $action ) {
    InvokeCmd "git branch -a"
    return
}

if( "b" -eq $action ) {
    $branchName = $arg1
    InvokeCmd "git branch $branchName"
    return
}

if( "st" -eq $action ) {
    $comment = $arg1
    InvokeCmd "git stash save -u '$comment'"
    return
}

if( "stp" -eq $action ) {
    InvokeCmd "git stash pop"
    return
}

if( "stc" -eq $action ) {
    InvokeCmd "git stash clear"
    return
}

if( "stl" -eq $action ) {
    Info "列出 stash 清單"
    InvokeCmd "git stash list"
    return
}

if( "s" -eq $action ) {
    $hash = $arg1
    if( $hash -ne "" ) {
        InvokeCmd "git show $hash"
        return
    }
    InvokeCmd "git status"
    return
}

if( "info" -eq $action )
{
    Write-Host "顯示目前專案的Git倉庫所佔用的檔案空間"
    Write-Host "'size'欄位加上'size-pack'欄位所顯示的值就是 Git 倉庫所佔用的檔案空間"
    InvokeCmd "git count-objects -vH"
    return
}

if( "apply-gitignore" -eq $action )
{
    # 修改完 .gitignore 檔案內容, 並重新套用
    InvokeCmd "git rm -r --cached ."
    # 再重新將專案根目錄加進版本控制清單中
    InvokeCmd "git add ."
    InvokeCmd "git commit -m 'update .gitignore'"
    # 手動將將已經被commit進Git倉庫的檔案刪除
    # git filter-branch --force --tree-filter 'rm -f -r "檔案路徑"' -- --all
    # 立刻刷新檔案列表
    InvokeCmd "git filter-branch --force"
    # 覆蓋遠端 git 倉庫
    InvokeCmd "git push --force --all"
    return
}

if( "freeup" -eq $action )
{
    # 清除Git的Reflog並手動調用垃圾回收機制, 讓Git倉庫將可用空間釋放出來
    # 如果並非迫切地需要釋出空間, 建議不要執行這個指令
    InvokeCmd "git reflog expire --expire=now --expire-unreachable=now --all"
    InvokeCmd "git gc --prune=all --aggressive"
    return
}

if( "rm" -eq $action )
{
    $file = $arg1
    Write-Host "Please input '$file' in .gitignore file"
    InvokeCmd "git rm $file"
    return
}

if( "cl" -eq $action )
{
    Info "刪除當前目錄下沒有被track過的檔案和資料夾"
    InvokeCmd "git clean -df"
    return
}

if( "sr" -eq $action )
{
    # git for-each-ref --sort=-committerdate refs/heads/
    InvokeCmd "git for-each-ref --sort=-committerdate"
    return
}

if( "plm" -eq $action )
{
    InvokeCmd "git submodule foreach git pull"
    return
}

if( "ab" -eq $action )
{
    InvokeCmd "git restore ."
    # InvokeCmd "git reset --hard HEAD"  # 清除 unstage files
    # InvokeCmd "git clean -fdx"  # 清除 untrack files
    return
}

Write-Host ""
WriteHostColor "git helper by flash" "flash"
Write-Host ""
Write-Host "a <file or folder> :add unstage files"
Write-Host "ab                 :abort all unstage files"
Write-Host "am 'comment desc'  :add unstage files and commit"
Write-Host "b [branch name]    :create branch or show current branch when no [branch name]"
Write-Host "c <branch name>    :checkout branch, if branch name is empty, checkout master"
Write-Host "cl                 :clean untrack files"
Write-Host "df                 :diff 上下比對"
Write-Host "dff                :diff horizontal 左右比對"
Write-Host "frf <folder>       :force remove folder (include history) in git repo"
Write-Host "h <hash>           :show hash info"
Write-Host "init               :init and add default .gitignore"
Write-Host "info               :顯示目前專案的 Git 倉庫所佔用的檔案空間"
Write-Host "l                  :show short log"
Write-Host "ll                 :show long log"
Write-Host "lf <hash>          :show hash changed files 只列出該 commit 中變更的檔案名稱，而不顯示內容變更細節"
Write-Host "lc                 :list unpushed commit 列出尚未 push 的 commit"
Write-Host "r                  :undo previous action"
Write-Host "rm <file>          :remove file in commited file"
Write-Host "rd <folder>        :remove add folder in git"
Write-Host "p                  :push"
Write-Host "pl                 :pull"
Write-Host "lb                 :list all branches"
Write-Host "m 'comment desc'   :add files and commit"
Write-Host "mtb <branch name>  :merge into branch"
Write-Host "s                  :show current status" 
Write-Host "s <hash>           :show hash changed files" 
Write-Host "sr                 :show remote branch names order by commited date desc" 
Write-Host "st [comment]       :stash save" 
Write-Host "stl                :stash list"
Write-Host "stp                :stash pop"
Write-Host "stc                :stash clear"               
Write-Host "u [file]           :undo uncommitted files and clean"
Write-Host "uc                 :abort commit files"
Write-Host "uf <hashId> <file> :revert commit file for hashId (針對指定 commit 的特定檔案進行 revert)"
Write-Host "plm                :pull all submodules"