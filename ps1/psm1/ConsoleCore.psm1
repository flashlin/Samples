$dlls = @( 
   #"$env:psm1HOME\ConsoleUtils.dll" 
   "$env:psm1HOME\ConsoleCore.dll"
)
Add-Type -LiteralPath $dlls

function Create_IEnumerable($arr) {
   [System.Collections.ArrayList]$list = @()
   $list.AddRange($arr)
   $list
}

Export-ModuleMember -Function *