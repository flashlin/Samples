function ProcessAllMatchs {
    param(
       [string]$text,
       [Microsoft.PowerShell.Commands.MatchInfo]$allMatches
    )
    $result = @()
    $currentIndex = 0
    foreach ($match in $allMatches.Matches) {
       # 輸出未匹配的部分
       if ($match.Index -gt $currentIndex) {
          $unmatchedText = $text.Substring($currentIndex, $match.Index - $currentIndex)
          $result += @{
             IsMatch = $False
             Text = $unmatchedText
          }
       }
       $result += @{
          IsMatch = $True
          Text = $match.Value
       }
       $currentIndex = $match.Index + $match.Length
    }
    # 輸出最後一個未匹配的部分
    if ($currentIndex -lt $text.Length) {
       $unmatchedText = $text.Substring($currentIndex)
       $result += @{
          IsMatch = $False
          Text = $unmatchedText
       }
    }
    return $result
}


function MatchTextPattern {
   param(
       [string]$text,
       [string]$pattern
   )
   $allMatches = Select-String -InputObject $text -Pattern $pattern -AllMatches
   if( $null -eq $allMatches ) {
       return @(
           @{ 
               IsMatch = $False
               Text = $text 
           }
       )
   }
   return ProcessAllMatchs $text $allMatches
}


function MatchTextFragment {
    param(
        $matchFragments,
        [string]$pattern
    )
    $result = @()
    for($i=0; $i -lt $matchFragments.Length; $i++) {
        $matchFragment = $matchFragments[$i]
        if( $matchFragment.IsMatch ) {
            $result += $matchFragment
            continue
        }
        $subResult = MatchText $matchFragment.Text $pattern
        $result += $subResult
    }
    return $result
}

<#
.PARAMETER text
The input text to be searched for matching patterns.
#>
function MatchText {
   param(
       [string]$text,
       [string[]]$patterns
   )
   $matchFragments = @()
   $first = $true
   foreach ($pattern in $patterns) {
       if( $first ) {
           $matchFragments = MatchTextPattern $text $pattern
           $first = $False
       } else {
           $matchFragments = MatchTextFragment $matchFragments $pattern
       }
   }
   return $matchFragments
}

Export-ModuleMember -Function *