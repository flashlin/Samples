function Last {
    param(
        [Parameter(Mandatory,ValueFromPipeline)]
        [Object]
        $inputData
    )
    $inputData | Select-Object -Last 0
}

function Select-Zip {
    [CmdletBinding()]
    Param(
        $First,
        $Second,
        $ResultSelector = { ,$args }
    )
    [System.Linq.Enumerable]::Zip($First, $Second, [Func[Object, Object, Object[]]]$ResultSelector)
}

function SubStr {
    param (
        [string]
        $text,
        [int]
        $index,
        [int]
        $length
    )
    if( $index -lt 0 ) {
        ""
        return
    }
    if( $index -ge $text.Length ) {
        ""
        return
    }
    $len = $text.Length - $index
    $len = [math]::min($len, $length)
    $text.Substring($index, $len)
}

function ConsoleStringToString {
    param(
        [string]$consoleText
    )
    return $consoleText -replace '[\x00-\x1F\x7F]', ''
}

function SplitSpaces {
    param(
        [Parameter(Mandatory,ValueFromPipeline)]
        [string]
        $inputData
    )
    begin{
    }
    process{
        $text = ConsoleStringToString $inputData
        $match = [regex]::Matches($text, "[^ ]+[ ]*")
        if( $match.Success -eq $false ) {
            return
        }
        for($i=0; $i -lt $match.Count; $i++) {
            $item = $match.Captures[$i]
            # Write-Host "$i '$($item)' $($item.Index) $($item.Length) '$($text.Substring($item.Index, $item.Length))'"
            [PSCustomObject]@{
                Index = $item.Index 
                Length = $item.Length
                Value = $item.Value 
            }
        }
    }
    end {
    }
}

function CaptureString {
    param(
        [Parameter(Mandatory,ValueFromPipeline)]
        [Object]
        $inputData,
        [Parameter(Mandatory)]
        [Object]
        $columns
    )
    process{
        if( $inputData -eq "" ) {
            return
        }
        foreach($col in $columns) {
            #Write-Host "$($col.Index) $col"
            if( $col -eq $columns | Last ) {
                $value = $inputData.Substring($col.Index)
                $value
                continue
            }
            $value = SubStr $inputData $col.Index $col.Length
            $value
        }
    }
}


function SplitTableString {
   param(
       [Parameter(Mandatory,ValueFromPipeline)]
       [Object]
       $inputData
   )
   begin{
       $first = $true
       $columns = @()
   }
   process{
       if( $first -eq $true) {
           $columns = $inputData | SplitSpaces
           $first = $false
           return
       }

       $text = ConsoleStringToString $inputData
       $values = $text | CaptureString -columns $columns
       if( $null -eq $values) {
           return
       }
       $items = Select-Zip -First $columns -Second $values `
           -ResultSelector { 
               param($a, $b) 
               @{
                   column=$a 
                   value=$b
               } 
           }
       $valueObj = New-Object PSCustomObject
       foreach($item in $items) {
           AddObjectProperty $valueObj $item.column.value.Trim() $item.value
       }
       $valueObj
   }
   end {
   }
}

Export-ModuleMember -Function *