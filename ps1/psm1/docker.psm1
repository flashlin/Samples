function InvokeDocker {
   param(
      [string]$param
   )
   $cmd = "$env:docker_exe " + $param
   InvokeCmd $cmd
}

function QueryDockerImages {
   param(
      [string]$keyword
   )
   $result = InvokeDocker "images -a --format ""{ 'ID':'{{.ID}}', 'Repo':'{{.Repository}}', 'Tag':'{{.Tag}}', 'Size':'{{.Size}}' }""" # --quiet
   $result | ForEach-Object {
      $json = $_ -replace "'", '"'
      $item = ConvertFrom-Json $json
      if ( $item.Repo -Match $keyword ) {
         return $item
      }
   }
}

function IsContainerExists {
   param(
      [string]$name
   )
   $isExists = InvokeDocker "ps -q --filter name=$name"
   if( $isExists -eq $true ) {
      return $true
   }
   return $false
}

function GetContainerExited {
   param(
      [string]$name
   )
   $id = InvokeDocker "ps -aq --filter name=$name"
   return $id
}

function RestartContainer {
   param(
      [string]$name,
      [string]$startArguments
   )
   $isExists = IsContainerExists $name
   if( $isExists -eq $true ) {
      return
   }
   $id = GetContainerExited $name
   if( $null -ne $id ) {
      InvokeDocker "start $id"
      return
   }
   InvokeDocker "run -it --name $name $startArguments" 
}

function RemoveContainer {
   param(
      [string]$name
   )
   if( $true -eq (IsContainerExists $name) ) {
      InvokeDocker "stop $name"
   }
   $id = GetContainerExited $name
   InvokeDocker "rm $id"  
}

Export-ModuleMember -Function *