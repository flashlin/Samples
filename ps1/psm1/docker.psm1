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

function RestartContainer {
   param(
      [string]$name,
      [string]$startArguments
   )
   $isExists = InvokeDocker "ps -q --filter name=$name"
   if( $isExists -eq $true ) {
      return
   }
   $id = InvokeDocker "ps -aq --filter name=$name"
   if( $null -ne $id ) {
      InvokeDocker "start $id"
      return
   }
   InvokeDocker "run -it --name $name $startArguments" 
}

Export-ModuleMember -Function *