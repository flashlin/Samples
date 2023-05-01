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

Export-ModuleMember -Function *