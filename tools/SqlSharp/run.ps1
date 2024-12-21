Remove-Item \VDisk\Github\qa-pair\qa-files\synthetic\outputs\*.*

$output_path = "\VDisk\Github\qa-pair\qa-files\synthetic\outputs"

# 建立資料庫
# .\SqlSharp\bin\Debug\net8.0\SqlSharp.exe -v extractCreateTableSql -i D:\VDisk\MyGitHub\SQL -o \VDisk\Github\qa-pair\qa-files\synthetic\outputs
.\SqlSharp\bin\Debug\net8.0\SqlSharp.exe -v extractCreateTableSql -i D:\VDisk\MyGitHub\SQL -o $output_path
.\SqlSharp\bin\Debug\net8.0\SqlSharp.exe -v extractCreateTableSql -i D:\coredev_tw\DbProjects -o $output_path


#.\SqlSharp\bin\Debug\net8.0\SqlSharp.exe -v extractSelectSql -i D:\VDisk\MyGitHub\SQL -o $output_path
#.\SqlSharp\bin\Debug\net8.0\SqlSharp.exe -v extractSelectSql -i D:\coredev_tw\DbProjects -o $output_path