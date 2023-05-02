$token = "YOUR_PRIVATE_TOKEN"
$token = Get-Content "d:\demo\gitlab.token"

$url = "https://gitlab.com/api/v4/projects?private_token=$token"
$projects = Invoke-RestMethod -Uri $url

foreach ($project in $projects) {
    Write-Host $project.name
    Write-Host $project.ssh_url_to_repo
}