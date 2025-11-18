param(
  [Parameter(Mandatory=$true)][string]$Entry,
  [string[]]$Args
)
$env:PYTHONTRACEMALLOC="25"
python $Entry @Args

