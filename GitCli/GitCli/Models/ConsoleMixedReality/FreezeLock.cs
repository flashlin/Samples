namespace GitCli.Models.ConsoleMixedReality;

public struct FreezeLock
{
    private int _freezeCount;

    public void Freeze()
    {
        _freezeCount++;
    }

    public void Unfreeze()
    {
        _freezeCount--;
    }

    public bool IsFrozen => _freezeCount > 0;
    public bool IsUnfrozen => !IsFrozen;
}