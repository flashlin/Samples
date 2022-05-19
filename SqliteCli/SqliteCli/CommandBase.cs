namespace SqliteCli;

public abstract class CommandBase
{
    public abstract bool IsMyCommand(string[] args);
    public abstract void Run(string[] args);
}