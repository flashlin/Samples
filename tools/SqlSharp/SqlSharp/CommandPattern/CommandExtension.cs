namespace SqlSharp.CommandPattern;

public static class CommandExtension
{
    public static async Task SafeExecuteAsync<TArgs>(this ICommand<TArgs>? command, TArgs args)
    {
        if (command != null)
        {
            await command.ExecuteAsync(args);
        }
    }
}