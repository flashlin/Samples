namespace SqlSharp.CommandPattern;

public interface ICommand<TArgs>
{
    ICommand<TArgs>? Next { get; set; } 
    Task ExecuteAsync(TArgs args);
}