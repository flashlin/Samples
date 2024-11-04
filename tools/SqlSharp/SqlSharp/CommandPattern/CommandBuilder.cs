namespace SqlSharp.CommandPattern;

public class CommandBuilder<TArgs>
{
    private ICommand<TArgs>? _currentHandler;
    private ICommand<TArgs>? _firstHandler;

    public CommandBuilder<TArgs> Use(ICommand<TArgs> handler)
    {
        if (_currentHandler == null)
        {
            _currentHandler = handler;
            _firstHandler = handler; 
        }
        else
        {
            _currentHandler.Next = handler;
            _currentHandler = handler;
        }
        return this;
    }

    public ICommand<TArgs> Build()
    {
        return _firstHandler ?? throw new InvalidOperationException("No handler was added");
    }
}