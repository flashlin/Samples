using System.Reactive.Disposables;
using T1.Standard.Net.SoapProtocols.WsdlXmlDeclrs;

namespace GitCli.Models.ConsoleMixedReality;


public class ConsoleElementEvent
{
    public InputEvent InputEvent { get; init; }
    public IConsoleElement Element { get; init; }
}

public class ConsoleInputObserver : IObservable<ConsoleElementEvent>
{
    private IObserver<ConsoleElementEvent>? _observer;
    private Task? _task;
    private readonly CancellationTokenSource _cancellationTokenSource = new();

    public IDisposable Subscribe(IObserver<ConsoleElementEvent> observer)
    {
        _observer = observer;
        return Disposable.Empty;
    }

    public CancellationTokenSource Cancellation => _cancellationTokenSource;

    private void RaiseKey(IConsoleElement element, InputEvent inputEvent)
    {
        _observer?.OnNext(new ConsoleElementEvent
        {
            InputEvent = inputEvent,
            Element = element,
        });
    }
}
