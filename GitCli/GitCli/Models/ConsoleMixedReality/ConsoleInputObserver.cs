using System.Reactive.Disposables;
using T1.Standard.Net.SoapProtocols.WsdlXmlDeclrs;

namespace GitCli.Models.ConsoleMixedReality;

public class ConsoleInputObserver : IObservable<ConsoleElementEvent>
{
    private IObserver<ConsoleElementEvent>? _observer;

    public IDisposable Subscribe(IObserver<ConsoleElementEvent> observer)
    {
        _observer = observer;
        return Disposable.Empty;
    }

    public CancellationTokenSource Cancellation { get; } = new();

    private void RaiseKey(IConsoleElement element, InputEvent inputEvent)
    {
        _observer?.OnNext(new ConsoleElementEvent
        {
            InputEvent = inputEvent,
            Element = element,
        });
    }
}
