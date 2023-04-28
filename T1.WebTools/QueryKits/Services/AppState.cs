using QueryKits.Extensions;

namespace QueryKits.Services;

public class AppState : IAppState
{
    private readonly IEventAggregator _eventAggregator;

    public AppState(IEventAggregator eventAggregator)
    { 
        _eventAggregator = eventAggregator;
    }

    public bool IsLoading { get; set; }

    public async Task PublishAsync(Action<AppState> changeFn)
    {
        changeFn(this);
        await _eventAggregator.PublishAsync(new UpdateAppReqEvent());
    }
    
    public Task PublishEventAsync<T>(T eventArgs)
        where T: EventArgs
    {
        return _eventAggregator.PublishAsync(eventArgs);
    }

    public void SubscribeEvent<T>(IHandle<T> handler)
        where T: EventArgs
    {
        _eventAggregator.Subscribe(handler);
    }
}