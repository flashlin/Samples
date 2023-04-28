using QueryKits.Extensions;

namespace QueryKits.Services;

public interface IAppState
{
    bool IsLoading { get; set; }
    Task PublishAsync(Action<AppState> changeFn);
    public void SubscribeEvent<T>(IHandle<T> handler)
        where T: EventArgs;
    Task PublishEventAsync<T>(T eventArgs)
        where T: EventArgs;
}