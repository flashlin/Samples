namespace QueryKits.Services;

public interface IAppState
{
    bool IsLoading { get; set; }
    void Publish(Action<AppState> changeFn);
    void SubscribeEvent<T>(Action<T> handler)
        where T: EventArgs;
    void PublishEvent<T>(T eventArgs)
        where T: EventArgs;
}