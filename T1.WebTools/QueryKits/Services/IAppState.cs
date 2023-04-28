namespace QueryKits.Services;

public interface IAppState
{
    bool IsLoading { get; set; }
    void Publish(Action<AppState> changeFn);
    public void SubscribeEvent<T>(Func<T, Task> handler)
        where T: EventArgs;
    void PublishEvent<T>(T eventArgs)
        where T: EventArgs, new();
}