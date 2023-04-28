namespace QueryKits.Extensions;

public interface IEventAggregator
{
    /// <summary>
    /// Subscribes an instance to all events declared through implementations of <see cref = "IHandle{T}" />
    /// </summary>
    /// <param name = "subscriber">The instance to subscribe for event publication.</param>
    void Subscribe<T>(IHandle<T> subscriber);

    /// <summary>
    /// Unsubscribes the instance from all events.
    /// </summary>
    /// <param name = "subscriber">The instance to unsubscribe.</param>
    void Unsubscribe<T>(IHandle<T> subscriber);

    /// <summary>
    /// Publishes a message.
    /// </summary>
    /// <param name = "message">The message instance.</param>
    /// <returns>A task that represents the asynchronous operation.</returns>
    Task PublishAsync(object message);
}