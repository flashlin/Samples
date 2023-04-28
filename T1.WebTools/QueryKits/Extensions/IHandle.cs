using T1.Standard.Extensions;

namespace QueryKits.Extensions;

public interface IHandle<in TMessage>
{
    /// <summary>
    /// Handles the message.
    /// </summary>
    /// <param name = "message">The message.</param>
    /// <returns>A task that represents the operation.</returns>
    Task HandleAsync(TMessage message);
}