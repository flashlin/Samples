# T1.SlackSdk

An easy-to-use and comprehensive API for writing Slack apps in .NET.

## Features

- **Easy Integration**: Simplified wrapper around SlackNet for easier Slack app development
- **Progress Messages**: Built-in support for sending progress messages to Slack channels
- **History Management**: Easy retrieval of Slack message history with thread support
- **User Management**: Simplified user information handling with caching
- **Configuration**: Streamlined configuration management using Microsoft.Extensions.Options

## Installation

```bash
dotnet add package T1.SlackSdk
```

## Quick Start

### Basic Setup

```csharp
// Configure Slack client using dependency injection
services.Configure<SlackConfig>(options =>
{
    options.Token = "your-bot-token";
});

services.AddScoped<ISlackClient, SlackClient>();

// Or create directly
var slackConfig = new SlackConfig
{
    Token = "your-bot-token"
};

var slackClient = new SlackClient(Options.Create(slackConfig));
```

### Send Progress Messages

```csharp
// Send initial progress message
var progressArgs = new SendProgressMessageArgs
{
    ChannelId = "channel-id",
    ProgressMessage = "🔄 Task processing started...",
    Username = "Bot",
    Ts = "thread-timestamp" // Optional: for thread replies
};
await slackClient.SendProgressMessageAsync(progressArgs);

// Send finish message
var finishArgs = new SendFinishProgressMessageArgs
{
    ChannelId = "channel-id",
    Ts = "original-message-timestamp",
    FinishMessage = "✅ Processing completed!",
    Message = "Result: All tasks completed successfully"
};
await slackClient.SendFinishProgressMessageAsync(finishArgs);
```

### Get Message History

```csharp
var historyArgs = new GetHistoryArgs
{
    ChannelId = "channel-id",
    Range = new DateTimeRange
    {
        Start = DateTime.Now.AddDays(-7),
        End = DateTime.Now
    }
};

var history = await slackClient.GetHistoryAsync(historyArgs);
foreach (var item in history)
{
    Console.WriteLine($"User: {item.User.Name}, Message: {item.Text}, Time: {item.Time}");
    
    // Access thread messages
    foreach (var threadMsg in item.ThreadMessages)
    {
        Console.WriteLine($"  Thread: {threadMsg.Text}");
    }
}
```

### Get User Information

```csharp
var user = await slackClient.GetUserInfoAsync("user-id");
Console.WriteLine($"User: {user.Name}, Email: {user.Email}");
```

## Configuration

The `SlackConfig` class supports the following properties:

- `Token`: Your Slack bot token (required)

## Dependencies

- .NET 8.0
- SlackNet 0.13.3
- Microsoft.Extensions.Options 8.0.2
- T1.Standard 1.0.81

## Features Details

### Message History
- Retrieves messages with full thread support
- Includes user information for each message
- Supports date range filtering
- Automatic caching for user information (10-minute cache)

### Progress Messages
- Send initial progress messages with custom text
- Update messages to show completion status
- Support for threaded conversations
- Customizable usernames and messages

### User Management
- Cached user information retrieval
- Automatic cache invalidation
- Efficient API usage with method caching

## License

MIT License - see [LICENSE](https://choosealicense.com/licenses/mit/) for details.

## Contributing

Feel free to submit issues and enhancement requests!

