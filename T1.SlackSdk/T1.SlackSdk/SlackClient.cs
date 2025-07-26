using Castle.Core.Internal;
using Microsoft.Extensions.Options;
using SlackNet;
using SlackNet.WebApi;
using T1.Standard.Common;
using T1.Standard.Extensions;

namespace T1.SlackSdk;

public class SlackClient : ISlackClient
{
    private readonly SlackApiClient _client;
    private readonly MethodCache _methodCache = new(TimeSpan.FromMinutes(10));

    public SlackClient(IOptions<SlackConfig> config)
    {
        _client = new SlackApiClient(config.Value.Token);
    }

    public Task<List<SlackHistoryItem>> GetHistoryAsync(GetHistoryArgs args)
    {
        return InternalGetHistoryAsync(args.ChannelId, args.Range);
    }

    public async Task SendProgressMessageAsync(SendProgressMessageArgs args)
    {
        if (string.IsNullOrEmpty(args.ProgressMessage))
        {
            args.ProgressMessage = "🔄 開始處理任務中...";
        }
        await _client.Chat.PostMessage(new Message
        {
            Channel = args.ChannelId,
            Text = args.ProgressMessage,
            LinkNames = false,
            Username = args.Username,
            ThreadTs = args.Ts,
        });
    }

    public async Task SendFinishProgressMessageAsync(SendFinishProgressMessageArgs args)
    {
        if (string.IsNullOrEmpty(args.FinishMessage))
        {
            args.FinishMessage = "✅ 處理完成！";
        }
        await _client.Chat.Update(new MessageUpdate
        {
            ChannelId = args.ChannelId,
            Ts = args.Ts,
            Text = args.FinishMessage,
        });
        await _client.Chat.PostMessage(new Message
        {
            Channel = args.ChannelId,
            ThreadTs = args.Ts,
            Text = args.Message
        }); 
    }

    private async Task<List<SlackHistoryItem>> InternalGetHistoryAsync(string channelId, DateTimeRange range)
    {
        var oldest = range.Start.ToSlackTs();
        var latest = range.End.ToSlackTs();
        var response = await _client.Conversations.History(
            channelId,
            latest,
            oldest,
            inclusive: true,
            limit: 100,
            includeAllMetadata: true
        );

        var result = new List<SlackHistoryItem>();
        foreach (var message in response.Messages)
        {
            var userInfo = await GetUserInfoAsync(message.User);
            var historyMessages = await GetThreadMessagesAsync(channelId, message.ThreadTs)
                .ToListAsync();
            var item = new SlackHistoryItem
            {
                Id = message.ClientMsgId,
                User = userInfo,
                Text = message.Text,
                Time = message.Ts.SlackTsToDateTime(),
                Ts = message.Ts,
                ThreadMessages = historyMessages.Skip(1).ToList()
            };
            item.ThreadMessages.Sort((a, b) => a.Time.CompareTo(b.Time));
            result.Add(item);
        }

        result.Sort((a, b) => a.Time.CompareTo(b.Time));
        return result;
    }

    public async Task<SlackUser> GetUserInfoAsync(string userId)
    {
        var result = await _methodCache.WithCacheAsync(
            async () => await InternalGetUserInfoAsync(userId),
            cacheKey: $"{nameof(GetUserInfoAsync)}_{userId}");
        return result ?? SlackUser.Empty;
    }

    private async IAsyncEnumerable<SlackMessage> GetThreadMessagesAsync(string channelId, string messageThreadTs)
    {
        if (string.IsNullOrEmpty(messageThreadTs))
        {
            yield break;
        }

        var threadResponse = await _client.Conversations.Replies(
            channelId,
            messageThreadTs
        );
        foreach (var threadMessage in threadResponse.Messages)
        {
            var userInfo = await GetUserInfoAsync(threadMessage.User);
            yield return new SlackMessage
            {
                User = userInfo,
                Text = threadMessage.Text,
                Time = threadMessage.Ts.SlackTsToDateTime(),
                Ts = threadMessage.Ts
            };
        }
    }

    private async Task<SlackUser> InternalGetUserInfoAsync(string userId)
    {
        if (string.IsNullOrEmpty(userId))
        {
            return SlackUser.Empty;
        }

        var userInfo = await _client.Users.Info(userId);
        var userName = GetUserName(userInfo.Profile);
        return new SlackUser
        {
            Id = userId,
            Name = userName,
        };
    }

    private string GetUserName(UserProfile userProfile)
    {
        if (!string.IsNullOrEmpty(userProfile.DisplayName))
        {
            return userProfile.DisplayName;
        }

        if (!string.IsNullOrEmpty(userProfile.RealName))
        {
            return userProfile.RealName;
        }

        var userName = string.Empty;
        if (!string.IsNullOrEmpty(userProfile.LastName))
        {
            userName = userProfile.LastName;
        }
        if (!string.IsNullOrEmpty(userProfile.FirstName))
        {
            userName += " " + userProfile.FirstName;
        }
        userName = userName.Trim();
        return userName;
    }
}