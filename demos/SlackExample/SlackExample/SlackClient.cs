using System.Diagnostics;
using Microsoft.Extensions.Options;
using SlackNet;
using T1.Standard.DesignPatterns.Cache;
using T1.Standard.Extensions;

namespace SlackExample;

public class SlackClient : ISlackClient
{
    private readonly SlackApiClient _client;
    private readonly MethodCache _methodCache = new(TimeSpan.FromMinutes(10));

    public SlackClient(IOptions<SlackConfig> config)
    {
        _client = new SlackApiClient(config.Value.Token);
    }

    public async Task<List<SlackHistoryItem>> GetHistoryAsync(string channelId, DateTimeRange range)
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
            var item = new SlackHistoryItem
            {
                User = userInfo,
                Text = message.Text,
                Time = message.Ts.SlackTsToDateTime(),
                ThreadMessages = await GetThreadMessagesAsync(channelId, message.ThreadTs).ToListAsync()
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
            () => InternalGetUserInfoAsync(userId),
            cacheKey: GetFullname(nameof(GetUserInfoAsync)));
        return result;
    }

    private string GetFullname(string methodName)
    {
        return $"{nameof(SlackClient)}::{methodName}";
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
                Time = threadMessage.Ts.SlackTsToDateTime()
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
        var userName = userInfo.Profile.DisplayName ?? userInfo.Profile.RealName;
        return new SlackUser
        {
            Id = userId,
            Name = userName,
        };
    }
}