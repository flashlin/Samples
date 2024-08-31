using Microsoft.Extensions.Options;
using SlackNet;
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

    public async Task<List<SlackHistoryItem>> GetHistoryAsync(GetHistoryArgs getHistoryArgs)
    {
        var count = 0;
        var currentRange = getHistoryArgs.Range;
        var result = new Dictionary<Guid, SlackHistoryItem>();
        while (count < getHistoryArgs.Limit)
        {
            var subResult = await InternalGetHistoryAsync(getHistoryArgs.ChannelId, currentRange);
            if (subResult.Count == 0)
            {
                break;
            }
            foreach (var item in subResult)
            {
                result[item.Id] = item;
            }
            count += subResult.Count;
            currentRange = new DateTimeRange
            {
                Start = subResult[^1].Time,
                End = getHistoryArgs.Range.End
            };
        }
        return result.Values.ToList();
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
            () => InternalGetUserInfoAsync(userId)!,
            cacheKey: GetFullname(nameof(GetUserInfoAsync)));
        return result!;
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