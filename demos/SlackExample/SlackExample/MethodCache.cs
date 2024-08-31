using System.Collections.Concurrent;

namespace SlackExample;

public class MethodCache
{
    private readonly ConcurrentDictionary<string, Lazy<Task<object>>> _cache = new();
    private readonly TimeSpan _defaultCacheDuration;

    public MethodCache(TimeSpan? defaultCacheDuration = null)
    {
        _defaultCacheDuration = defaultCacheDuration ?? TimeSpan.FromMinutes(10);
    }

    private async Task<TResult> GetOrAddAsync<TResult>(string key, Func<Task<TResult>> factory, TimeSpan? userCacheDuration = null)
    {
        var lazyResult = _cache.GetOrAdd(key, _ => new Lazy<Task<object>>(async () =>
        {
            var result = await factory();
            ScheduleRemoval(key, userCacheDuration ?? _defaultCacheDuration);
            return result!;
        }));

        return (TResult)await lazyResult.Value;
    }
    
    private async Task<TResult> GetOrAdd<TResult>(string key, Func<TResult> factory, TimeSpan? userCacheDuration = null)
    {
        var lazyResult = _cache.GetOrAdd(key, _ => new Lazy<Task<object>>(() =>
        {
            var result = factory();
            ScheduleRemoval(key, userCacheDuration ?? _defaultCacheDuration);
            return Task.FromResult((object)result!);
        }));

        return (TResult)await lazyResult.Value;
    }
    
    public async Task<TResult> WithCacheAsync<TResult>(Func<Task<TResult>> method,
        string cacheKey,
        TimeSpan? cacheDuration = null)
    {
        return await GetOrAddAsync(cacheKey, method, cacheDuration);
    }
    
    public async Task<TResult> WithCache<TResult>(Func<TResult> method,
        string cacheKey,
        TimeSpan? cacheDuration = null)
    {
        return await GetOrAdd(cacheKey, method, cacheDuration);
    }

    private void ScheduleRemoval(string key, TimeSpan delay)
    {
        Task.Delay(delay).ContinueWith(_ => _cache.TryRemove(key, out var _));
    }
}