import '../../../../core/constants/app_constants.dart';
import '../entities/video_item.dart';
import '../repositories/video_repository.dart';

/// Pure domain logic for sorting and filtering video lists across the 6 top tabs
/// and channel-specific views, as well as coordinating user actions on video cards.
class VideoUseCases {
  const VideoUseCases(this._repository);

  final VideoRepository _repository;

  /// Filters and sorts the main [影片清單]:
  /// 1. Excludes any video marked as watched (`isWatched == true`) OR whose channel is excluded (`isExcluded == true`).
  /// 2. Places downloaded offline videos (`isOfflineAvailable == true`) at the top (已下載優先).
  /// 3. Places subscribed channel videos (`isSubscribed == true`) next (已訂閱優先).
  /// 4. Sorts remaining videos by `publishedAt` descending (newest preview first).
  List<VideoItem> buildActiveFeedList(List<VideoItem> allRecords) {
    final visible = allRecords
        .where((item) => !item.isWatched && !item.isExcluded)
        .toList();

    visible.sort((a, b) {
      final aDownloaded = a.isOfflineAvailable ? 1 : 0;
      final bDownloaded = b.isOfflineAvailable ? 1 : 0;
      if (aDownloaded != bDownloaded) {
        return bDownloaded.compareTo(aDownloaded);
      }

      final aSubscribed = a.isSubscribed ? 1 : 0;
      final bSubscribed = b.isSubscribed ? 1 : 0;
      if (aSubscribed != bSubscribed) {
        return bSubscribed.compareTo(aSubscribed);
      }

      return b.publishedAt.compareTo(a.publishedAt);
    });

    return visible;
  }

  /// Filters and sorts all videos belonging to a specific [channelName] (used when clicking [進入頻道]):
  /// Places downloaded offline videos first, followed by unwatched videos, then newest `publishedAt`.
  List<VideoItem> buildChannelVideosList(
    List<VideoItem> allRecords,
    String channelName,
  ) {
    final target = channelName.trim().toLowerCase();
    final channelVideos = allRecords
        .where((item) => item.channelName.trim().toLowerCase() == target)
        .toList();

    channelVideos.sort((a, b) {
      final aDownloaded = a.isOfflineAvailable ? 1 : 0;
      final bDownloaded = b.isOfflineAvailable ? 1 : 0;
      if (aDownloaded != bDownloaded) {
        return bDownloaded.compareTo(aDownloaded);
      }
      final aUnwatched = a.isWatched ? 0 : 1;
      final bUnwatched = b.isWatched ? 0 : 1;
      if (aUnwatched != bUnwatched) {
        return bUnwatched.compareTo(aUnwatched);
      }
      return b.publishedAt.compareTo(a.publishedAt);
    });

    return channelVideos;
  }

  /// Filters and sorts the [訂閱清單]:
  /// Shows all videos belonging to subscribed channels (`isSubscribed == true` and not excluded),
  /// prioritizing downloaded videos first, then newest `publishedAt`.
  List<VideoItem> buildSubscribedList(List<VideoItem> allRecords) {
    final subscribed = allRecords
        .where((item) => item.isSubscribed && !item.isExcluded)
        .toList();
    subscribed.sort((a, b) {
      final aDownloaded = a.isOfflineAvailable ? 1 : 0;
      final bDownloaded = b.isOfflineAvailable ? 1 : 0;
      if (aDownloaded != bDownloaded) {
        return bDownloaded.compareTo(aDownloaded);
      }
      return b.publishedAt.compareTo(a.publishedAt);
    });
    return subscribed;
  }

  /// Filters and sorts the [排除頻道] list:
  /// Shows all videos belonging to excluded channels (`isExcluded == true`).
  List<VideoItem> buildExcludedList(List<VideoItem> allRecords) {
    final excluded = allRecords.where((item) => item.isExcluded).toList()
      ..sort((a, b) => b.updatedAt.compareTo(a.updatedAt));
    return excluded;
  }

  /// Filters and sorts the [下載佇列清單]:
  /// Shows videos currently downloading first, then queued items, then failed items.
  List<VideoItem> buildDownloadQueueList(List<VideoItem> allRecords) {
    final queueItems = allRecords
        .where(
          (item) =>
              item.downloadStatus == VideoDownloadStatus.downloading ||
              item.downloadStatus == VideoDownloadStatus.queued ||
              item.downloadStatus == VideoDownloadStatus.failed,
        )
        .toList();

    queueItems.sort((a, b) {
      int priority(VideoDownloadStatus status) => switch (status) {
            VideoDownloadStatus.downloading => 0,
            VideoDownloadStatus.queued => 1,
            VideoDownloadStatus.failed => 2,
            _ => 3,
          };
      final pCompare =
          priority(a.downloadStatus).compareTo(priority(b.downloadStatus));
      if (pCompare != 0) {
        return pCompare;
      }
      return a.updatedAt.compareTo(b.updatedAt);
    });

    return queueItems;
  }

  /// Filters and sorts the [已看過清單]:
  /// Shows all watched videos ordered by most recently marked watched, up to 10,000 items.
  List<VideoItem> buildWatchedHistoryList(List<VideoItem> allRecords) {
    final watched = allRecords.where((item) => item.isWatched).toList()
      ..sort((a, b) => b.updatedAt.compareTo(a.updatedAt));
    if (watched.length > AppConstants.maxStoredRecords) {
      return watched.sublist(0, AppConstants.maxStoredRecords);
    }
    return watched;
  }

  /// Filters and sorts the [已下載過清單]:
  /// Includes both currently offline videos AND videos whose local file was removed
  /// (`hasEverDownloaded == true`).
  List<VideoItem> buildDownloadedHistoryList(List<VideoItem> allRecords) {
    final downloadedHistory = allRecords
        .where((item) => item.hasEverDownloaded || item.isOfflineAvailable)
        .toList()
      ..sort((a, b) => b.updatedAt.compareTo(a.updatedAt));
    return downloadedHistory;
  }

  Future<List<VideoItem>> fetchInitialOrRefreshFeed({String? query}) async {
    final records = await _repository.refreshRemotePreviewFeed(query: query);
    await _repository.enforceMaxRecordLimit(AppConstants.maxStoredRecords);
    return records;
  }

  Future<List<VideoItem>> fetchChannelVideos(String channelName) async {
    final records = await _repository.fetchChannelVideos(channelName);
    await _repository.enforceMaxRecordLimit(AppConstants.maxStoredRecords);
    return records;
  }

  Future<List<VideoItem>> loadLocalRecords() {
    return _repository.getAllLocalRecords();
  }

  Future<VideoItem> markVideoAsWatched(String videoId) async {
    final updated = await _repository.markAsWatched(videoId);
    await _repository.enforceMaxRecordLimit(AppConstants.maxStoredRecords);
    return updated;
  }

  Future<VideoItem> restoreWatchedVideo(String videoId) {
    return _repository.unmarkWatched(videoId);
  }

  Future<List<VideoItem>> subscribeToChannel(String channelName) {
    return _repository.subscribeChannel(channelName);
  }

  Future<List<VideoItem>> unsubscribeFromChannel(String channelName) {
    return _repository.unsubscribeChannel(channelName);
  }

  Future<List<VideoItem>> excludeChannel(String channelName) {
    return _repository.excludeChannel(channelName);
  }

  Future<List<VideoItem>> unexcludeChannel(String channelName) {
    return _repository.unexcludeChannel(channelName);
  }

  Future<VideoItem> removeLocalFileKeepingHistory(String videoId) {
    return _repository.removeLocalDownloadedFile(videoId);
  }

  Future<String> getPlayableStreamUrl(String videoId) {
    return _repository.resolveStreamUrl(videoId);
  }
}
