import '../entities/video_item.dart';

/// Repository contract for managing YouTube video feeds, local records, subscriptions, and downloads.
abstract interface class VideoRepository {
  /// Fetches latest preview videos from YouTube, merges them with local state,
  /// enforces the 10,000 record cap, and returns all stored records.
  Future<List<VideoItem>> refreshRemotePreviewFeed({String? query});

  /// Returns all locally cached video records without making network calls.
  Future<List<VideoItem>> getAllLocalRecords();

  /// Updates or inserts a single [VideoItem] and enforces the maximum record cap.
  Future<void> saveVideoRecord(VideoItem item);

  /// Marks a video as watched (`isWatched = true`) so it is removed from the main feed
  /// and appears in the [已看過清單].
  Future<VideoItem> markAsWatched(String videoId);

  /// Restores a watched video back to unwatched (`isWatched = false`).
  Future<VideoItem> unmarkWatched(String videoId);

  /// Directly subscribes to the given [channelName] (marks all videos of this channel
  /// as `isSubscribed = true` and persists the channel subscription).
  Future<List<VideoItem>> subscribeChannel(String channelName);

  /// Unsubscribes from the given [channelName].
  Future<List<VideoItem>> unsubscribeChannel(String channelName);

  /// Updates the download queue state of a video.
  Future<VideoItem> updateDownloadState({
    required String videoId,
    required VideoDownloadStatus status,
    double? progress,
    String? localFilePath,
    bool clearLocalFilePath = false,
    bool? hasEverDownloaded,
  });

  /// Deletes the downloaded video file from device storage while preserving
  /// `hasEverDownloaded = true` in the database.
  Future<VideoItem> removeLocalDownloadedFile(String videoId);

  /// Resolves a playable stream URL for online playback or background download.
  Future<String> resolveStreamUrl(String videoId);

  /// Enforces the maximum record count (default 10,000) using FIFO eviction.
  Future<int> enforceMaxRecordLimit(int maxRecords);
}
