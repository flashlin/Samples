import 'package:equatable/equatable.dart';

/// Represents the current download lifecycle state of a video card.
enum VideoDownloadStatus {
  /// Not queued and not downloaded locally.
  none,

  /// Waiting in the background queue to start downloading.
  queued,

  /// Actively downloading in the background queue.
  downloading,

  /// Downloaded and stored on the local device filesystem.
  downloaded,

  /// Last download attempt failed.
  failed;

  /// Returns true when the card should appear greyed out with a downloading indicator.
  bool get isBusyInQueue =>
      this == VideoDownloadStatus.queued ||
      this == VideoDownloadStatus.downloading;
}

/// Core domain entity representing a YouTube video item and its local state.
class VideoItem extends Equatable {
  const VideoItem({
    required this.id,
    required this.title,
    required this.channelName,
    required this.thumbnailUrl,
    required this.durationSeconds,
    required this.publishedAt,
    required this.updatedAt,
    this.downloadStatus = VideoDownloadStatus.none,
    this.downloadProgress = 0.0,
    this.localFilePath,
    this.isWatched = false,
    this.hasEverDownloaded = false,
    this.isSubscribed = false,
    this.lastPlaybackPositionSeconds = 0,
  });

  final String id;
  final String title;
  final String channelName;
  final String thumbnailUrl;
  final int durationSeconds;
  final DateTime publishedAt;
  final DateTime updatedAt;
  final VideoDownloadStatus downloadStatus;
  final double downloadProgress;
  final String? localFilePath;
  final bool isWatched;

  /// Remains true even after the user deletes the local MP4 file,
  /// so the item stays recorded in the [已下載過清單] tab.
  final bool hasEverDownloaded;

  /// True if the user has subscribed to this video's channel.
  final bool isSubscribed;

  final int lastPlaybackPositionSeconds;

  /// True if the video has a downloaded local file ready for offline playback.
  bool get isOfflineAvailable =>
      downloadStatus == VideoDownloadStatus.downloaded &&
      localFilePath != null &&
      localFilePath!.isNotEmpty;

  /// True if the card should be rendered in greyscale with a downloading icon.
  bool get isDownloadingOrQueued => downloadStatus.isBusyInQueue;

  Duration get duration => Duration(seconds: durationSeconds);

  VideoItem copyWith({
    String? title,
    String? channelName,
    String? thumbnailUrl,
    int? durationSeconds,
    DateTime? publishedAt,
    DateTime? updatedAt,
    VideoDownloadStatus? downloadStatus,
    double? downloadProgress,
    String? localFilePath,
    bool clearLocalFilePath = false,
    bool? isWatched,
    bool? hasEverDownloaded,
    bool? isSubscribed,
    int? lastPlaybackPositionSeconds,
  }) {
    return VideoItem(
      id: id,
      title: title ?? this.title,
      channelName: channelName ?? this.channelName,
      thumbnailUrl: thumbnailUrl ?? this.thumbnailUrl,
      durationSeconds: durationSeconds ?? this.durationSeconds,
      publishedAt: publishedAt ?? this.publishedAt,
      updatedAt: updatedAt ?? this.updatedAt,
      downloadStatus: downloadStatus ?? this.downloadStatus,
      downloadProgress: downloadProgress ?? this.downloadProgress,
      localFilePath:
          clearLocalFilePath ? null : (localFilePath ?? this.localFilePath),
      isWatched: isWatched ?? this.isWatched,
      hasEverDownloaded: hasEverDownloaded ?? this.hasEverDownloaded,
      isSubscribed: isSubscribed ?? this.isSubscribed,
      lastPlaybackPositionSeconds:
          lastPlaybackPositionSeconds ?? this.lastPlaybackPositionSeconds,
    );
  }

  @override
  List<Object?> get props => <Object?>[
        id,
        title,
        channelName,
        thumbnailUrl,
        durationSeconds,
        publishedAt,
        updatedAt,
        downloadStatus,
        downloadProgress,
        localFilePath,
        isWatched,
        hasEverDownloaded,
        isSubscribed,
        lastPlaybackPositionSeconds,
      ];
}
