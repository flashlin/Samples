import '../../domain/entities/video_item.dart';

/// Data transfer object for serializing [VideoItem] to and from SQLite.
class VideoItemModel extends VideoItem {
  const VideoItemModel({
    required super.id,
    required super.title,
    required super.channelName,
    required super.thumbnailUrl,
    required super.durationSeconds,
    required super.publishedAt,
    required super.updatedAt,
    super.downloadStatus,
    super.downloadProgress,
    super.localFilePath,
    super.isWatched,
    super.hasEverDownloaded,
    super.isSubscribed,
    super.lastPlaybackPositionSeconds,
  });

  factory VideoItemModel.fromEntity(VideoItem entity) {
    return VideoItemModel(
      id: entity.id,
      title: entity.title,
      channelName: entity.channelName,
      thumbnailUrl: entity.thumbnailUrl,
      durationSeconds: entity.durationSeconds,
      publishedAt: entity.publishedAt,
      updatedAt: entity.updatedAt,
      downloadStatus: entity.downloadStatus,
      downloadProgress: entity.downloadProgress,
      localFilePath: entity.localFilePath,
      isWatched: entity.isWatched,
      hasEverDownloaded: entity.hasEverDownloaded,
      isSubscribed: entity.isSubscribed,
      lastPlaybackPositionSeconds: entity.lastPlaybackPositionSeconds,
    );
  }

  factory VideoItemModel.fromMap(Map<String, Object?> map) {
    final statusName = (map['download_status'] as String?) ?? 'none';
    final status = VideoDownloadStatus.values.firstWhere(
      (e) => e.name == statusName,
      orElse: () => VideoDownloadStatus.none,
    );

    return VideoItemModel(
      id: map['id']! as String,
      title: (map['title'] as String?) ?? '',
      channelName: (map['channel_name'] as String?) ?? '',
      thumbnailUrl: (map['thumbnail_url'] as String?) ?? '',
      durationSeconds: (map['duration_seconds'] as int?) ?? 0,
      publishedAt: DateTime.fromMillisecondsSinceEpoch(
        (map['published_at_ms'] as int?) ?? 0,
      ),
      updatedAt: DateTime.fromMillisecondsSinceEpoch(
        (map['updated_at_ms'] as int?) ?? 0,
      ),
      downloadStatus: status,
      downloadProgress: ((map['download_progress'] as num?) ?? 0.0).toDouble(),
      localFilePath: map['local_file_path'] as String?,
      isWatched: ((map['is_watched'] as int?) ?? 0) == 1,
      hasEverDownloaded: ((map['has_ever_downloaded'] as int?) ?? 0) == 1,
      isSubscribed: ((map['is_subscribed'] as int?) ?? 0) == 1,
      lastPlaybackPositionSeconds:
          (map['last_playback_position_seconds'] as int?) ?? 0,
    );
  }

  Map<String, Object?> toMap() {
    return <String, Object?>{
      'id': id,
      'title': title,
      'channel_name': channelName,
      'thumbnail_url': thumbnailUrl,
      'duration_seconds': durationSeconds,
      'published_at_ms': publishedAt.millisecondsSinceEpoch,
      'updated_at_ms': updatedAt.millisecondsSinceEpoch,
      'download_status': downloadStatus.name,
      'download_progress': downloadProgress,
      'local_file_path': localFilePath,
      'is_watched': isWatched ? 1 : 0,
      'has_ever_downloaded': hasEverDownloaded ? 1 : 0,
      'is_subscribed': isSubscribed ? 1 : 0,
      'last_playback_position_seconds': lastPlaybackPositionSeconds,
    };
  }
}
