import 'dart:io';

import '../../domain/entities/video_item.dart';
import '../../domain/repositories/video_repository.dart';
import '../datasources/sqlite_local_datasource.dart';
import '../datasources/youtube_remote_datasource.dart';
import '../models/video_item_model.dart';

/// Concrete implementation of [VideoRepository] coordinating remote YouTube
/// previews, local SQLite state, channel subscriptions, channel exclusions,
/// channel video feeds, and local MP4 file management.
class VideoRepositoryImpl implements VideoRepository {
  VideoRepositoryImpl({
    required LocalVideoDataSource localDataSource,
    required RemoteYoutubeDataSource remoteDataSource,
  })  : _localDataSource = localDataSource,
        _remoteDataSource = remoteDataSource;

  final LocalVideoDataSource _localDataSource;
  final RemoteYoutubeDataSource _remoteDataSource;

  @override
  Future<List<VideoItem>> refreshRemotePreviewFeed({String? query}) async {
    final existingModels = await _localDataSource.getAllVideos();
    final subscribedChannels = await _localDataSource.getSubscribedChannels();
    final excludedChannels = await _localDataSource.getExcludedChannels();

    // Build a list of channels the user is interested in (subscribed or downloaded,
    // excluding any blocked channels) to feed into smart discovery on refresh.
    final interestedSet = <String>{
      ...subscribedChannels,
      for (final item in existingModels)
        if (item.hasEverDownloaded || item.isOfflineAvailable) item.channelName,
    }..removeAll(excludedChannels);

    final remoteModels = await _remoteDataSource.fetchPreviewVideos(
      query: query,
      interestedChannels: interestedSet.toList(),
    );

    return _mergeAndSaveRemoteVideos(
      remoteModels: remoteModels,
      existingModels: existingModels,
      subscribedChannels: subscribedChannels,
      excludedChannels: excludedChannels,
    );
  }

  @override
  Future<List<VideoItem>> fetchChannelVideos(String channelName) async {
    final remoteModels =
        await _remoteDataSource.fetchVideosForChannel(channelName);
    final existingModels = await _localDataSource.getAllVideos();
    final subscribedChannels = await _localDataSource.getSubscribedChannels();
    final excludedChannels = await _localDataSource.getExcludedChannels();

    return _mergeAndSaveRemoteVideos(
      remoteModels: remoteModels,
      existingModels: existingModels,
      subscribedChannels: subscribedChannels,
      excludedChannels: excludedChannels,
    );
  }

  Future<List<VideoItem>> _mergeAndSaveRemoteVideos({
    required List<VideoItemModel> remoteModels,
    required List<VideoItemModel> existingModels,
    required Set<String> subscribedChannels,
    required Set<String> excludedChannels,
  }) async {
    final existingById = <String, VideoItemModel>{
      for (final item in existingModels) item.id: item,
    };

    final mergedToUpsert = <VideoItemModel>[];

    for (final remote in remoteModels) {
      final isChannelSubscribed =
          subscribedChannels.contains(remote.channelName);
      final isChannelExcluded = excludedChannels.contains(remote.channelName);
      final existing = existingById[remote.id];
      if (existing == null) {
        mergedToUpsert.add(
          VideoItemModel.fromEntity(
            remote.copyWith(
              isSubscribed: isChannelSubscribed,
              isExcluded: isChannelExcluded,
            ),
          ),
        );
      } else {
        // Preserve local user state (watched, downloaded, local file, queue status, subscription, exclusion)
        final merged = VideoItemModel.fromEntity(
          existing.copyWith(
            title: remote.title,
            channelName: remote.channelName,
            thumbnailUrl: remote.thumbnailUrl,
            durationSeconds: remote.durationSeconds,
            isSubscribed: existing.isSubscribed || isChannelSubscribed,
            isExcluded: existing.isExcluded || isChannelExcluded,
          ),
        );
        mergedToUpsert.add(merged);
      }
    }

    await _localDataSource.upsertVideos(mergedToUpsert);
    return _localDataSource.getAllVideos();
  }

  @override
  Future<List<VideoItem>> getAllLocalRecords() {
    return _localDataSource.getAllVideos();
  }

  @override
  Future<void> saveVideoRecord(VideoItem item) {
    return _localDataSource.upsertVideo(VideoItemModel.fromEntity(item));
  }

  @override
  Future<VideoItem> markAsWatched(String videoId) async {
    final current = await _requireVideo(videoId);
    final updated = current.copyWith(
      isWatched: true,
      updatedAt: DateTime.now(),
    );
    await _localDataSource.upsertVideo(VideoItemModel.fromEntity(updated));
    return updated;
  }

  @override
  Future<VideoItem> unmarkWatched(String videoId) async {
    final current = await _requireVideo(videoId);
    final updated = current.copyWith(
      isWatched: false,
      updatedAt: DateTime.now(),
    );
    await _localDataSource.upsertVideo(VideoItemModel.fromEntity(updated));
    return updated;
  }

  @override
  Future<List<VideoItem>> subscribeChannel(String channelName) async {
    await _localDataSource.setChannelSubscription(
      channelName,
      isSubscribed: true,
    );
    return _localDataSource.getAllVideos();
  }

  @override
  Future<List<VideoItem>> unsubscribeChannel(String channelName) async {
    await _localDataSource.setChannelSubscription(
      channelName,
      isSubscribed: false,
    );
    return _localDataSource.getAllVideos();
  }

  @override
  Future<List<VideoItem>> excludeChannel(String channelName) async {
    await _localDataSource.setChannelExclusion(
      channelName,
      isExcluded: true,
    );
    return _localDataSource.getAllVideos();
  }

  @override
  Future<List<VideoItem>> unexcludeChannel(String channelName) async {
    await _localDataSource.setChannelExclusion(
      channelName,
      isExcluded: false,
    );
    return _localDataSource.getAllVideos();
  }

  @override
  Future<VideoItem> updateDownloadState({
    required String videoId,
    required VideoDownloadStatus status,
    double? progress,
    String? localFilePath,
    bool clearLocalFilePath = false,
    bool? hasEverDownloaded,
  }) async {
    final current = await _requireVideo(videoId);
    final updated = current.copyWith(
      downloadStatus: status,
      downloadProgress: progress ?? current.downloadProgress,
      localFilePath: localFilePath,
      clearLocalFilePath: clearLocalFilePath,
      hasEverDownloaded: hasEverDownloaded ?? current.hasEverDownloaded,
      updatedAt: DateTime.now(),
    );
    await _localDataSource.upsertVideo(VideoItemModel.fromEntity(updated));
    return updated;
  }

  @override
  Future<VideoItem> removeLocalDownloadedFile(String videoId) async {
    final current = await _requireVideo(videoId);
    final filePath = current.localFilePath;

    if (filePath != null && filePath.isNotEmpty) {
      final file = File(filePath);
      if (await file.exists()) {
        await file.delete();
      }
    }

    // Clear the local file and reset current downloadStatus to none,
    // while strictly keeping hasEverDownloaded = true!
    final updated = current.copyWith(
      downloadStatus: VideoDownloadStatus.none,
      downloadProgress: 0.0,
      clearLocalFilePath: true,
      hasEverDownloaded: true,
      updatedAt: DateTime.now(),
    );
    await _localDataSource.upsertVideo(VideoItemModel.fromEntity(updated));
    return updated;
  }

  @override
  Future<String> resolveStreamUrl(String videoId) {
    return _remoteDataSource.resolvePlayableStreamUrl(videoId);
  }

  @override
  Future<int> enforceMaxRecordLimit(int maxRecords) {
    return _localDataSource.enforceRecordCap(maxRecords);
  }

  Future<VideoItemModel> _requireVideo(String videoId) async {
    final item = await _localDataSource.getVideoById(videoId);
    if (item == null) {
      throw StateError('Video record not found for id: $videoId');
    }
    return item;
  }
}
