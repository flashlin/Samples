import 'package:flutter_riverpod/flutter_riverpod.dart';

import '../../data/datasources/sqlite_local_datasource.dart';
import '../../data/datasources/youtube_remote_datasource.dart';
import '../../data/repositories/video_repository_impl.dart';
import '../../data/services/background_download_scheduler.dart';
import '../../domain/entities/video_item.dart';
import '../../domain/repositories/video_repository.dart';
import '../../domain/usecases/video_usecases.dart';

/// Top navigation tab enum matching the 6 required sections.
enum TopNavigationTab {
  videoFeed('影片清單'),
  downloadedHistory('已下載過清單'),
  subscribedList('訂閱清單'),
  downloadQueue('下載佇列清單'),
  watchedHistory('已看過清單'),
  excludedChannels('排除頻道');

  const TopNavigationTab(this.label);
  final String label;
}

/// Immutable state holding all video records, derived lists for the 6 tabs,
/// and current playback/loading states.
class VideoBoardState {
  const VideoBoardState({
    this.allRecords = const <VideoItem>[],
    this.activeTab = TopNavigationTab.videoFeed,
    this.isLoading = false,
    this.activePlayingVideoId,
    this.errorMessage,
  });

  final List<VideoItem> allRecords;
  final TopNavigationTab activeTab;
  final bool isLoading;
  final String? activePlayingVideoId;
  final String? errorMessage;

  VideoBoardState copyWith({
    List<VideoItem>? allRecords,
    TopNavigationTab? activeTab,
    bool? isLoading,
    String? activePlayingVideoId,
    bool clearActivePlayingVideo = false,
    String? errorMessage,
    bool clearError = false,
  }) {
    return VideoBoardState(
      allRecords: allRecords ?? this.allRecords,
      activeTab: activeTab ?? this.activeTab,
      isLoading: isLoading ?? this.isLoading,
      activePlayingVideoId: clearActivePlayingVideo
          ? null
          : (activePlayingVideoId ?? this.activePlayingVideoId),
      errorMessage: clearError ? null : (errorMessage ?? this.errorMessage),
    );
  }
}

final localDataSourceProvider = Provider<LocalVideoDataSource>((ref) {
  return SqliteLocalDataSource();
});

final remoteDataSourceProvider = Provider<RemoteYoutubeDataSource>((ref) {
  return YoutubeExplodeRemoteDataSource();
});

final videoRepositoryProvider = Provider<VideoRepository>((ref) {
  return VideoRepositoryImpl(
    localDataSource: ref.watch(localDataSourceProvider),
    remoteDataSource: ref.watch(remoteDataSourceProvider),
  );
});

final videoUseCasesProvider = Provider<VideoUseCases>((ref) {
  return VideoUseCases(ref.watch(videoRepositoryProvider));
});

final fileDownloaderProvider = Provider<FileDownloader>((ref) {
  return DioFileDownloader();
});

final videoBoardProvider =
    StateNotifierProvider<VideoBoardNotifier, VideoBoardState>((ref) {
  final useCases = ref.watch(videoUseCasesProvider);
  final repository = ref.watch(videoRepositoryProvider);
  final downloader = ref.watch(fileDownloaderProvider);
  return VideoBoardNotifier(
    useCases: useCases,
    repository: repository,
    fileDownloader: downloader,
  );
});

/// StateNotifier coordinating all user actions across the 6 tabs, channel pages, and video cards.
class VideoBoardNotifier extends StateNotifier<VideoBoardState> {
  VideoBoardNotifier({
    required VideoUseCases useCases,
    required VideoRepository repository,
    required FileDownloader fileDownloader,
    bool autoInitialize = true,
  })  : _useCases = useCases,
        super(const VideoBoardState()) {
    _scheduler = BackgroundDownloadScheduler(
      repository: repository,
      fileDownloader: fileDownloader,
      onVideoUpdated: _upsertSingleRecordInState,
    );
    if (autoInitialize) {
      initializeFeed();
    }
  }

  final VideoUseCases _useCases;
  late final BackgroundDownloadScheduler _scheduler;

  List<VideoItem> get activeFeedVideos =>
      _useCases.buildActiveFeedList(state.allRecords);

  List<VideoItem> get subscribedVideos =>
      _useCases.buildSubscribedList(state.allRecords);

  List<VideoItem> get excludedVideos =>
      _useCases.buildExcludedList(state.allRecords);

  List<VideoItem> get downloadQueueVideos =>
      _useCases.buildDownloadQueueList(state.allRecords);

  List<VideoItem> get watchedHistoryVideos =>
      _useCases.buildWatchedHistoryList(state.allRecords);

  List<VideoItem> get downloadedHistoryVideos =>
      _useCases.buildDownloadedHistoryList(state.allRecords);

  List<VideoItem> getVideosForChannel(String channelName) =>
      _useCases.buildChannelVideosList(state.allRecords, channelName);

  Future<void> initializeFeed({String? query}) async {
    state = state.copyWith(isLoading: true, clearError: true);
    try {
      // Load existing cached records immediately so offline videos show right away
      final cached = await _useCases.loadLocalRecords();
      if (cached.isNotEmpty) {
        state = state.copyWith(allRecords: cached);
      }

      // Refresh latest YouTube preview cards (with random discovery + related channels)
      final refreshed = await _useCases.fetchInitialOrRefreshFeed(query: query);
      state = state.copyWith(allRecords: refreshed, isLoading: false);
    } catch (e) {
      state = state.copyWith(
        isLoading: false,
        errorMessage: '無法更新最新影片清單: $e',
      );
    }
  }

  /// Fetches all videos for a specific [channelName] when entering the channel page.
  Future<void> loadChannelVideos(String channelName) async {
    state = state.copyWith(isLoading: true, clearError: true);
    try {
      final updatedAll = await _useCases.fetchChannelVideos(channelName);
      if (mounted) {
        state = state.copyWith(allRecords: updatedAll, isLoading: false);
      }
    } catch (e) {
      if (mounted) {
        state = state.copyWith(
          isLoading: false,
          errorMessage: '無法載入頻道 ($channelName) 影片清單: $e',
        );
      }
    }
  }

  void selectTab(TopNavigationTab tab) {
    state = state.copyWith(activeTab: tab);
  }

  void setActivePlayingVideo(String? videoId) {
    if (videoId == null) {
      state = state.copyWith(clearActivePlayingVideo: true);
    } else {
      state = state.copyWith(activePlayingVideoId: videoId);
    }
  }

  /// Marks a video as watched: immediately removes it from [影片清單]
  /// and records it in [已看過清單] (capped at 10,000 records).
  Future<void> markAsWatched(String videoId) async {
    final shouldStopPlayer = state.activePlayingVideoId == videoId;
    final updated = await _useCases.markVideoAsWatched(videoId);
    _upsertSingleRecordInState(
      updated,
      clearPlaying: shouldStopPlayer,
    );
  }

  /// Restores a watched video back to unwatched so it returns to [影片清單].
  Future<void> restoreWatched(String videoId) async {
    final updated = await _useCases.restoreWatchedVideo(videoId);
    _upsertSingleRecordInState(updated);
  }

  /// Directly subscribes to a channel (or unsubscribes if `subscribe == false`),
  /// immediately updating all cards from this channel and prioritizing them in [影片清單].
  Future<void> setChannelSubscription(
    String channelName, {
    required bool subscribe,
  }) async {
    final updatedAll = subscribe
        ? await _useCases.subscribeToChannel(channelName)
        : await _useCases.unsubscribeFromChannel(channelName);
    if (mounted) {
      state = state.copyWith(allRecords: updatedAll);
    }
  }

  /// Adds or removes a channel from the exclusion list ([排除頻道]).
  /// When excluded, all videos from this channel are immediately hidden from [影片清單].
  Future<void> setChannelExclusion(
    String channelName, {
    required bool exclude,
  }) async {
    final updatedAll = exclude
        ? await _useCases.excludeChannel(channelName)
        : await _useCases.unexcludeChannel(channelName);
    if (mounted) {
      state = state.copyWith(allRecords: updatedAll);
    }
  }

  /// Enqueues a video for background downloading.
  /// Card immediately becomes greyed out (`queued` -> `downloading`) with downloading icon.
  Future<void> startBackgroundDownload(String videoId) async {
    final queued = await _scheduler.enqueueDownload(videoId);
    _upsertSingleRecordInState(queued);
  }

  /// Cancels an active or queued download task.
  Future<void> cancelDownload(String videoId) async {
    final reset = await _scheduler.cancelDownload(videoId);
    _upsertSingleRecordInState(reset);
  }

  /// Removes the local MP4 file of a downloaded video while keeping
  /// `hasEverDownloaded = true` so it remains in [已下載過清單].
  Future<void> removeLocalDownloadedVideo(String videoId) async {
    final updated = await _useCases.removeLocalFileKeepingHistory(videoId);
    _upsertSingleRecordInState(updated);
  }

  Future<String> resolveStreamUrl(String videoId) {
    return _useCases.getPlayableStreamUrl(videoId);
  }

  void _upsertSingleRecordInState(
    VideoItem updatedItem, {
    bool clearPlaying = false,
  }) {
    if (!mounted) {
      return;
    }
    final updatedList = <VideoItem>[];
    bool replaced = false;
    for (final existing in state.allRecords) {
      if (existing.id == updatedItem.id) {
        updatedList.add(updatedItem);
        replaced = true;
      } else {
        updatedList.add(existing);
      }
    }
    if (!replaced) {
      updatedList.insert(0, updatedItem);
    }
    state = state.copyWith(
      allRecords: updatedList,
      clearActivePlayingVideo: clearPlaying,
    );
  }
}
