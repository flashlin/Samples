import 'dart:io';

import 'package:dio/dio.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:go_youtu/core/constants/app_constants.dart';
import 'package:go_youtu/features/video/data/datasources/sqlite_local_datasource.dart';
import 'package:go_youtu/features/video/data/datasources/youtube_remote_datasource.dart';
import 'package:go_youtu/features/video/data/models/video_item_model.dart';
import 'package:go_youtu/features/video/data/repositories/video_repository_impl.dart';
import 'package:go_youtu/features/video/data/services/background_download_scheduler.dart';
import 'package:go_youtu/features/video/domain/entities/video_item.dart';
import 'package:go_youtu/features/video/domain/usecases/video_usecases.dart';
import 'package:sqflite_common_ffi/sqflite_ffi.dart';

class _FakeRemoteDataSource implements RemoteYoutubeDataSource {
  @override
  Future<List<VideoItemModel>> fetchPreviewVideos({String? query}) async {
    final baseTime = DateTime(2026, 11, 16, 12);
    return <VideoItemModel>[
      VideoItemModel(
        id: 'vid_oldest',
        title: 'Oldest Video',
        channelName: 'Channel A',
        thumbnailUrl: 'https://example.com/1.jpg',
        durationSeconds: 120,
        publishedAt: baseTime.subtract(const Duration(hours: 5)),
        updatedAt: baseTime.subtract(const Duration(hours: 5)),
      ),
      VideoItemModel(
        id: 'vid_subscribed',
        title: 'Subscribed Channel Video',
        channelName: 'Channel Subscribed',
        thumbnailUrl: 'https://example.com/sub.jpg',
        durationSeconds: 180,
        publishedAt: baseTime.subtract(const Duration(hours: 4)),
        updatedAt: baseTime.subtract(const Duration(hours: 4)),
      ),
      VideoItemModel(
        id: 'vid_middle',
        title: 'Middle Video',
        channelName: 'Channel B',
        thumbnailUrl: 'https://example.com/2.jpg',
        durationSeconds: 240,
        publishedAt: baseTime.subtract(const Duration(hours: 2)),
        updatedAt: baseTime.subtract(const Duration(hours: 2)),
      ),
      VideoItemModel(
        id: 'vid_newest',
        title: 'Newest Video',
        channelName: 'Channel C',
        thumbnailUrl: 'https://example.com/3.jpg',
        durationSeconds: 360,
        publishedAt: baseTime,
        updatedAt: baseTime,
      ),
    ];
  }

  @override
  Future<String> resolvePlayableStreamUrl(String videoId) async {
    return 'https://example.com/stream/$videoId.mp4';
  }
}

class _FakeFileDownloader implements FileDownloader {
  @override
  Future<void> downloadToFile({
    required String url,
    required String savePath,
    required CancelToken cancelToken,
    required void Function(double progress) onProgress,
  }) async {
    onProgress(0.5);
    final file = File(savePath);
    await file.parent.create(recursive: true);
    await file.writeAsBytes(const <int>[0, 1, 2, 3]);
    onProgress(1.0);
  }
}

void main() {
  sqfliteFfiInit();

  late Database db;
  late SqliteLocalDataSource localDataSource;
  late VideoRepositoryImpl repository;
  late VideoUseCases useCases;
  late Directory tempDir;

  setUp(() async {
    tempDir = await Directory.systemTemp.createTemp('go_youtu_test_');
    db = await databaseFactoryFfi.openDatabase(
      inMemoryDatabasePath,
      options: OpenDatabaseOptions(
        version: SqliteLocalDataSource.schemaVersion,
        onCreate: SqliteLocalDataSource.createSchema,
      ),
    );
    localDataSource = SqliteLocalDataSource(injectedDatabase: db);
    repository = VideoRepositoryImpl(
      localDataSource: localDataSource,
      remoteDataSource: _FakeRemoteDataSource(),
    );
    useCases = VideoUseCases(repository);
  });

  tearDown(() async {
    await db.close();
    if (await tempDir.exists()) {
      await tempDir.delete(recursive: true);
    }
  });

  test(
    '1. 卡片清單排列方式：已下載優先 -> 已訂閱的優先 -> 最新時間，且已看過的影片移除不顯示',
    () async {
      await useCases.fetchInitialOrRefreshFeed();

      // 1) Mark vid_oldest as downloaded
      await repository.updateDownloadState(
        videoId: 'vid_oldest',
        status: VideoDownloadStatus.downloaded,
        progress: 1.0,
        localFilePath: '${tempDir.path}/vid_oldest.mp4',
        hasEverDownloaded: true,
      );

      // 2) Subscribe to 'Channel Subscribed' (vid_subscribed)
      await useCases.subscribeToChannel('Channel Subscribed');

      // 3) Mark vid_newest as watched
      await useCases.markVideoAsWatched('vid_newest');

      final allRecords = await useCases.loadLocalRecords();
      final activeFeed = useCases.buildActiveFeedList(allRecords);
      final subscribedList = useCases.buildSubscribedList(allRecords);
      final watchedHistory = useCases.buildWatchedHistoryList(allRecords);

      // vid_newest is watched -> must not appear in activeFeed
      expect(activeFeed.map((e) => e.id).contains('vid_newest'), isFalse);
      expect(watchedHistory.map((e) => e.id), contains('vid_newest'));

      // Subscribed list must contain vid_subscribed
      expect(subscribedList.map((e) => e.id), contains('vid_subscribed'));

      // Order in activeFeed must be:
      // [0] vid_oldest (已下載優先)
      // [1] vid_subscribed (已訂閱優先)
      // [2] vid_middle (一般未看過影片)
      expect(
        activeFeed.map((e) => e.id).toList(),
        equals(<String>['vid_oldest', 'vid_subscribed', 'vid_middle']),
      );
    },
  );

  test(
    '2. 移除已下載影片檔案後，切換回線上狀態但保留「已下載過」紀錄',
    () async {
      await useCases.fetchInitialOrRefreshFeed();

      final scheduler = BackgroundDownloadScheduler(
        repository: repository,
        fileDownloader: _FakeFileDownloader(),
        directoryResolver: () async => tempDir,
      );

      final queued = await scheduler.enqueueDownload('vid_middle');
      expect(queued.isDownloadingOrQueued, isTrue);

      await Future<void>.delayed(const Duration(milliseconds: 100));

      final afterDownload = await localDataSource.getVideoById('vid_middle');
      expect(afterDownload, isNotNull);
      expect(afterDownload!.isOfflineAvailable, isTrue);
      expect(afterDownload.hasEverDownloaded, isTrue);
      expect(File(afterDownload.localFilePath!).existsSync(), isTrue);

      final afterRemove =
          await useCases.removeLocalFileKeepingHistory('vid_middle');
      expect(afterRemove.isOfflineAvailable, isFalse);
      expect(afterRemove.localFilePath, isNull);
      expect(afterRemove.hasEverDownloaded, isTrue);

      final allRecords = await useCases.loadLocalRecords();
      final downloadedHistory = useCases.buildDownloadedHistoryList(allRecords);
      expect(downloadedHistory.map((e) => e.id), contains('vid_middle'));
    },
  );

  test(
    '3. 紀錄筆數超過上限時自動淘汰最舊紀錄 (FIFO Cap / 10,000 limit mechanism)',
    () async {
      final now = DateTime(2026, 11, 16, 14);
      final batchItems = List<VideoItemModel>.generate(15, (i) {
        return VideoItemModel(
          id: 'item_$i',
          title: 'Video $i',
          channelName: 'Channel',
          thumbnailUrl: 'https://example.com/$i.jpg',
          durationSeconds: 100,
          publishedAt: now.add(Duration(minutes: i)),
          updatedAt: now.add(Duration(minutes: i)),
          isWatched: true,
        );
      });

      await localDataSource.upsertVideos(batchItems);
      final evictedCount = await repository.enforceMaxRecordLimit(10);
      expect(evictedCount, equals(5));

      final remaining = await repository.getAllLocalRecords();
      expect(remaining.length, equals(10));
      expect(remaining.map((e) => e.id).contains('item_0'), isFalse);
      expect(remaining.map((e) => e.id).contains('item_14'), isTrue);
      expect(AppConstants.maxStoredRecords, equals(10000));
    },
  );
}
