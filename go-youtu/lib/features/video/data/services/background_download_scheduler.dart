import 'dart:async';
import 'dart:collection';
import 'dart:io';

import 'package:dio/dio.dart';
import 'package:path/path.dart' as p;
import 'package:path_provider/path_provider.dart';

import '../../../../core/constants/app_constants.dart';
import '../../domain/entities/video_item.dart';
import '../../domain/repositories/video_repository.dart';

typedef VideoStateUpdatedCallback = void Function(VideoItem updatedItem);

/// Contract for abstracting file downloads so unit tests can run deterministically.
abstract interface class FileDownloader {
  Future<void> downloadToFile({
    required String url,
    required String savePath,
    required CancelToken cancelToken,
    required void Function(double progress) onProgress,
  });
}

/// Default Dio-based file downloader.
class DioFileDownloader implements FileDownloader {
  DioFileDownloader({Dio? dio}) : _dio = dio ?? Dio();

  final Dio _dio;

  @override
  Future<void> downloadToFile({
    required String url,
    required String savePath,
    required CancelToken cancelToken,
    required void Function(double progress) onProgress,
  }) async {
    await _dio.download(
      url,
      savePath,
      cancelToken: cancelToken,
      onReceiveProgress: (received, total) {
        if (total > 0) {
          onProgress((received / total).clamp(0.0, 1.0));
        }
      },
    );
  }
}

/// Background download queue scheduler.
/// When a user taps Download on a card:
/// 1. The card state immediately becomes `queued` (greyed out with downloading icon).
/// 2. The video is placed in the FIFO queue and downloaded sequentially in the background.
/// 3. Once complete, its state becomes `downloaded` and `hasEverDownloaded = true`,
///    moving the card to the top of [影片清單].
class BackgroundDownloadScheduler {
  BackgroundDownloadScheduler({
    required VideoRepository repository,
    required FileDownloader fileDownloader,
    Future<Directory> Function()? directoryResolver,
    this.onVideoUpdated,
  })  : _repository = repository,
        _fileDownloader = fileDownloader,
        _directoryResolver =
            directoryResolver ?? getApplicationDocumentsDirectory;

  final VideoRepository _repository;
  final FileDownloader _fileDownloader;
  final Future<Directory> Function() _directoryResolver;
  final VideoStateUpdatedCallback? onVideoUpdated;

  final Queue<String> _pendingQueue = Queue<String>();
  final Map<String, CancelToken> _activeCancelTokens = <String, CancelToken>{};
  bool _isProcessing = false;

  bool isQueuedOrActive(String videoId) =>
      _pendingQueue.contains(videoId) ||
      _activeCancelTokens.containsKey(videoId);

  /// Adds [videoId] to the background download queue and immediately marks it `queued`.
  Future<VideoItem> enqueueDownload(String videoId) async {
    if (!_pendingQueue.contains(videoId) &&
        !_activeCancelTokens.containsKey(videoId)) {
      _pendingQueue.addLast(videoId);
    }

    final queuedItem = await _repository.updateDownloadState(
      videoId: videoId,
      status: VideoDownloadStatus.queued,
      progress: 0.0,
    );
    onVideoUpdated?.call(queuedItem);

    unawaited(_processNextInQueue());
    return queuedItem;
  }

  /// Cancels a queued or currently downloading video task.
  Future<VideoItem> cancelDownload(String videoId) async {
    _pendingQueue.remove(videoId);
    final token = _activeCancelTokens.remove(videoId);
    if (token != null && !token.isCancelled) {
      token.cancel('Cancelled by user');
    }

    final resetItem = await _repository.updateDownloadState(
      videoId: videoId,
      status: VideoDownloadStatus.none,
      progress: 0.0,
      clearLocalFilePath: true,
    );
    onVideoUpdated?.call(resetItem);
    return resetItem;
  }

  Future<void> _processNextInQueue() async {
    if (_isProcessing) {
      return;
    }
    _isProcessing = true;

    try {
      while (_pendingQueue.isNotEmpty) {
        final videoId = _pendingQueue.removeFirst();
        final cancelToken = CancelToken();
        _activeCancelTokens[videoId] = cancelToken;

        try {
          final downloadingItem = await _repository.updateDownloadState(
            videoId: videoId,
            status: VideoDownloadStatus.downloading,
            progress: 0.02,
          );
          onVideoUpdated?.call(downloadingItem);

          final baseDir = await _directoryResolver();
          final offlineDir = Directory(
            p.join(baseDir.path, AppConstants.offlineVideoDirName),
          );
          if (!await offlineDir.exists()) {
            await offlineDir.create(recursive: true);
          }

          final targetPath = p.join(offlineDir.path, '$videoId.mp4');
          final streamUrl = await _repository.resolveStreamUrl(videoId);

          int lastReportedPercent = 2;
          await _fileDownloader.downloadToFile(
            url: streamUrl,
            savePath: targetPath,
            cancelToken: cancelToken,
            onProgress: (progress) {
              final pct = (progress * 100).round();
              if (pct - lastReportedPercent >= 5 || pct == 100) {
                lastReportedPercent = pct;
                unawaited(
                  _repository
                      .updateDownloadState(
                        videoId: videoId,
                        status: VideoDownloadStatus.downloading,
                        progress: progress,
                      )
                      .then((updated) => onVideoUpdated?.call(updated)),
                );
              }
            },
          );

          final completedItem = await _repository.updateDownloadState(
            videoId: videoId,
            status: VideoDownloadStatus.downloaded,
            progress: 1.0,
            localFilePath: targetPath,
            hasEverDownloaded: true,
          );
          onVideoUpdated?.call(completedItem);
        } on DioException catch (e) {
          if (CancelToken.isCancel(e)) {
            continue;
          }
          final failedItem = await _repository.updateDownloadState(
            videoId: videoId,
            status: VideoDownloadStatus.failed,
            progress: 0.0,
          );
          onVideoUpdated?.call(failedItem);
        } catch (_) {
          if (!cancelToken.isCancelled) {
            final failedItem = await _repository.updateDownloadState(
              videoId: videoId,
              status: VideoDownloadStatus.failed,
              progress: 0.0,
            );
            onVideoUpdated?.call(failedItem);
          }
        } finally {
          _activeCancelTokens.remove(videoId);
        }
      }
    } finally {
      _isProcessing = false;
    }
  }
}
