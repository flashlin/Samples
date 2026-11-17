import 'dart:io';

import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:video_player/video_player.dart';

import '../../domain/entities/video_item.dart';
import '../pages/channel_videos_page.dart';
import '../pages/fullscreen_player_page.dart';
import '../providers/video_providers.dart';

/// Displays a single YouTube video preview card with:
/// - Greyed-out visual overlay + downloading icon when queued or downloading in background.
/// - Inline video player (offline local MP4 or online stream).
/// - Streamlined control bar underneath the card:
///   1. 進入頻道 (Opens ChannelVideosPage showing all videos from this channel)
///   2. 訂閱 / 已訂閱 (Directly subscribe to channel, prioritizing its videos)
///   3. 排除 / 取消排除 (Exclude channel from feed -> listed in [排除頻道])
///   4. 標記已看過 (Mark as watched -> removes from feed)
///   5. Download (Enqueues in background download queue & greys out card)
///   6. 標記 Icon (離線 / 線上)
///   7. 移除已下載影片但保留已下載過紀錄 (Remove local file keeping history)
///   8. 全螢幕播放 (Fullscreen playback)
class VideoCardWidget extends ConsumerStatefulWidget {
  const VideoCardWidget({
    super.key,
    required this.video,
    this.showRestoreWatchedButton = false,
    this.showEnterChannelButton = true,
  });

  final VideoItem video;
  final bool showRestoreWatchedButton;
  final bool showEnterChannelButton;

  @override
  ConsumerState<VideoCardWidget> createState() => _VideoCardWidgetState();
}

class _VideoCardWidgetState extends ConsumerState<VideoCardWidget> {
  VideoPlayerController? _controller;
  bool _isInitializingPlayer = false;
  String? _playerError;

  static const ColorFilter _greyscaleFilter = ColorFilter.matrix(<double>[
    0.2126, 0.7152, 0.0722, 0, 0, //
    0.2126, 0.7152, 0.0722, 0, 0, //
    0.2126, 0.7152, 0.0722, 0, 0, //
    0, 0, 0, 1, 0, //
  ]);

  @override
  void didUpdateWidget(covariant VideoCardWidget oldWidget) {
    super.didUpdateWidget(oldWidget);
    final activeId = ref.read(videoBoardProvider).activePlayingVideoId;
    if (activeId != widget.video.id && _controller != null) {
      _disposeController();
    }
  }

  @override
  void dispose() {
    _disposeController();
    super.dispose();
  }

  void _disposeController() {
    final old = _controller;
    _controller = null;
    old?.removeListener(_onControllerUpdated);
    old?.dispose();
  }

  void _onControllerUpdated() {
    if (mounted) {
      setState(() {});
    }
  }

  Future<VideoPlayerController?> _ensureControllerInitialized() async {
    if (_controller != null && _controller!.value.isInitialized) {
      return _controller;
    }

    setState(() {
      _isInitializingPlayer = true;
      _playerError = null;
    });

    ref.read(videoBoardProvider.notifier).setActivePlayingVideo(widget.video.id);

    try {
      _disposeController();
      late final VideoPlayerController newController;

      if (widget.video.isOfflineAvailable &&
          widget.video.localFilePath != null &&
          File(widget.video.localFilePath!).existsSync()) {
        newController = VideoPlayerController.file(
          File(widget.video.localFilePath!),
        );
      } else {
        final streamUrl = await ref
            .read(videoBoardProvider.notifier)
            .resolveStreamUrl(widget.video.id);
        newController = VideoPlayerController.networkUrl(Uri.parse(streamUrl));
      }

      await newController.initialize();
      newController.addListener(_onControllerUpdated);
      _controller = newController;
      if (mounted) {
        setState(() {
          _isInitializingPlayer = false;
        });
      }
      return newController;
    } catch (e) {
      if (mounted) {
        setState(() {
          _isInitializingPlayer = false;
          _playerError = '無法初始化播放器: $e';
        });
      }
      return null;
    }
  }

  Future<void> _handlePlay() async {
    final ctrl = await _ensureControllerInitialized();
    await ctrl?.play();
  }

  Future<void> _handleFullscreen() async {
    final ctrl = await _ensureControllerInitialized();
    if (ctrl == null || !mounted) {
      return;
    }
    await ctrl.play();
    if (!mounted) {
      return;
    }
    await Navigator.of(context).push(
      MaterialPageRoute<void>(
        builder: (_) => FullscreenPlayerPage(
          video: widget.video,
          controller: ctrl,
        ),
      ),
    );
  }

  void _handleEnterChannel() {
    Navigator.of(context).push(
      MaterialPageRoute<void>(
        builder: (_) => ChannelVideosPage(
          channelName: widget.video.channelName,
        ),
      ),
    );
  }

  String _formatSeconds(int totalSeconds) {
    final duration = Duration(seconds: totalSeconds);
    final minutes = duration.inMinutes.remainder(60).toString().padLeft(2, '0');
    final seconds = duration.inSeconds.remainder(60).toString().padLeft(2, '0');
    if (duration.inHours > 0) {
      return '${duration.inHours}:$minutes:$seconds';
    }
    return '$minutes:$seconds';
  }

  @override
  Widget build(BuildContext context) {
    final video = widget.video;
    final isGreyedOutDownloading = video.isDownloadingOrQueued;
    final isOffline = video.isOfflineAvailable;
    final isSubscribed = video.isSubscribed;
    final isExcluded = video.isExcluded;
    final activePlayingId = ref.watch(
      videoBoardProvider.select((s) => s.activePlayingVideoId),
    );
    final isThisCardActivePlayer =
        activePlayingId == video.id && _controller != null;

    Widget cardBody = Card(
      margin: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
      elevation: isOffline ? 4 : (isSubscribed ? 3 : 2),
      shape: RoundedRectangleBorder(
        borderRadius: BorderRadius.circular(14),
        side: BorderSide(
          color: isOffline
              ? Colors.green.shade600
              : (isSubscribed
                  ? Colors.red.shade500
                  : (isGreyedOutDownloading
                      ? Colors.grey.shade500
                      : Colors.grey.shade300)),
          width: (isOffline || isSubscribed) ? 2 : 1,
        ),
      ),
      clipBehavior: Clip.antiAlias,
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.stretch,
        children: <Widget>[
          // 1. Video Preview / Inline Player Area
          _buildMediaPreviewSection(
            video: video,
            isThisCardActivePlayer: isThisCardActivePlayer,
            isGreyedOutDownloading: isGreyedOutDownloading,
          ),

          // 2. Title & Metadata Header
          Padding(
            padding: const EdgeInsets.fromLTRB(12, 10, 12, 6),
            child: Row(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: <Widget>[
                Expanded(
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: <Widget>[
                      Text(
                        video.title,
                        maxLines: 2,
                        overflow: TextOverflow.ellipsis,
                        style: const TextStyle(
                          fontSize: 15,
                          fontWeight: FontWeight.w700,
                        ),
                      ),
                      const SizedBox(height: 4),
                      Text(
                        '${video.channelName} • 片長 ${_formatSeconds(video.durationSeconds)}',
                        style: TextStyle(
                          fontSize: 12,
                          color: Colors.grey.shade700,
                        ),
                      ),
                    ],
                  ),
                ),
                const SizedBox(width: 8),
                // 標記 Icon (離線 / 線上)
                _buildConnectionModeBadge(isOffline: isOffline, video: video),
              ],
            ),
          ),

          if (_playerError != null)
            Padding(
              padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 4),
              child: Text(
                _playerError!,
                style: const TextStyle(color: Colors.red, fontSize: 12),
              ),
            ),

          const Divider(height: 1),

          // 3. Card Bottom Action Controls
          Padding(
            padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 6),
            child: Wrap(
              spacing: 6,
              runSpacing: 6,
              alignment: WrapAlignment.spaceBetween,
              crossAxisAlignment: WrapCrossAlignment.center,
              children: <Widget>[
                // (A) 進入頻道按鈕 (點擊進入該頻道的所有影片清單)
                if (widget.showEnterChannelButton)
                  _ActionChipButton(
                    icon: Icons.video_collection_rounded,
                    label: '進入頻道',
                    color: Colors.blue.shade800,
                    onTap: _handleEnterChannel,
                  ),

                // (B) 訂閱按鈕 (直接訂閱 / 取消訂閱該頻道)
                _ActionChipButton(
                  icon: isSubscribed
                      ? Icons.notifications_active_rounded
                      : Icons.subscriptions_rounded,
                  label: isSubscribed ? '已訂閱' : '訂閱',
                  color: isSubscribed
                      ? Colors.red.shade800
                      : Colors.pink.shade700,
                  onTap: () => ref
                      .read(videoBoardProvider.notifier)
                      .setChannelSubscription(
                        video.channelName,
                        subscribe: !isSubscribed,
                      ),
                ),

                // (C) 排除按鈕 (將該影片頻道列入排除名單 / 取消排除)
                _ActionChipButton(
                  icon: isExcluded
                      ? Icons.check_circle_outline_rounded
                      : Icons.block_rounded,
                  label: isExcluded ? '取消排除' : '排除',
                  color: isExcluded
                      ? Colors.teal.shade700
                      : Colors.brown.shade700,
                  onTap: () => ref
                      .read(videoBoardProvider.notifier)
                      .setChannelExclusion(
                        video.channelName,
                        exclude: !isExcluded,
                      ),
                ),

                // (D) 標記已看過 (或還原未看過)
                if (!video.isWatched)
                  _ActionChipButton(
                    icon: Icons.visibility_off_outlined,
                    label: '標記已看過',
                    color: Colors.deepOrange.shade700,
                    onTap: () => ref
                        .read(videoBoardProvider.notifier)
                        .markAsWatched(video.id),
                  )
                else if (widget.showRestoreWatchedButton)
                  _ActionChipButton(
                    icon: Icons.restore,
                    label: '放回未看過',
                    color: Colors.teal.shade700,
                    onTap: () => ref
                        .read(videoBoardProvider.notifier)
                        .restoreWatched(video.id),
                  ),

                // (E) Download 按鈕 (按下即變灰並排入後台下載佇列)
                _ActionChipButton(
                  icon: isGreyedOutDownloading
                      ? Icons.downloading
                      : (isOffline
                          ? Icons.download_done
                          : Icons.download_rounded),
                  label: isGreyedOutDownloading
                      ? '下載中 ${(video.downloadProgress * 100).round()}%'
                      : (isOffline ? '已下載' : 'Download'),
                  color: isOffline
                      ? Colors.green.shade700
                      : Colors.indigo.shade700,
                  onTap: (isOffline || isGreyedOutDownloading)
                      ? null
                      : () => ref
                          .read(videoBoardProvider.notifier)
                          .startBackgroundDownload(video.id),
                ),

                // (F) 移除已下載影片但要紀錄已下載過
                _ActionChipButton(
                  icon: Icons.delete_sweep_outlined,
                  label: '移除已下載檔(保留紀錄)',
                  color: Colors.red.shade700,
                  onTap: isOffline
                      ? () => ref
                          .read(videoBoardProvider.notifier)
                          .removeLocalDownloadedVideo(video.id)
                      : null,
                ),

                // (G) 全螢幕播放
                _ActionChipButton(
                  icon: Icons.fullscreen_rounded,
                  label: '全螢幕播放',
                  color: Colors.purple.shade700,
                  onTap: _handleFullscreen,
                ),
              ],
            ),
          ),
        ],
      ),
    );

    // 當 download 按鈕按下時, 該卡片就灰灰的顯示 下載中圖示, 保存到 queue 中讓後台排程背景下載
    if (isGreyedOutDownloading) {
      cardBody = Stack(
        children: <Widget>[
          ColorFiltered(
            colorFilter: _greyscaleFilter,
            child: Opacity(
              opacity: 0.62,
              child: cardBody,
            ),
          ),
          Positioned(
            top: 20,
            right: 24,
            child: Container(
              padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
              decoration: BoxDecoration(
                color: Colors.black87,
                borderRadius: BorderRadius.circular(20),
                boxShadow: const <BoxShadow>[
                  BoxShadow(color: Colors.black26, blurRadius: 6),
                ],
              ),
              child: Row(
                mainAxisSize: MainAxisSize.min,
                children: <Widget>[
                  const SizedBox(
                    width: 18,
                    height: 18,
                    child: CircularProgressIndicator(
                      strokeWidth: 2.4,
                      color: Colors.amberAccent,
                    ),
                  ),
                  const SizedBox(width: 8),
                  Text(
                    video.downloadStatus == VideoDownloadStatus.queued
                        ? '背景佇列排隊下載中...'
                        : '背景下載中 ${(video.downloadProgress * 100).round()}%',
                    style: const TextStyle(
                      color: Colors.white,
                      fontSize: 12,
                      fontWeight: FontWeight.bold,
                    ),
                  ),
                ],
              ),
            ),
          ),
        ],
      );
    }

    return cardBody;
  }

  Widget _buildMediaPreviewSection({
    required VideoItem video,
    required bool isThisCardActivePlayer,
    required bool isGreyedOutDownloading,
  }) {
    if (_isInitializingPlayer) {
      return Container(
        height: 200,
        color: Colors.black87,
        alignment: Alignment.center,
        child: const Column(
          mainAxisSize: MainAxisSize.min,
          children: <Widget>[
            CircularProgressIndicator(color: Colors.white),
            SizedBox(height: 10),
            Text(
              '正在載入影片串流...',
              style: TextStyle(color: Colors.white70, fontSize: 13),
            ),
          ],
        ),
      );
    }

    if (isThisCardActivePlayer &&
        _controller != null &&
        _controller!.value.isInitialized) {
      return Container(
        color: Colors.black,
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: <Widget>[
            SizedBox(
              height: 210,
              child: Center(
                child: AspectRatio(
                  aspectRatio: _controller!.value.aspectRatio > 0
                      ? _controller!.value.aspectRatio
                      : (16 / 9),
                  child: GestureDetector(
                    onTap: () {
                      if (_controller!.value.isPlaying) {
                        _controller!.pause();
                      } else {
                        _controller!.play();
                      }
                    },
                    child: VideoPlayer(_controller!),
                  ),
                ),
              ),
            ),
            VideoProgressIndicator(
              _controller!,
              allowScrubbing: true,
              colors: const VideoProgressColors(
                playedColor: Colors.redAccent,
                bufferedColor: Colors.white38,
                backgroundColor: Colors.white24,
              ),
            ),
          ],
        ),
      );
    }

    return Stack(
      alignment: Alignment.center,
      children: <Widget>[
        Container(
          height: 190,
          width: double.infinity,
          color: Colors.grey.shade900,
          child: Image.network(
            video.thumbnailUrl,
            fit: BoxFit.cover,
            errorBuilder: (_, __, ___) => Center(
              child: Icon(
                Icons.smart_display_rounded,
                size: 64,
                color: Colors.red.shade400,
              ),
            ),
          ),
        ),
        // Priority badges (已下載優先置頂 / 已訂閱優先)
        Positioned(
          top: 10,
          left: 10,
          child: Row(
            mainAxisSize: MainAxisSize.min,
            children: <Widget>[
              if (video.isOfflineAvailable)
                Container(
                  margin: const EdgeInsets.only(right: 6),
                  padding:
                      const EdgeInsets.symmetric(horizontal: 10, vertical: 4),
                  decoration: BoxDecoration(
                    color: Colors.green.shade700,
                    borderRadius: BorderRadius.circular(8),
                  ),
                  child: const Row(
                    mainAxisSize: MainAxisSize.min,
                    children: <Widget>[
                      Icon(Icons.vertical_align_top,
                          size: 15, color: Colors.white),
                      SizedBox(width: 4),
                      Text(
                        '已下載優先',
                        style: TextStyle(
                          color: Colors.white,
                          fontSize: 11,
                          fontWeight: FontWeight.bold,
                        ),
                      ),
                    ],
                  ),
                ),
              if (video.isSubscribed)
                Container(
                  padding:
                      const EdgeInsets.symmetric(horizontal: 10, vertical: 4),
                  decoration: BoxDecoration(
                    color: Colors.red.shade700,
                    borderRadius: BorderRadius.circular(8),
                  ),
                  child: const Row(
                    mainAxisSize: MainAxisSize.min,
                    children: <Widget>[
                      Icon(Icons.notifications_active_rounded,
                          size: 14, color: Colors.white),
                      SizedBox(width: 4),
                      Text(
                        '已訂閱優先',
                        style: TextStyle(
                          color: Colors.white,
                          fontSize: 11,
                          fontWeight: FontWeight.bold,
                        ),
                      ),
                    ],
                  ),
                ),
            ],
          ),
        ),
        // Duration badge
        Positioned(
          bottom: 8,
          right: 8,
          child: Container(
            padding: const EdgeInsets.symmetric(horizontal: 6, vertical: 3),
            decoration: BoxDecoration(
              color: Colors.black87,
              borderRadius: BorderRadius.circular(4),
            ),
            child: Text(
              _formatSeconds(video.durationSeconds),
              style: const TextStyle(
                color: Colors.white,
                fontSize: 11,
                fontWeight: FontWeight.w600,
              ),
            ),
          ),
        ),
        // Center quick-play overlay icon
        Material(
          color: Colors.black45,
          shape: const CircleBorder(),
          child: InkWell(
            customBorder: const CircleBorder(),
            onTap: _handlePlay,
            child: const Padding(
              padding: EdgeInsets.all(12),
              child: Icon(
                Icons.play_arrow_rounded,
                size: 38,
                color: Colors.white,
              ),
            ),
          ),
        ),
      ],
    );
  }

  Widget _buildConnectionModeBadge({
    required bool isOffline,
    required VideoItem video,
  }) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
      decoration: BoxDecoration(
        color: isOffline ? Colors.green.shade50 : Colors.blue.shade50,
        borderRadius: BorderRadius.circular(8),
        border: Border.all(
          color: isOffline ? Colors.green.shade600 : Colors.blue.shade600,
        ),
      ),
      child: Row(
        mainAxisSize: MainAxisSize.min,
        children: <Widget>[
          Icon(
            isOffline ? Icons.offline_pin_rounded : Icons.wifi_rounded,
            size: 16,
            color: isOffline ? Colors.green.shade800 : Colors.blue.shade800,
          ),
          const SizedBox(width: 4),
          Text(
            isOffline ? '離線' : '線上',
            style: TextStyle(
              fontSize: 12,
              fontWeight: FontWeight.bold,
              color: isOffline ? Colors.green.shade800 : Colors.blue.shade800,
            ),
          ),
        ],
      ),
    );
  }
}

class _ActionChipButton extends StatelessWidget {
  const _ActionChipButton({
    required this.icon,
    required this.label,
    required this.color,
    this.onTap,
  });

  final IconData icon;
  final String label;
  final Color color;
  final VoidCallback? onTap;

  @override
  Widget build(BuildContext context) {
    final enabled = onTap != null;
    return Material(
      color: enabled ? color.withValues(alpha: 0.1) : Colors.grey.shade200,
      borderRadius: BorderRadius.circular(8),
      child: InkWell(
        borderRadius: BorderRadius.circular(8),
        onTap: onTap,
        child: Padding(
          padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 6),
          child: Row(
            mainAxisSize: MainAxisSize.min,
            children: <Widget>[
              Icon(
                icon,
                size: 16,
                color: enabled ? color : Colors.grey.shade500,
              ),
              const SizedBox(width: 4),
              Text(
                label,
                style: TextStyle(
                  fontSize: 12,
                  fontWeight: FontWeight.w600,
                  color: enabled ? color : Colors.grey.shade500,
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }
}
