import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:video_player/video_player.dart';

import '../../../../core/constants/app_constants.dart';
import '../../domain/entities/video_item.dart';

/// Fullscreen video player view supporting play, pause, +30s, -30s, and progress scrubbing.
/// Controls (top title bar & bottom playback progress menu) are hidden by default
/// upon entering fullscreen, and toggle visibility whenever the user taps the screen.
class FullscreenPlayerPage extends StatefulWidget {
  const FullscreenPlayerPage({
    super.key,
    required this.video,
    required this.controller,
  });

  final VideoItem video;
  final VideoPlayerController controller;

  @override
  State<FullscreenPlayerPage> createState() => _FullscreenPlayerPageState();
}

class _FullscreenPlayerPageState extends State<FullscreenPlayerPage> {
  /// Hidden by default when entering fullscreen so the video has unobstructed space.
  bool _showControls = false;

  @override
  void initState() {
    super.initState();
    SystemChrome.setEnabledSystemUIMode(SystemUiMode.immersiveSticky);
    SystemChrome.setPreferredOrientations(const <DeviceOrientation>[
      DeviceOrientation.landscapeLeft,
      DeviceOrientation.landscapeRight,
    ]);
    widget.controller.addListener(_onControllerTick);
  }

  @override
  void dispose() {
    widget.controller.removeListener(_onControllerTick);
    SystemChrome.setEnabledSystemUIMode(SystemUiMode.edgeToEdge);
    SystemChrome.setPreferredOrientations(const <DeviceOrientation>[
      DeviceOrientation.portraitUp,
      DeviceOrientation.portraitDown,
    ]);
    super.dispose();
  }

  void _onControllerTick() {
    if (mounted && _showControls) {
      setState(() {});
    }
  }

  void _toggleControlsVisibility() {
    setState(() {
      _showControls = !_showControls;
    });
  }

  Future<void> _seekRelative(Duration offset) async {
    final value = widget.controller.value;
    if (!value.isInitialized) {
      return;
    }
    final target = value.position + offset;
    final clamped = target < Duration.zero
        ? Duration.zero
        : (target > value.duration ? value.duration : target);
    await widget.controller.seekTo(clamped);
  }

  String _formatDuration(Duration d) {
    final minutes = d.inMinutes.remainder(60).toString().padLeft(2, '0');
    final seconds = d.inSeconds.remainder(60).toString().padLeft(2, '0');
    if (d.inHours > 0) {
      return '${d.inHours}:$minutes:$seconds';
    }
    return '$minutes:$seconds';
  }

  @override
  Widget build(BuildContext context) {
    final value = widget.controller.value;

    return Scaffold(
      backgroundColor: Colors.black,
      body: SafeArea(
        child: Stack(
          alignment: Alignment.center,
          children: <Widget>[
            // Fullscreen tap layer + Video Player
            GestureDetector(
              behavior: HitTestBehavior.opaque,
              onTap: _toggleControlsVisibility,
              child: SizedBox.expand(
                child: Center(
                  child: value.isInitialized
                      ? AspectRatio(
                          aspectRatio: value.aspectRatio > 0
                              ? value.aspectRatio
                              : (16 / 9),
                          child: VideoPlayer(widget.controller),
                        )
                      : const CircularProgressIndicator(color: Colors.white),
                ),
              ),
            ),

            // Top bar with title, offline/online badge, and exit button (only shown when _showControls is true)
            if (_showControls)
              Positioned(
                top: 12,
                left: 16,
                right: 16,
                child: Container(
                  padding:
                      const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
                  decoration: BoxDecoration(
                    color: Colors.black54,
                    borderRadius: BorderRadius.circular(12),
                  ),
                  child: Row(
                    children: <Widget>[
                      IconButton(
                        icon: const Icon(
                          Icons.fullscreen_exit,
                          color: Colors.white,
                          size: 30,
                        ),
                        tooltip: '退出全螢幕',
                        onPressed: () => Navigator.of(context).pop(),
                      ),
                      const SizedBox(width: 8),
                      Expanded(
                        child: GestureDetector(
                          onTap: _toggleControlsVisibility,
                          child: Text(
                            widget.video.title,
                            maxLines: 1,
                            overflow: TextOverflow.ellipsis,
                            style: const TextStyle(
                              color: Colors.white,
                              fontSize: 16,
                              fontWeight: FontWeight.w600,
                            ),
                          ),
                        ),
                      ),
                      const SizedBox(width: 8),
                      Container(
                        padding: const EdgeInsets.symmetric(
                          horizontal: 10,
                          vertical: 4,
                        ),
                        decoration: BoxDecoration(
                          color: widget.video.isOfflineAvailable
                              ? Colors.green.shade700
                              : Colors.blue.shade700,
                          borderRadius: BorderRadius.circular(12),
                        ),
                        child: Row(
                          mainAxisSize: MainAxisSize.min,
                          children: <Widget>[
                            Icon(
                              widget.video.isOfflineAvailable
                                  ? Icons.offline_pin
                                  : Icons.cloud_queue,
                              size: 16,
                              color: Colors.white,
                            ),
                            const SizedBox(width: 4),
                            Text(
                              widget.video.isOfflineAvailable ? '離線播放' : '線上串流',
                              style: const TextStyle(
                                color: Colors.white,
                                fontSize: 12,
                                fontWeight: FontWeight.bold,
                              ),
                            ),
                          ],
                        ),
                      ),
                    ],
                  ),
                ),
              ),

            // Bottom control bar: Play, Pause, -30s, +30s, Progress (only shown when _showControls is true)
            if (_showControls)
              Positioned(
                bottom: 12,
                left: 16,
                right: 16,
                child: Container(
                  padding:
                      const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
                  decoration: BoxDecoration(
                    color: Colors.black87,
                    borderRadius: BorderRadius.circular(12),
                  ),
                  child: Column(
                    mainAxisSize: MainAxisSize.min,
                    children: <Widget>[
                      VideoProgressIndicator(
                        widget.controller,
                        allowScrubbing: true,
                        colors: const VideoProgressColors(
                          playedColor: Colors.redAccent,
                          bufferedColor: Colors.white38,
                          backgroundColor: Colors.white12,
                        ),
                      ),
                      const SizedBox(height: 8),
                      Row(
                        mainAxisAlignment: MainAxisAlignment.spaceBetween,
                        children: <Widget>[
                          Text(
                            '${_formatDuration(value.position)} / ${_formatDuration(value.duration)}',
                            style: const TextStyle(
                              color: Colors.white70,
                              fontSize: 12,
                            ),
                          ),
                          Row(
                            children: <Widget>[
                              IconButton(
                                icon: const Icon(
                                  Icons.replay_30,
                                  color: Colors.white,
                                ),
                                tooltip: '倒退 30 秒',
                                onPressed: () => _seekRelative(
                                  -AppConstants.seekStepDuration,
                                ),
                              ),
                              IconButton(
                                icon: const Icon(
                                  Icons.play_arrow,
                                  color: Colors.greenAccent,
                                ),
                                tooltip: 'Play 播放',
                                onPressed: () => widget.controller.play(),
                              ),
                              IconButton(
                                icon: const Icon(
                                  Icons.pause,
                                  color: Colors.amberAccent,
                                ),
                                tooltip: 'Pause 暫停',
                                onPressed: () => widget.controller.pause(),
                              ),
                              IconButton(
                                icon: const Icon(
                                  Icons.forward_30,
                                  color: Colors.white,
                                ),
                                tooltip: '前進 30 秒',
                                onPressed: () => _seekRelative(
                                  AppConstants.seekStepDuration,
                                ),
                              ),
                            ],
                          ),
                          TextButton.icon(
                            onPressed: () => Navigator.of(context).pop(),
                            icon: const Icon(
                              Icons.fullscreen_exit,
                              color: Colors.white,
                            ),
                            label: const Text(
                              '結束全螢幕',
                              style: TextStyle(color: Colors.white),
                            ),
                          ),
                        ],
                      ),
                    ],
                  ),
                ),
              ),
          ],
        ),
      ),
    );
  }
}
