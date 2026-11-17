import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

import '../providers/video_providers.dart';
import '../widgets/video_card_widget.dart';

/// Page opened when the user taps [進入頻道] on a video card.
/// Displays all videos from [channelName] and fetches additional videos for that channel.
class ChannelVideosPage extends ConsumerStatefulWidget {
  const ChannelVideosPage({
    super.key,
    required this.channelName,
  });

  final String channelName;

  @override
  ConsumerState<ChannelVideosPage> createState() => _ChannelVideosPageState();
}

class _ChannelVideosPageState extends ConsumerState<ChannelVideosPage> {
  @override
  void initState() {
    super.initState();
    Future<void>.microtask(() {
      ref
          .read(videoBoardProvider.notifier)
          .loadChannelVideos(widget.channelName);
    });
  }

  @override
  Widget build(BuildContext context) {
    final boardState = ref.watch(videoBoardProvider);
    final notifier = ref.read(videoBoardProvider.notifier);
    final channelVideos = notifier.getVideosForChannel(widget.channelName);

    final isSubscribed =
        channelVideos.isNotEmpty && channelVideos.first.isSubscribed;
    final isExcluded =
        channelVideos.isNotEmpty && channelVideos.first.isExcluded;

    return Scaffold(
      backgroundColor: Colors.grey.shade100,
      appBar: AppBar(
        backgroundColor: Colors.red.shade700,
        foregroundColor: Colors.white,
        title: Row(
          children: <Widget>[
            const Icon(Icons.video_collection_rounded, size: 22),
            const SizedBox(width: 8),
            Expanded(
              child: Text(
                widget.channelName,
                maxLines: 1,
                overflow: TextOverflow.ellipsis,
                style: const TextStyle(
                  fontSize: 17,
                  fontWeight: FontWeight.bold,
                ),
              ),
            ),
          ],
        ),
        actions: <Widget>[
          IconButton(
            icon: const Icon(Icons.refresh_rounded),
            tooltip: '重新整理此頻道影片',
            onPressed: boardState.isLoading
                ? null
                : () => notifier.loadChannelVideos(widget.channelName),
          ),
        ],
      ),
      body: Column(
        children: <Widget>[
          if (boardState.isLoading)
            const LinearProgressIndicator(minHeight: 3),
          // Channel header bar with video count, subscribe, and exclude actions
          Container(
            color: Colors.white,
            padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 10),
            child: Row(
              mainAxisAlignment: MainAxisAlignment.spaceBetween,
              children: <Widget>[
                Expanded(
                  child: Text(
                    '頻道影片清單（共 ${channelVideos.length} 部）',
                    style: const TextStyle(
                      fontSize: 13,
                      fontWeight: FontWeight.w700,
                    ),
                  ),
                ),
                Row(
                  mainAxisSize: MainAxisSize.min,
                  children: <Widget>[
                    FilledButton.tonalIcon(
                      onPressed: () => notifier.setChannelSubscription(
                        widget.channelName,
                        subscribe: !isSubscribed,
                      ),
                      icon: Icon(
                        isSubscribed
                            ? Icons.notifications_active_rounded
                            : Icons.subscriptions_rounded,
                        size: 16,
                      ),
                      label: Text(
                        isSubscribed ? '已訂閱頻道' : '訂閱頻道',
                        style: const TextStyle(fontSize: 12),
                      ),
                    ),
                    const SizedBox(width: 8),
                    OutlinedButton.icon(
                      onPressed: () => notifier.setChannelExclusion(
                        widget.channelName,
                        exclude: !isExcluded,
                      ),
                      icon: Icon(
                        isExcluded
                            ? Icons.check_circle_outline_rounded
                            : Icons.block_rounded,
                        size: 16,
                      ),
                      label: Text(
                        isExcluded ? '取消排除' : '排除頻道',
                        style: const TextStyle(fontSize: 12),
                      ),
                    ),
                  ],
                ),
              ],
            ),
          ),
          const Divider(height: 1),
          if (channelVideos.isEmpty && !boardState.isLoading)
            const Expanded(
              child: Center(
                child: Text(
                  '目前找不到此頻道的影片，請點擊右上角重新整理。',
                  style: TextStyle(color: Colors.black54),
                ),
              ),
            )
          else
            Expanded(
              child: RefreshIndicator(
                onRefresh: () => notifier.loadChannelVideos(widget.channelName),
                child: ListView.builder(
                  padding: const EdgeInsets.only(bottom: 24),
                  itemCount: channelVideos.length,
                  itemBuilder: (context, index) {
                    final video = channelVideos[index];
                    return VideoCardWidget(
                      key: ValueKey<String>('channel_${video.id}'),
                      video: video,
                      showRestoreWatchedButton: true,
                      showEnterChannelButton: false,
                    );
                  },
                ),
              ),
            ),
        ],
      ),
    );
  }
}
