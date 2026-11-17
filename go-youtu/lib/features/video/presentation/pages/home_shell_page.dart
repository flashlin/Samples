import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

import '../../../../core/constants/app_constants.dart';
import '../../domain/entities/video_item.dart';
import '../providers/video_providers.dart';
import '../widgets/video_card_widget.dart';

/// Main application page that opens directly into the YouTube preview card list
/// and displays the top bar with the 6 required sections:
/// [影片清單] [已下載過清單] [訂閱清單] [下載佇列清單] [已看過清單] [排除頻道]
class HomeShellPage extends ConsumerStatefulWidget {
  const HomeShellPage({super.key});

  @override
  ConsumerState<HomeShellPage> createState() => _HomeShellPageState();
}

class _HomeShellPageState extends ConsumerState<HomeShellPage> {
  final TextEditingController _searchController = TextEditingController();

  @override
  void dispose() {
    _searchController.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final boardState = ref.watch(videoBoardProvider);
    final notifier = ref.read(videoBoardProvider.notifier);

    final feedList = notifier.activeFeedVideos;
    final downloadedHistoryList = notifier.downloadedHistoryVideos;
    final subscribedList = notifier.subscribedVideos;
    final queueList = notifier.downloadQueueVideos;
    final watchedList = notifier.watchedHistoryVideos;
    final excludedList = notifier.excludedVideos;

    return Scaffold(
      backgroundColor: Colors.grey.shade100,
      appBar: AppBar(
        backgroundColor: Colors.red.shade700,
        foregroundColor: Colors.white,
        elevation: 2,
        titleSpacing: 12,
        title: const Row(
          children: <Widget>[
            Icon(Icons.play_circle_fill_rounded, size: 28),
            SizedBox(width: 8),
            Text(
              'Go-Youtu 影音離線快取',
              style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold),
            ),
          ],
        ),
        actions: <Widget>[
          IconButton(
            icon: const Icon(Icons.refresh_rounded),
            tooltip: '重新整理最新預覽清單',
            onPressed: boardState.isLoading
                ? null
                : () => notifier.initializeFeed(
                      query: _searchController.text.trim(),
                    ),
          ),
        ],
        // 應用程式最上方顯示一列功能: [影片清單] [已下載過清單] [訂閱清單] [下載佇列清單] [已看過清單] [排除頻道]
        bottom: PreferredSize(
          preferredSize: const Size.fromHeight(54),
          child: Container(
            color: Colors.red.shade800,
            padding: const EdgeInsets.symmetric(horizontal: 6, vertical: 6),
            child: SingleChildScrollView(
              scrollDirection: Axis.horizontal,
              child: Row(
                children: <Widget>[
                  _TopSectionButton(
                    label: '[${TopNavigationTab.videoFeed.label}]',
                    count: feedList.length,
                    isSelected:
                        boardState.activeTab == TopNavigationTab.videoFeed,
                    onTap: () => notifier.selectTab(TopNavigationTab.videoFeed),
                  ),
                  const SizedBox(width: 6),
                  _TopSectionButton(
                    label: '[${TopNavigationTab.downloadedHistory.label}]',
                    count: downloadedHistoryList.length,
                    isSelected: boardState.activeTab ==
                        TopNavigationTab.downloadedHistory,
                    onTap: () =>
                        notifier.selectTab(TopNavigationTab.downloadedHistory),
                  ),
                  const SizedBox(width: 6),
                  _TopSectionButton(
                    label: '[${TopNavigationTab.subscribedList.label}]',
                    count: subscribedList.length,
                    isSelected:
                        boardState.activeTab == TopNavigationTab.subscribedList,
                    onTap: () =>
                        notifier.selectTab(TopNavigationTab.subscribedList),
                  ),
                  const SizedBox(width: 6),
                  _TopSectionButton(
                    label: '[${TopNavigationTab.downloadQueue.label}]',
                    count: queueList.length,
                    isSelected:
                        boardState.activeTab == TopNavigationTab.downloadQueue,
                    onTap: () =>
                        notifier.selectTab(TopNavigationTab.downloadQueue),
                  ),
                  const SizedBox(width: 6),
                  _TopSectionButton(
                    label: '[${TopNavigationTab.watchedHistory.label}]',
                    count: watchedList.length,
                    isSelected:
                        boardState.activeTab == TopNavigationTab.watchedHistory,
                    onTap: () =>
                        notifier.selectTab(TopNavigationTab.watchedHistory),
                  ),
                  const SizedBox(width: 6),
                  _TopSectionButton(
                    label: '[${TopNavigationTab.excludedChannels.label}]',
                    count: excludedList.map((e) => e.channelName).toSet().length,
                    isSelected: boardState.activeTab ==
                        TopNavigationTab.excludedChannels,
                    onTap: () =>
                        notifier.selectTab(TopNavigationTab.excludedChannels),
                  ),
                ],
              ),
            ),
          ),
        ),
      ),
      body: Column(
        children: <Widget>[
          if (boardState.isLoading)
            const LinearProgressIndicator(minHeight: 3),
          if (boardState.errorMessage != null)
            Container(
              width: double.infinity,
              color: Colors.amber.shade100,
              padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
              child: Text(
                boardState.errorMessage!,
                style: TextStyle(color: Colors.brown.shade900, fontSize: 12),
              ),
            ),
          Expanded(
            child: switch (boardState.activeTab) {
              TopNavigationTab.videoFeed => _buildVideoFeedTab(
                  feedList: feedList,
                  notifier: notifier,
                ),
              TopNavigationTab.downloadedHistory => _buildDownloadedHistoryTab(
                  downloadedHistoryList: downloadedHistoryList,
                  notifier: notifier,
                ),
              TopNavigationTab.subscribedList => _buildSubscribedTab(
                  subscribedList: subscribedList,
                  notifier: notifier,
                ),
              TopNavigationTab.downloadQueue => _buildDownloadQueueTab(
                  queueList: queueList,
                  notifier: notifier,
                ),
              TopNavigationTab.watchedHistory => _buildWatchedHistoryTab(
                  watchedList: watchedList,
                ),
              TopNavigationTab.excludedChannels => _buildExcludedChannelsTab(
                  excludedList: excludedList,
                  notifier: notifier,
                ),
            },
          ),
        ],
      ),
    );
  }

  /// Tab 1: [影片清單] (已下載優先 / 已訂閱優先 / 已看過與已排除頻道的影片不顯示)
  Widget _buildVideoFeedTab({
    required List<VideoItem> feedList,
    required VideoBoardNotifier notifier,
  }) {
    return Column(
      children: <Widget>[
        Padding(
          padding: const EdgeInsets.fromLTRB(12, 10, 12, 4),
          child: Row(
            children: <Widget>[
              Expanded(
                child: TextField(
                  controller: _searchController,
                  decoration: InputDecoration(
                    hintText: '搜尋 YouTube 最新預覽關鍵字...',
                    prefixIcon: const Icon(Icons.search, size: 20),
                    suffixIcon: Row(
                      mainAxisSize: MainAxisSize.min,
                      children: <Widget>[
                        IconButton(
                          icon:
                              const Icon(Icons.content_paste_rounded, size: 18),
                          tooltip: '貼上剪貼簿文字 (支援 Mac 複製中文直接貼上)',
                          onPressed: () async {
                            final data =
                                await Clipboard.getData(Clipboard.kTextPlain);
                            final text = data?.text?.trim();
                            if (text != null && text.isNotEmpty) {
                              _searchController.text = text;
                              _searchController.selection =
                                  TextSelection.fromPosition(
                                TextPosition(
                                    offset: _searchController.text.length),
                              );
                            }
                          },
                        ),
                        IconButton(
                          icon: const Icon(Icons.clear_rounded, size: 18),
                          tooltip: '清除搜尋文字',
                          onPressed: () => _searchController.clear(),
                        ),
                      ],
                    ),
                    isDense: true,
                    filled: true,
                    fillColor: Colors.white,
                    contentPadding: const EdgeInsets.symmetric(
                      horizontal: 12,
                      vertical: 10,
                    ),
                    border: OutlineInputBorder(
                      borderRadius: BorderRadius.circular(10),
                      borderSide: BorderSide(color: Colors.grey.shade300),
                    ),
                  ),
                  onSubmitted: (value) =>
                      notifier.initializeFeed(query: value.trim()),
                ),
              ),
              const SizedBox(width: 8),
              ElevatedButton(
                style: ElevatedButton.styleFrom(
                  backgroundColor: Colors.red.shade700,
                  foregroundColor: Colors.white,
                ),
                onPressed: () => notifier.initializeFeed(
                  query: _searchController.text.trim(),
                ),
                child: const Text('載入'),
              ),
            ],
          ),
        ),
        if (feedList.isEmpty)
          const Expanded(
            child: Center(
              child: Text(
                '目前沒有可顯示的預覽影片，請點擊重新整理或輸入關鍵字搜尋。',
                style: TextStyle(color: Colors.black54),
              ),
            ),
          )
        else
          Expanded(
            child: RefreshIndicator(
              onRefresh: () => notifier.initializeFeed(
                query: _searchController.text.trim(),
              ),
              child: ListView.builder(
                padding: const EdgeInsets.only(bottom: 24),
                itemCount: feedList.length,
                itemBuilder: (context, index) {
                  final video = feedList[index];
                  return VideoCardWidget(
                    key: ValueKey<String>('feed_${video.id}'),
                    video: video,
                  );
                },
              ),
            ),
          ),
      ],
    );
  }

  /// Tab 2: [已下載過清單]
  Widget _buildDownloadedHistoryTab({
    required List<VideoItem> downloadedHistoryList,
    required VideoBoardNotifier notifier,
  }) {
    if (downloadedHistoryList.isEmpty) {
      return const Center(
        child: Text(
          '尚無任何已下載過的影片紀錄。',
          style: TextStyle(color: Colors.black54, fontSize: 15),
        ),
      );
    }

    return ListView.builder(
      padding: const EdgeInsets.all(12),
      itemCount: downloadedHistoryList.length,
      itemBuilder: (context, index) {
        final item = downloadedHistoryList[index];
        final isStillOnDisk = item.isOfflineAvailable;

        return Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: <Widget>[
            Container(
              margin: const EdgeInsets.symmetric(horizontal: 12),
              padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 6),
              decoration: BoxDecoration(
                color: isStillOnDisk
                    ? Colors.green.shade100
                    : Colors.orange.shade100,
                borderRadius: const BorderRadius.vertical(
                  top: Radius.circular(10),
                ),
              ),
              child: Row(
                mainAxisAlignment: MainAxisAlignment.spaceBetween,
                children: <Widget>[
                  Row(
                    children: <Widget>[
                      Icon(
                        isStillOnDisk
                            ? Icons.check_circle_rounded
                            : Icons.history_rounded,
                        size: 16,
                        color: isStillOnDisk
                            ? Colors.green.shade800
                            : Colors.orange.shade900,
                      ),
                      const SizedBox(width: 6),
                      Text(
                        isStillOnDisk
                            ? '狀態：已下載（實體離線檔案存在）'
                            : '狀態：已下載過（實體檔案已移除，保留歷史紀錄）',
                        style: TextStyle(
                          fontSize: 12,
                          fontWeight: FontWeight.bold,
                          color: isStillOnDisk
                              ? Colors.green.shade900
                              : Colors.orange.shade900,
                        ),
                      ),
                    ],
                  ),
                  if (!isStillOnDisk && !item.isDownloadingOrQueued)
                    TextButton.icon(
                      style: TextButton.styleFrom(
                        padding: const EdgeInsets.symmetric(horizontal: 8),
                        minimumSize: const Size(0, 28),
                      ),
                      onPressed: () =>
                          notifier.startBackgroundDownload(item.id),
                      icon: const Icon(Icons.download_rounded, size: 16),
                      label: const Text('重新下載', style: TextStyle(fontSize: 12)),
                    ),
                ],
              ),
            ),
            VideoCardWidget(
              key: ValueKey<String>('dl_history_${item.id}'),
              video: item,
              showRestoreWatchedButton: true,
            ),
          ],
        );
      },
    );
  }

  /// Tab 3: [訂閱清單]
  Widget _buildSubscribedTab({
    required List<VideoItem> subscribedList,
    required VideoBoardNotifier notifier,
  }) {
    final channels = subscribedList
        .map((e) => e.channelName)
        .toSet()
        .toList()
      ..sort();

    if (subscribedList.isEmpty) {
      return const Center(
        child: Text(
          '尚無訂閱任何頻道，請在影片卡片下方點擊「訂閱」按鈕直接訂閱。',
          style: TextStyle(color: Colors.black54, fontSize: 15),
        ),
      );
    }

    return Column(
      crossAxisAlignment: CrossAxisAlignment.stretch,
      children: <Widget>[
        Container(
          color: Colors.red.shade50,
          padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: <Widget>[
              Text(
                '已訂閱頻道 (${channels.length})：',
                style: TextStyle(
                  fontSize: 12,
                  fontWeight: FontWeight.bold,
                  color: Colors.red.shade900,
                ),
              ),
              const SizedBox(height: 6),
              Wrap(
                spacing: 8,
                runSpacing: 6,
                children: channels.map((channel) {
                  return Chip(
                    label: Text(
                      channel,
                      style: const TextStyle(
                        fontSize: 12,
                        fontWeight: FontWeight.w600,
                      ),
                    ),
                    backgroundColor: Colors.white,
                    deleteIcon: const Icon(Icons.close, size: 16),
                    deleteButtonTooltipMessage: '取消訂閱此頻道',
                    onDeleted: () => notifier.setChannelSubscription(
                      channel,
                      subscribe: false,
                    ),
                  );
                }).toList(),
              ),
            ],
          ),
        ),
        Expanded(
          child: ListView.builder(
            padding: const EdgeInsets.only(bottom: 24),
            itemCount: subscribedList.length,
            itemBuilder: (context, index) {
              final video = subscribedList[index];
              return VideoCardWidget(
                key: ValueKey<String>('sub_${video.id}'),
                video: video,
                showRestoreWatchedButton: true,
              );
            },
          ),
        ),
      ],
    );
  }

  /// Tab 4: [下載佇列清單]
  Widget _buildDownloadQueueTab({
    required List<VideoItem> queueList,
    required VideoBoardNotifier notifier,
  }) {
    if (queueList.isEmpty) {
      return const Center(
        child: Text(
          '目前背景下載佇列沒有排程中的任務。',
          style: TextStyle(color: Colors.black54, fontSize: 15),
        ),
      );
    }

    return ListView.builder(
      padding: const EdgeInsets.all(12),
      itemCount: queueList.length,
      itemBuilder: (context, index) {
        final item = queueList[index];
        final pct = (item.downloadProgress * 100).round();
        final statusLabel = switch (item.downloadStatus) {
          VideoDownloadStatus.downloading => '正在背景下載中 ($pct%)',
          VideoDownloadStatus.queued => '排隊等候背景下載中...',
          VideoDownloadStatus.failed => '下載失敗，可點擊重試',
          _ => item.downloadStatus.name,
        };

        return Card(
          margin: const EdgeInsets.only(bottom: 10),
          child: ListTile(
            leading: const SizedBox(
              width: 36,
              height: 36,
              child: Center(
                child: CircularProgressIndicator(strokeWidth: 2.5),
              ),
            ),
            title: Text(
              item.title,
              maxLines: 1,
              overflow: TextOverflow.ellipsis,
              style: const TextStyle(fontWeight: FontWeight.w600),
            ),
            subtitle: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: <Widget>[
                const SizedBox(height: 4),
                Text(statusLabel, style: const TextStyle(fontSize: 12)),
                const SizedBox(height: 6),
                LinearProgressIndicator(value: item.downloadProgress),
              ],
            ),
            trailing: IconButton(
              icon: const Icon(Icons.cancel_outlined, color: Colors.red),
              tooltip: '取消排程下載',
              onPressed: () => notifier.cancelDownload(item.id),
            ),
          ),
        );
      },
    );
  }

  /// Tab 5: [已看過清單] (最多紀錄 1 萬筆資訊)
  Widget _buildWatchedHistoryTab({
    required List<VideoItem> watchedList,
  }) {
    return Column(
      children: <Widget>[
        Container(
          width: double.infinity,
          padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 8),
          color: Colors.grey.shade200,
          child: Text(
            '已看過紀錄：共 ${watchedList.length} / ${AppConstants.maxStoredRecords} 筆（超過上限自動淘汰最舊紀錄）',
            style: TextStyle(fontSize: 12, color: Colors.grey.shade800),
          ),
        ),
        if (watchedList.isEmpty)
          const Expanded(
            child: Center(
              child: Text(
                '尚無標記為已看過的影片。',
                style: TextStyle(color: Colors.black54, fontSize: 15),
              ),
            ),
          )
        else
          Expanded(
            child: ListView.builder(
              padding: const EdgeInsets.only(bottom: 24),
              itemCount: watchedList.length,
              itemBuilder: (context, index) {
                final video = watchedList[index];
                return VideoCardWidget(
                  key: ValueKey<String>('watched_${video.id}'),
                  video: video,
                  showRestoreWatchedButton: true,
                );
              },
            ),
          ),
      ],
    );
  }

  /// Tab 6: [排除頻道] (顯示被列入排除名單的頻道與其被隱藏的影片，支援移除排除)
  Widget _buildExcludedChannelsTab({
    required List<VideoItem> excludedList,
    required VideoBoardNotifier notifier,
  }) {
    final channels = excludedList
        .map((e) => e.channelName)
        .toSet()
        .toList()
      ..sort();

    if (excludedList.isEmpty) {
      return const Center(
        child: Text(
          '目前沒有任何被排除的頻道。可在影片卡片下方點擊「排除」將該頻道列入排除名單。',
          style: TextStyle(color: Colors.black54, fontSize: 15),
          textAlign: TextAlign.center,
        ),
      );
    }

    return Column(
      crossAxisAlignment: CrossAxisAlignment.stretch,
      children: <Widget>[
        Container(
          color: Colors.brown.shade50,
          padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: <Widget>[
              Text(
                '已排除頻道名單 (${channels.length}) — 點擊 ✕ 可恢復顯示該頻道：',
                style: TextStyle(
                  fontSize: 12,
                  fontWeight: FontWeight.bold,
                  color: Colors.brown.shade900,
                ),
              ),
              const SizedBox(height: 6),
              Wrap(
                spacing: 8,
                runSpacing: 6,
                children: channels.map((channel) {
                  return Chip(
                    avatar: Icon(
                      Icons.block_rounded,
                      size: 16,
                      color: Colors.brown.shade700,
                    ),
                    label: Text(
                      channel,
                      style: const TextStyle(
                        fontSize: 12,
                        fontWeight: FontWeight.w600,
                      ),
                    ),
                    backgroundColor: Colors.white,
                    deleteIcon: const Icon(Icons.close, size: 16),
                    deleteButtonTooltipMessage: '從排除名單移除',
                    onDeleted: () => notifier.setChannelExclusion(
                      channel,
                      exclude: false,
                    ),
                  );
                }).toList(),
              ),
            ],
          ),
        ),
        Expanded(
          child: ListView.builder(
            padding: const EdgeInsets.only(bottom: 24),
            itemCount: excludedList.length,
            itemBuilder: (context, index) {
              final video = excludedList[index];
              return VideoCardWidget(
                key: ValueKey<String>('excluded_${video.id}'),
                video: video,
                showRestoreWatchedButton: true,
              );
            },
          ),
        ),
      ],
    );
  }
}

class _TopSectionButton extends StatelessWidget {
  const _TopSectionButton({
    required this.label,
    required this.count,
    required this.isSelected,
    required this.onTap,
  });

  final String label;
  final int count;
  final bool isSelected;
  final VoidCallback onTap;

  @override
  Widget build(BuildContext context) {
    return Material(
      color: isSelected ? Colors.white : Colors.red.shade900,
      borderRadius: BorderRadius.circular(10),
      child: InkWell(
        borderRadius: BorderRadius.circular(10),
        onTap: onTap,
        child: Padding(
          padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
          child: Row(
            mainAxisSize: MainAxisSize.min,
            children: <Widget>[
              Text(
                label,
                style: TextStyle(
                  fontSize: 13,
                  fontWeight: FontWeight.bold,
                  color: isSelected ? Colors.red.shade900 : Colors.white,
                ),
              ),
              const SizedBox(width: 6),
              Container(
                padding: const EdgeInsets.symmetric(horizontal: 6, vertical: 2),
                decoration: BoxDecoration(
                  color: isSelected ? Colors.red.shade700 : Colors.white24,
                  borderRadius: BorderRadius.circular(10),
                ),
                child: Text(
                  '$count',
                  style: const TextStyle(
                    fontSize: 11,
                    fontWeight: FontWeight.bold,
                    color: Colors.white,
                  ),
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }
}
