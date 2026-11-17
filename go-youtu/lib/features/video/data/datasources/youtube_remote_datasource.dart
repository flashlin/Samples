import 'dart:math';

import 'package:youtube_explode_dart/youtube_explode_dart.dart';

import '../../../../core/constants/app_constants.dart';
import '../../../../core/error/app_exceptions.dart';
import '../models/video_item_model.dart';

/// Contract for fetching YouTube video previews, channel videos, and stream URLs.
abstract interface class RemoteYoutubeDataSource {
  /// Fetches preview videos. When [query] is null/empty, mixes random discovery
  /// topics with [interestedChannels] (from subscribed/downloaded channels).
  Future<List<VideoItemModel>> fetchPreviewVideos({
    String? query,
    List<String> interestedChannels = const <String>[],
  });

  /// Fetches a list of videos specifically belonging to [channelName].
  Future<List<VideoItemModel>> fetchVideosForChannel(String channelName);

  /// Resolves a direct playable stream URL for a video.
  Future<String> resolvePlayableStreamUrl(String videoId);
}

/// Implementation using `youtube_explode_dart` (no API key required).
/// Includes randomized discovery + interest-based recommendations and a rich fallback catalog.
class YoutubeExplodeRemoteDataSource implements RemoteYoutubeDataSource {
  YoutubeExplodeRemoteDataSource({
    YoutubeExplode? client,
    Random? random,
  })  : _yt = client ?? YoutubeExplode(),
        _random = random ?? Random();

  final YoutubeExplode _yt;
  final Random _random;

  @override
  Future<List<VideoItemModel>> fetchPreviewVideos({
    String? query,
    List<String> interestedChannels = const <String>[],
  }) async {
    final now = DateTime.now();
    final selectedQueries = _buildRefreshQueries(
      customQuery: query,
      interestedChannels: interestedChannels,
    );

    final fetched = <String, VideoItemModel>{};

    try {
      for (final q in selectedQueries) {
        final searchList = await _yt.search.search(q);
        for (final video in searchList.take(8)) {
          final id = video.id.value;
          if (fetched.containsKey(id)) {
            continue;
          }
          final publishedAt = video.uploadDate ??
              now.subtract(
                Duration(
                  minutes: _random.nextInt(720) + fetched.length * 15,
                ),
              );
          fetched[id] = VideoItemModel(
            id: id,
            title: video.title,
            channelName: video.author.isNotEmpty ? video.author : 'YouTube Creator',
            thumbnailUrl: video.thumbnails.highResUrl,
            durationSeconds: video.duration?.inSeconds ?? 240,
            publishedAt: publishedAt,
            updatedAt: now,
          );
        }
      }
    } catch (_) {
      // Fallback to randomized curated catalog if offline or rate-limited
    }

    if (fetched.isEmpty) {
      for (final fallback in _buildFallbackPreviewCatalog(
        now,
        interestedChannels: interestedChannels,
      )) {
        fetched[fallback.id] = fallback;
      }
    }

    return fetched.values.toList();
  }

  @override
  Future<List<VideoItemModel>> fetchVideosForChannel(String channelName) async {
    final now = DateTime.now();
    final fetched = <String, VideoItemModel>{};

    try {
      final searchList = await _yt.search.search('$channelName official videos');
      for (final video in searchList.take(15)) {
        final id = video.id.value;
        if (fetched.containsKey(id)) {
          continue;
        }
        final author =
            video.author.isNotEmpty ? video.author : channelName;
        // Prefer videos matching the channel or related results from that search
        final resolvedChannel =
            author.toLowerCase().contains(channelName.toLowerCase())
                ? author
                : channelName;
        final publishedAt = video.uploadDate ??
            now.subtract(Duration(hours: fetched.length * 4 + 1));

        fetched[id] = VideoItemModel(
          id: id,
          title: video.title,
          channelName: resolvedChannel,
          thumbnailUrl: video.thumbnails.highResUrl,
          durationSeconds: video.duration?.inSeconds ?? 300,
          publishedAt: publishedAt,
          updatedAt: now,
        );
      }
    } catch (_) {
      // Fallback channel videos if offline
    }

    if (fetched.isEmpty) {
      for (final item in _buildFallbackChannelVideos(channelName, now)) {
        fetched[item.id] = item;
      }
    }

    return fetched.values.toList();
  }

  @override
  Future<String> resolvePlayableStreamUrl(String videoId) async {
    try {
      final manifest =
          await _yt.videos.streamsClient.getManifest(VideoId(videoId));
      if (manifest.muxed.isNotEmpty) {
        final streamInfo = manifest.muxed.withHighestBitrate();
        return streamInfo.url.toString();
      }
      if (manifest.hls.isNotEmpty) {
        return manifest.hls.first.url.toString();
      }
    } catch (_) {
      return 'https://commondatastorage.googleapis.com/gtv-videos-bucket/sample/BigBuckBunny.mp4';
    }
    throw const RemoteSourceException('No playable stream found for video');
  }

  /// Combines related interest channels (subscribed/downloaded) with randomly
  /// picked discovery topics so every refresh adds fresh & relevant channel content.
  List<String> _buildRefreshQueries({
    String? customQuery,
    required List<String> interestedChannels,
  }) {
    if (customQuery != null && customQuery.trim().isNotEmpty) {
      return <String>[customQuery.trim()];
    }

    final queries = <String>[];

    // 1. Add up to 2 related queries based on user's interested/subscribed channels
    if (interestedChannels.isNotEmpty) {
      final shuffledInterests = List<String>.of(interestedChannels)
        ..shuffle(_random);
      for (final channel in shuffledInterests.take(2)) {
        queries.add('$channel latest');
      }
    }

    // 2. Randomly pick 2~3 discovery topics from the diverse pool
    final pool = List<String>.of(AppConstants.discoveryFeedQueries)
      ..shuffle(_random);
    for (final topic in pool) {
      if (queries.length >= 4) {
        break;
      }
      if (!queries.contains(topic)) {
        queries.add(topic);
      }
    }

    return queries;
  }

  List<VideoItemModel> _buildFallbackPreviewCatalog(
    DateTime now, {
    required List<String> interestedChannels,
  }) {
    final allSamples = <(String, String, String, int)>[
      (
        'b_sQ9bMltGU',
        'Flutter Clean Architecture Full Course - Solid Principles in Dart',
        'Flutter Community Dev',
        845,
      ),
      (
        'fq4N0hgOWzU',
        'Google Keynote - Building Multiplatform Apps with Dart & Flutter',
        'Google Developers',
        1260,
      ),
      (
        'wE7khGHVkYY',
        'Mastering Riverpod State Management & Background Tasks in Flutter',
        'Dart Engineering',
        620,
      ),
      (
        '5VbAwh4T1nU',
        'Offline-First Mobile Architecture: SQLite & Queue Scheduling',
        'Mobile System Design',
        930,
      ),
      (
        'ekzHIouo8Q4',
        'High Performance Video Playback & Custom Controls on Android/iOS',
        'Flutter Media Lab',
        515,
      ),
      (
        'aircAruvnKk',
        'But what is a neural network? | Deep Learning Chapter 1',
        '3Blue1Brown',
        1153,
      ),
      (
        'Mus_vwhTCq0',
        'JavaScript & WebAssembly High Speed Runtime Architecture',
        'Fireship Tech',
        410,
      ),
      (
        'rfscVS0vtbw',
        'Clean Code & Refactoring Best Practices for Software Engineers',
        'Clean Coders Studio',
        980,
      ),
      (
        '8jLOx1hD3_o',
        'Next-Gen AI Agents & Autonomous Coding Workflows Explained',
        'AI Research Hub',
        740,
      ),
      (
        'kCc8FmEb1nY',
        'Let\'s build GPT from scratch, in code, spelled out',
        'Andrej Karpathy',
        7020,
      ),
    ];

    final shuffled = List<(String, String, String, int)>.of(allSamples)
      ..shuffle(_random);
    final selected = shuffled.take(6).toList();

    // Also inject a fresh video for an interested channel if any exists
    if (interestedChannels.isNotEmpty) {
      final favChannel =
          interestedChannels[_random.nextInt(interestedChannels.length)];
      final dynamicId =
          'fav_${favChannel.hashCode.abs()}_${now.millisecondsSinceEpoch % 1000}';
      selected.insert(
        0,
        (
          dynamicId,
          '[$favChannel] 最新推薦精選特輯 — 相關興趣頻道新上架',
          favChannel,
          660,
        ),
      );
    }

    return List<VideoItemModel>.generate(selected.length, (index) {
      final (id, title, author, durationSec) = selected[index];
      return VideoItemModel(
        id: id,
        title: title,
        channelName: author,
        thumbnailUrl: 'https://i.ytimg.com/vi/$id/hqdefault.jpg',
        durationSeconds: durationSec,
        publishedAt: now.subtract(Duration(hours: index * 3 + 1)),
        updatedAt: now.subtract(Duration(minutes: index)),
      );
    });
  }

  List<VideoItemModel> _buildFallbackChannelVideos(
    String channelName,
    DateTime now,
  ) {
    final hash = channelName.hashCode.abs();
    return List<VideoItemModel>.generate(4, (i) {
      final id = 'ch_${hash}_ep${i + 1}';
      return VideoItemModel(
        id: id,
        title: '$channelName 頻道精選系列影片 EP.${i + 1}',
        channelName: channelName,
        thumbnailUrl: 'https://i.ytimg.com/vi/fq4N0hgOWzU/hqdefault.jpg',
        durationSeconds: 360 + i * 120,
        publishedAt: now.subtract(Duration(days: i, hours: 2)),
        updatedAt: now,
      );
    });
  }
}
