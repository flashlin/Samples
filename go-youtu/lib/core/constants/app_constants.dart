/// Application-wide constants for Go-Youtu.
class AppConstants {
  const AppConstants._();

  /// Maximum number of video records stored locally (FIFO eviction when exceeded).
  static const int maxStoredRecords = 10000;

  /// Step duration for fast-forward and rewind controls on each video card.
  static const Duration seekStepDuration = Duration(seconds: 30);

  /// Default search queries used to populate the initial YouTube preview feed.
  static const List<String> defaultFeedQueries = <String>[
    'Flutter clean architecture tutorial',
    'Google developers keynote',
    'Dart programming language latest',
  ];

  /// SQLite database file name.
  static const String databaseFileName = 'go_youtu_records.db';

  /// Directory name under application documents where offline MP4 files are saved.
  static const String offlineVideoDirName = 'offline_videos';
}
