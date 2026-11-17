/// Application-wide constants for Go-Youtu.
class AppConstants {
  const AppConstants._();

  /// Maximum number of video records stored locally (FIFO eviction when exceeded).
  static const int maxStoredRecords = 10000;

  /// Step duration for fast-forward and rewind controls in fullscreen player.
  static const Duration seekStepDuration = Duration(seconds: 30);

  /// Default search queries used on first launch.
  static const List<String> defaultFeedQueries = <String>[
    'Flutter clean architecture tutorial',
    'Google developers keynote',
    'Dart programming language latest',
  ];

  /// Diverse pool of discovery queries used to randomly inject new & related
  /// channels whenever the user refreshes the video feed.
  static const List<String> discoveryFeedQueries = <String>[
    'Flutter mobile app development',
    'Google AI Gemini developer',
    'Android Kotlin modern architecture',
    'iOS Swift UI engineering',
    'Tech gadget review 2026',
    'Open source software engineering',
    'Cloud native microservices tutorial',
    'Machine learning deepmind lecture',
    'Coding productivity lofi music',
    'System design interview architecture',
    'Cybersecurity explained',
    'WebAssembly & Rust programming',
    'Game engine graphics tech demo',
    'Science & space exploration documentary',
  ];

  /// SQLite database file name.
  static const String databaseFileName = 'go_youtu_records.db';

  /// Directory name under application documents where offline MP4 files are saved.
  static const String offlineVideoDirName = 'offline_videos';
}
