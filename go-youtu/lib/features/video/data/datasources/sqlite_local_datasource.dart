import 'package:path/path.dart' as p;
import 'package:sqflite/sqflite.dart';

import '../../../../core/constants/app_constants.dart';
import '../../../../core/error/app_exceptions.dart';
import '../models/video_item_model.dart';

/// Contract for local persistence of video records and channel subscriptions.
abstract interface class LocalVideoDataSource {
  Future<List<VideoItemModel>> getAllVideos();
  Future<VideoItemModel?> getVideoById(String id);
  Future<void> upsertVideo(VideoItemModel video);
  Future<void> upsertVideos(List<VideoItemModel> videos);
  Future<Set<String>> getSubscribedChannels();
  Future<void> setChannelSubscription(
    String channelName, {
    required bool isSubscribed,
  });
  Future<int> enforceRecordCap(int maxRecords);
}

/// SQLite implementation of [LocalVideoDataSource] with 10,000 record FIFO eviction
/// and channel subscription persistence.
class SqliteLocalDataSource implements LocalVideoDataSource {
  SqliteLocalDataSource({Database? injectedDatabase})
      : _database = injectedDatabase;

  static const String tableName = 'video_records';
  static const String subscribedChannelsTable = 'subscribed_channels';
  static const int schemaVersion = 2;

  Database? _database;

  Future<Database> get database async {
    if (_database != null) {
      return _database!;
    }
    final dbDir = await getDatabasesPath();
    final dbPath = p.join(dbDir, AppConstants.databaseFileName);
    _database = await openDatabase(
      dbPath,
      version: schemaVersion,
      onCreate: createSchema,
      onUpgrade: upgradeSchema,
    );
    return _database!;
  }

  static Future<void> createSchema(Database db, int version) async {
    await db.execute('''
      CREATE TABLE $tableName (
        id TEXT PRIMARY KEY,
        title TEXT NOT NULL,
        channel_name TEXT NOT NULL,
        thumbnail_url TEXT NOT NULL,
        duration_seconds INTEGER NOT NULL,
        published_at_ms INTEGER NOT NULL,
        updated_at_ms INTEGER NOT NULL,
        download_status TEXT NOT NULL,
        download_progress REAL NOT NULL,
        local_file_path TEXT,
        is_watched INTEGER NOT NULL DEFAULT 0,
        has_ever_downloaded INTEGER NOT NULL DEFAULT 0,
        is_subscribed INTEGER NOT NULL DEFAULT 0,
        last_playback_position_seconds INTEGER NOT NULL DEFAULT 0
      )
    ''');
    await db.execute('''
      CREATE TABLE $subscribedChannelsTable (
        channel_name TEXT PRIMARY KEY,
        subscribed_at_ms INTEGER NOT NULL
      )
    ''');
    await db.execute(
      'CREATE INDEX idx_${tableName}_updated_at ON $tableName (updated_at_ms ASC)',
    );
    await db.execute(
      'CREATE INDEX idx_${tableName}_watched ON $tableName (is_watched)',
    );
    await db.execute(
      'CREATE INDEX idx_${tableName}_subscribed ON $tableName (is_subscribed)',
    );
  }

  static Future<void> upgradeSchema(
    Database db,
    int oldVersion,
    int newVersion,
  ) async {
    if (oldVersion < 2) {
      await db.execute(
        'ALTER TABLE $tableName ADD COLUMN is_subscribed INTEGER NOT NULL DEFAULT 0',
      );
      await db.execute('''
        CREATE TABLE IF NOT EXISTS $subscribedChannelsTable (
          channel_name TEXT PRIMARY KEY,
          subscribed_at_ms INTEGER NOT NULL
        )
      ''');
      await db.execute(
        'CREATE INDEX IF NOT EXISTS idx_${tableName}_subscribed ON $tableName (is_subscribed)',
      );
    }
  }

  @override
  Future<List<VideoItemModel>> getAllVideos() async {
    try {
      final db = await database;
      final rows = await db.query(
        tableName,
        orderBy: 'published_at_ms DESC',
        limit: AppConstants.maxStoredRecords,
      );
      return rows.map(VideoItemModel.fromMap).toList();
    } catch (e) {
      throw LocalStorageException(
        'Failed to read local video records',
        cause: e,
      );
    }
  }

  @override
  Future<VideoItemModel?> getVideoById(String id) async {
    try {
      final db = await database;
      final rows = await db.query(
        tableName,
        where: 'id = ?',
        whereArgs: <Object?>[id],
        limit: 1,
      );
      if (rows.isEmpty) {
        return null;
      }
      return VideoItemModel.fromMap(rows.first);
    } catch (e) {
      throw LocalStorageException('Failed to query video $id', cause: e);
    }
  }

  @override
  Future<void> upsertVideo(VideoItemModel video) async {
    try {
      final db = await database;
      await db.insert(
        tableName,
        video.toMap(),
        conflictAlgorithm: ConflictAlgorithm.replace,
      );
    } catch (e) {
      throw LocalStorageException('Failed to save video ${video.id}', cause: e);
    }
  }

  @override
  Future<void> upsertVideos(List<VideoItemModel> videos) async {
    if (videos.isEmpty) {
      return;
    }
    try {
      final db = await database;
      final batch = db.batch();
      for (final video in videos) {
        batch.insert(
          tableName,
          video.toMap(),
          conflictAlgorithm: ConflictAlgorithm.replace,
        );
      }
      await batch.commit(noResult: true);
    } catch (e) {
      throw LocalStorageException('Failed to batch upsert videos', cause: e);
    }
  }

  @override
  Future<Set<String>> getSubscribedChannels() async {
    try {
      final db = await database;
      final rows = await db.query(subscribedChannelsTable);
      return rows
          .map((r) => (r['channel_name'] as String?) ?? '')
          .where((name) => name.isNotEmpty)
          .toSet();
    } catch (e) {
      throw LocalStorageException(
        'Failed to read subscribed channels',
        cause: e,
      );
    }
  }

  @override
  Future<void> setChannelSubscription(
    String channelName, {
    required bool isSubscribed,
  }) async {
    try {
      final db = await database;
      final nowMs = DateTime.now().millisecondsSinceEpoch;
      await db.transaction((txn) async {
        if (isSubscribed) {
          await txn.insert(
            subscribedChannelsTable,
            <String, Object?>{
              'channel_name': channelName,
              'subscribed_at_ms': nowMs,
            },
            conflictAlgorithm: ConflictAlgorithm.replace,
          );
        } else {
          await txn.delete(
            subscribedChannelsTable,
            where: 'channel_name = ?',
            whereArgs: <Object?>[channelName],
          );
        }

        await txn.update(
          tableName,
          <String, Object?>{
            'is_subscribed': isSubscribed ? 1 : 0,
            'updated_at_ms': nowMs,
          },
          where: 'channel_name = ?',
          whereArgs: <Object?>[channelName],
        );
      });
    } catch (e) {
      throw LocalStorageException(
        'Failed to update subscription for $channelName',
        cause: e,
      );
    }
  }

  @override
  Future<int> enforceRecordCap(int maxRecords) async {
    try {
      final db = await database;
      final countResult =
          await db.rawQuery('SELECT COUNT(*) AS cnt FROM $tableName');
      final totalCount = Sqflite.firstIntValue(countResult) ?? 0;
      if (totalCount <= maxRecords) {
        return 0;
      }

      final overflowCount = totalCount - maxRecords;
      // Evict the oldest records by updated_at_ms that are not actively downloading.
      final deleted = await db.rawDelete(
        '''
        DELETE FROM $tableName
        WHERE id IN (
          SELECT id FROM $tableName
          WHERE download_status NOT IN ('queued', 'downloading')
          ORDER BY updated_at_ms ASC
          LIMIT ?
        )
        ''',
        <Object?>[overflowCount],
      );
      return deleted;
    } catch (e) {
      throw LocalStorageException(
        'Failed to enforce record cap of $maxRecords',
        cause: e,
      );
    }
  }
}
